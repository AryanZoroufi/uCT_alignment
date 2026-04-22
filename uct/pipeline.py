"""
End-to-end pipeline: VOX → mesh → align → segment → atlas-guided labelling → SMC refine.

Steps:
  1. Convert ref.vox and sample.vox to STL  (sigma=3, decimate=0.05, laplacian=100)
  2. Align sample onto ref  (PCA + ICP)
  3. Segment both meshes into disconnected parts  (min_faces=500)
  4. For each part compute: log(volume), PC1/PC2 axis ratio
  5. Match each mesh's segments to the atlas bones via Hungarian assignment
     - atlas bones define which index (1, 2, …) each matched segment gets
     - ref and sample are matched independently, so ref_i and sample_i both
       correspond to atlas bone i
  6. Save atlas-matched segments and all segments:
       <out>/ref_1.stl, ref_2.stl   ← atlas-matched segments from ref
       <out>/sample_1.stl, sample_2.stl  ← atlas-matched segments from sample (pre-SMC)
       <out>/all_segments/ref_part_*.stl
       <out>/all_segments/sample_part_*.stl
  7. SMC aggregate alignment: treat all atlas-matched sample segments as one aggregate,
     find the best rigid+scale transform vs. the ref aggregate, then apply the SAME
     transform to each sample segment individually — preserving inter-bone structure.
     Use --no-smc to skip step 7.
  8. Convex-hull cartilage volume:
       ref_agg_convex  = convex hull of (ref_1 ∪ ref_2)
       ref_cartilage   = V(ref_agg_convex) − V(ref_1_convex) − V(ref_2_convex)
       sample_cartilage = same for sample pair
       injection_volume = ref_cartilage − sample_cartilage  (signed, mm³)
     Saves six convex-hull STLs and injection_volume.txt.

Usage:
    python pipeline.py ref.vox sample.vox bone_1_atlas.stl bone_2_atlas.stl
    python pipeline.py ref.vox sample.vox bone_1_atlas.stl bone_2_atlas.stl -o results/
    python pipeline.py ref.vox sample.vox bone_1_atlas.stl bone_2_atlas.stl --no-smc
"""

import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent))
from vox_to_stl   import vox_to_stl
from align_mesh   import align_mesh
from smc_align    import smc_align_aggregate, smc_align, smc_align_per_bone



# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _split_mesh(stl_path: Path, min_faces: int) -> list[trimesh.Trimesh]:
    """Load STL, merge vertices, return connected components >= min_faces, largest first."""
    mesh = trimesh.load(str(stl_path), force="mesh", process=False)
    mesh.merge_vertices()
    components = mesh.split(only_watertight=False)
    components = sorted(components, key=lambda m: len(m.faces), reverse=True)
    kept = [c for c in components if len(c.faces) >= min_faces]
    print(f"  {len(components)} components, {len(kept)} kept (>= {min_faces} faces)")
    return kept


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _segment_features(seg: trimesh.Trimesh) -> np.ndarray:
    """
    [log_volume, axis_ratio]

    log_volume  : ln of mesh volume (convex hull fallback for non-watertight)
    axis_ratio  : std along PC1 / std along PC2
    """
    vol = float(abs(seg.volume))
    if vol < 1e-12:
        vol = float(seg.convex_hull.volume)
    log_vol = np.log(vol + 1e-12)

    verts  = seg.vertices - seg.vertices.mean(axis=0)
    cov    = (verts.T @ verts) / max(len(verts) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    std1   = np.sqrt(max(float(eigvals[0]), 0.0))
    std2   = np.sqrt(max(float(eigvals[1]), 0.0))
    axis_ratio = std1 / std2 if std2 > 1e-12 else 1.0

    return np.array([log_vol, axis_ratio], dtype=np.float64)


# ---------------------------------------------------------------------------
# Atlas matching
# ---------------------------------------------------------------------------

def _match_to_atlas(
    segs:        list[trimesh.Trimesh],
    atlas_meshes: list[trimesh.Trimesh],
    mesh_label:  str,
) -> tuple[dict[int, trimesh.Trimesh], list[trimesh.Trimesh]]:
    """
    Match segments to atlas bones using Hungarian assignment on
    z-score-normalised [log_volume, axis_ratio] features.

    Parameters
    ----------
    segs         : disconnected segments from one mesh (ref or sample)
    atlas_meshes : list of atlas bones, ordered by desired output index (1-based)
    mesh_label   : 'ref' or 'sample', used only for printing

    Returns
    -------
    matched  : dict { atlas_idx (1-based) → matched segment }
    leftover : segments that were not matched to any atlas bone
    """
    seg_feat   = np.stack([_segment_features(s) for s in segs])
    atlas_feat = np.stack([_segment_features(a) for a in atlas_meshes])

    n_atlas = len(atlas_meshes)

    # Print summary
    print(f"\n  {mesh_label} segments ({len(segs)}):")
    for i, (f, s) in enumerate(zip(seg_feat, segs)):
        print(f"    [{i+1:>3}]  faces={len(s.faces):>8,}  "
              f"log_vol={f[0]:.3f}  axis_ratio={f[1]:.3f}")
    print(f"\n  Atlas bones ({n_atlas}):")
    for i, f in enumerate(atlas_feat):
        print(f"    [{i+1:>3}]  log_vol={f[0]:.3f}  axis_ratio={f[1]:.3f}")

    # z-score normalise jointly so both features contribute equally
    all_feat = np.vstack([seg_feat, atlas_feat])
    mu  = all_feat.mean(axis=0)
    sig = all_feat.std(axis=0) + 1e-10
    seg_norm   = (seg_feat   - mu) / sig
    atlas_norm = (atlas_feat - mu) / sig

    # Cost matrix: (n_segs × n_atlas)
    # linear_sum_assignment on a rectangular (n_segs > n_atlas) matrix picks
    # one segment per atlas bone — exactly what we want.
    cost = cdist(seg_norm, atlas_norm)          # (n_segs, n_atlas)
    seg_idx, atlas_idx = linear_sum_assignment(cost)

    matched  = {}
    used_seg = set()
    print(f"\n  Atlas matching ({mesh_label}):")
    for si, ai in zip(seg_idx, atlas_idx):
        matched[ai + 1] = segs[si]   # 1-based atlas index
        used_seg.add(si)
        print(f"    atlas[{ai+1}] ← {mesh_label}[{si+1}]  "
              f"cost={cost[si, ai]:.4f}")

    leftover = [s for i, s in enumerate(segs) if i not in used_seg]
    return matched, leftover


# ---------------------------------------------------------------------------
# Articular surface distance
# ---------------------------------------------------------------------------

def _articular_region(
    b1: trimesh.Trimesh,
    b2: trimesh.Trimesh,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (articular_mask, facing_score) for bone-2 vertices.

    The articular side of bone 2 is defined geometrically: whichever X-side
    of bone 2 faces bone 1 (determined by comparing X centroids).
    facing_score is normalised to [0, 1] with 1 = most articular (red in viewer).
    articular_mask selects the top `percentile`% by facing_score.
    """
    b1_cx = b1.vertices[:, 0].mean()
    b2_cx = b2.vertices[:, 0].mean()
    x = b2.vertices[:, 0]
    # high facing_score = the X-side of bone 2 that points toward bone 1
    facing_score = x if b1_cx > b2_cx else -x
    cutoff        = np.percentile(facing_score, 100 - percentile)
    articular_mask = facing_score >= cutoff
    norm = (facing_score - facing_score.min()) / max(facing_score.max() - facing_score.min(), 1e-12)
    return articular_mask, norm


def _articular_distance(
    bone2_path: str,
    bone1_path: str,
    percentile: float = 20.0,
    ply_path: str | None = None,
) -> tuple[float, float, float, float]:
    """
    Area-weighted mean nearest-surface distance from bone 2's articular region
    to bone 1, plus the raw articular volume estimate (Σ d·a).

    Vertex areas are distributed from adjacent face areas (1/3 each face),
    so the summary is independent of mesh resolution.

    The articular region is the `percentile`% of bone-2 vertices whose X
    coordinate faces bone 1 (X-centroid comparison).
    Red in the PLY heatmap = articular (facing bone 1), blue = opposite side.

    Returns
    -------
    mean_dist_raw    : area-weighted mean distance in STL coordinate units
    bone2_length     : bone 2 longest PCA diameter (used for normalisation)
    mean_dist_norm   : mean_dist_raw / bone2_length  (dimensionless)
    articular_volume : Σ_{j ∈ articular} d_j · a_j  (units³, ~ volume of gap)
    """
    import matplotlib.cm as cm

    b1 = trimesh.load(bone1_path, force="mesh", process=False)
    b2 = trimesh.load(bone2_path, force="mesh", process=False)
    b1.merge_vertices(); b2.merge_vertices()

    # Point-to-mesh distance: nearest point on any bone-1 face (not just vertices)
    _, dists, _ = trimesh.proximity.closest_point(b1, b2.vertices)

    # Per-vertex area: distribute each face's area equally to its 3 vertices
    vertex_areas = np.zeros(len(b2.vertices), dtype=np.float64)
    face_areas   = b2.area_faces.astype(np.float64)
    np.add.at(vertex_areas, b2.faces[:, 0], face_areas / 3.0)
    np.add.at(vertex_areas, b2.faces[:, 1], face_areas / 3.0)
    np.add.at(vertex_areas, b2.faces[:, 2], face_areas / 3.0)

    # Articular region = X-side of bone 2 facing bone 1
    articular_mask, facing_norm = _articular_region(b1, b2, percentile)
    art_dists  = dists[articular_mask].astype(np.float64)
    art_areas  = vertex_areas[articular_mask]

    # Area-weighted mean distance (independent of mesh resolution)
    total_art_area   = float(art_areas.sum())
    mean_dist        = float(np.sum(art_dists * art_areas) / max(total_art_area, 1e-12))
    # Raw volume estimate: integral of gap depth over articular surface
    articular_volume = float(np.sum(art_dists * art_areas))

    # Bone-2 characteristic length = longest PCA axis diameter
    v       = b2.vertices - b2.vertices.mean(0)
    cov     = (v.T @ v) / max(len(v) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    bone2_length = float(np.sqrt(max(eigvals[-1], 0.0))) * 2

    mean_dist_norm = mean_dist / bone2_length if bone2_length > 1e-12 else mean_dist

    # --- colored PLY export (red = articular/facing bone 1, blue = away) -----
    if ply_path is not None:
        colors = (cm.RdYlBu_r(facing_norm)[:, :3] * 255).astype(np.uint8)
        b2.visual = trimesh.visual.ColorVisuals(mesh=b2, vertex_colors=colors)
        b2.export(ply_path)
        print(f"  Articular heatmap → {ply_path}")

    return mean_dist, bone2_length, mean_dist_norm, articular_volume


# ---------------------------------------------------------------------------
# Rerun visualization
# ---------------------------------------------------------------------------

def _visualize_rerun(out_dir: Path, articular_percentile: float) -> None:
    import rerun as rr
    import matplotlib.cm as cm

    rr.init("uCT_pipeline", spawn=True)

    def _articular_colors(bone2_path: str, bone1_path: str) -> tuple:
        b1 = trimesh.load(bone1_path, force="mesh", process=False)
        b2 = trimesh.load(bone2_path, force="mesh", process=False)
        b1.merge_vertices(); b2.merge_vertices()
        _, facing_norm = _articular_region(b1, b2, percentile=20.0)
        colors = (cm.RdYlBu_r(facing_norm)[:, :3] * 255).astype(np.uint8)
        return b2, colors

    ref_1    = trimesh.load(str(out_dir / "ref_1.stl"),    force="mesh", process=False)
    sample_1 = trimesh.load(str(out_dir / "sample_1.stl"), force="mesh", process=False)
    ref_1.merge_vertices(); sample_1.merge_vertices()
    ref_2,    ref_2_colors    = _articular_colors(str(out_dir / "ref_2.stl"),    str(out_dir / "ref_1.stl"))
    sample_2, sample_2_colors = _articular_colors(str(out_dir / "sample_2.stl"), str(out_dir / "sample_1.stl"))

    grey_ref    = np.full((len(ref_1.vertices),    3), 180, dtype=np.uint8)
    grey_sample = np.full((len(sample_1.vertices), 3), 180, dtype=np.uint8)

    rr.log("ref/bone_1",    rr.Mesh3D(vertex_positions=ref_1.vertices,    triangle_indices=ref_1.faces,    vertex_colors=grey_ref))
    rr.log("ref/bone_2",    rr.Mesh3D(vertex_positions=ref_2.vertices,    triangle_indices=ref_2.faces,    vertex_colors=ref_2_colors))
    rr.log("sample/bone_1", rr.Mesh3D(vertex_positions=sample_1.vertices, triangle_indices=sample_1.faces, vertex_colors=grey_sample))
    rr.log("sample/bone_2", rr.Mesh3D(vertex_positions=sample_2.vertices, triangle_indices=sample_2.faces, vertex_colors=sample_2_colors))

    print("  Rerun viewer launched — close it to continue.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    ref_vox:          str,
    sample_vox:       str,
    atlas_paths:      list[str],
    output_dir:       str | None = None,
    run_smc:              bool = True,
    smc_restarts:         int = 3,
    alpha:                float = 0.0,
    volume_scale:         float = 1.0,
    sigma:                float = 3.0,
    decimate:             float = 0.05,
    taubin_iterations:    int = 100,
    cartilage_threshold:  float = 10.0,
    articular_percentile: float = 20.0,
    visualize:            bool = False,
) -> None:

    ref_vox    = Path(ref_vox)
    sample_vox = Path(sample_vox)
    atlas_paths = [Path(p) for p in atlas_paths]

    out_dir = Path(output_dir) if output_dir \
              else Path(f"{ref_vox.stem}_vs_{sample_vox.stem}")
    all_seg_dir = out_dir / "all_segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_seg_dir.mkdir(exist_ok=True)
    print(f"Output directory: {out_dir}/\n")

    # ------------------------------------------------------------------ 1/5
    print("=" * 60)
    print("STEP 1/5  Convert VOX → STL")
    print("=" * 60)

    ref_stl    = out_dir / f"{ref_vox.stem}.stl"
    sample_stl = out_dir / f"{sample_vox.stem}.stl"

    mesh_kwargs = dict(
        gaussian_sigma    = sigma,
        decimate          = decimate,
        taubin_iterations = taubin_iterations,
        step_size         = 2,
    )

    print(f"\n[ref] {ref_vox.name}")
    vox_to_stl(str(ref_vox), str(ref_stl), **mesh_kwargs)

    print(f"\n[sample] {sample_vox.name}")
    vox_to_stl(str(sample_vox), str(sample_stl), **mesh_kwargs)

    # ------------------------------------------------------------------ 2/5
    print("\n" + "=" * 60)
    print("STEP 2/5  Align sample → ref  (PCA + ICP)")
    print("=" * 60)

    sample_aligned_stl = out_dir / f"{sample_vox.stem}_aligned.stl"
    align_mesh(str(ref_stl), str(sample_stl), str(sample_aligned_stl))

    # ------------------------------------------------------------------ 3/5
    print("\n" + "=" * 60)
    print("STEP 3/5  Segment both meshes")
    print("=" * 60)

    MIN_FACES = 500

    print(f"\n[ref]")
    ref_segs = _split_mesh(ref_stl, MIN_FACES)

    print(f"\n[sample]")
    sample_segs = _split_mesh(sample_aligned_stl, MIN_FACES)

    # ------------------------------------------------------------------ 4/5
    print("\n" + "=" * 60)
    print("STEP 4/5  Load atlas bones & match segments")
    print("=" * 60)

    atlas_meshes = []
    for p in atlas_paths:
        a = trimesh.load(str(p), force="mesh", process=False)
        a.merge_vertices()
        atlas_meshes.append(a)
        print(f"  Loaded atlas: {p.name}  "
              f"({len(a.vertices):,} verts, {len(a.faces):,} faces)")

    ref_matched,    ref_leftover    = _match_to_atlas(ref_segs,    atlas_meshes, "ref")
    sample_matched, sample_leftover = _match_to_atlas(sample_segs, atlas_meshes, "sample")

    # ------------------------------------------------------------------ 5/5
    print("\n" + "=" * 60)
    print("STEP 5/5  Save results")
    print("=" * 60)

    print(f"\n  Atlas-matched segments → {out_dir}/")
    for atlas_i in sorted(ref_matched):
        r = ref_matched[atlas_i]
        s = sample_matched[atlas_i]
        r_path = out_dir / f"ref_{atlas_i}.stl"
        s_path = out_dir / f"sample_{atlas_i}.stl"
        r.export(str(r_path))
        s.export(str(s_path))
        print(f"    {r_path.name} ({len(r.faces):,} faces)  |  "
              f"{s_path.name} ({len(s.faces):,} faces)")

    # --- all segments (including atlas-matched) in all_segments/ ------------
    n_digits = len(str(max(len(ref_segs), len(sample_segs))))
    print(f"\n  All segments → {all_seg_dir}/")

    for i, seg in enumerate(ref_segs, start=1):
        p = all_seg_dir / f"ref_part_{str(i).zfill(n_digits)}.stl"
        seg.export(str(p))
        print(f"    {p.name}  ({len(seg.faces):,} faces)")

    for i, seg in enumerate(sample_segs, start=1):
        p = all_seg_dir / f"sample_part_{str(i).zfill(n_digits)}.stl"
        seg.export(str(p))
        print(f"    {p.name}  ({len(seg.faces):,} faces)")

    # ------------------------------------------------------------------ 7/8
    sorted_atlas = sorted(ref_matched)
    ref_seg_paths    = [out_dir / f"ref_{i}.stl"    for i in sorted_atlas]
    sample_seg_paths = [out_dir / f"sample_{i}.stl" for i in sorted_atlas]

    if run_smc:
        import jax
        gc.collect(); jax.clear_caches()
        print("\n" + "=" * 60)
        print("STEP 7/8  SMC aggregate alignment")
        print("=" * 60)
        print("\n  Treating all atlas-matched sample segments as one aggregate.")
        print("  One shared transform maximises aggregate IoU with ref aggregate.")
        print("  The same transform (with aggregate centroid as pivot) is applied")
        print("  to each sample segment individually → inter-bone structure preserved.\n")

        smc_align_aggregate(
            ref_paths    = [str(p) for p in ref_seg_paths],
            sample_paths = [str(p) for p in sample_seg_paths],
            output_dir   = out_dir,
            seed         = 0,
            n_restarts   = smc_restarts,
        )

        gc.collect(); jax.clear_caches()
        print("\n" + "=" * 60)
        print("STEP 7b/8  SMC per-bone refinement  (x-translation + scale)")
        print("=" * 60)
        print("\n  Refines each bone pair individually after aggregate alignment.")
        print("  Only x-translation and per-axis scale are searched;")
        print("  rotations and y/z translation are locked to preserve structure.\n")

        smc_align_per_bone(
            ref_paths    = [str(p) for p in ref_seg_paths],
            sample_paths = [str(p) for p in sample_seg_paths],
            output_dir   = out_dir,
            seed         = 0,
            n_restarts   = smc_restarts,
        )
    else:
        print("\n  (SMC step skipped — use without --no-smc to enable)")

    # ------------------------------------------------------------------ 8/8
    print("\n" + "=" * 60)
    print("STEP 8/8  Injection score  (articular surface distance)")
    print("=" * 60)
    print(f"\n  Method: area-weighted mean nearest-surface distance from bone-2")
    print(f"  articular region (top {articular_percentile:.0f}% by X-facing score) to bone-1.")
    print(f"  Volume estimate = Σ d·a over articular surface (units³ × volume_scale).")
    print(f"  Positive score = sample joint gap narrower than ref = bone grew in.\n")

    ref_dist,    ref_b2_len,    ref_dist_norm,    ref_art_vol    = _articular_distance(
        str(out_dir / "ref_2.stl"),    str(out_dir / "ref_1.stl"),    articular_percentile,
        ply_path=str(out_dir / "ref_2_articular.ply"))
    sample_dist, sample_b2_len, sample_dist_norm, sample_art_vol = _articular_distance(
        str(out_dir / "sample_2.stl"), str(out_dir / "sample_1.stl"), articular_percentile,
        ply_path=str(out_dir / "sample_2_articular.ply"))

    injection_volume_raw  = (ref_art_vol   - sample_art_vol)  * volume_scale
    injection_norm        = ref_dist_norm  - sample_dist_norm

    print(f"  ref    area-weighted dist: {ref_dist:.4f}  articular vol: {ref_art_vol:.2f}"
          f"  (bone2 length: {ref_b2_len:.1f})  → normalised: {ref_dist_norm:.6f}")
    print(f"  sample area-weighted dist: {sample_dist:.4f}  articular vol: {sample_art_vol:.2f}"
          f"  (bone2 length: {sample_b2_len:.1f})  → normalised: {sample_dist_norm:.6f}")
    print(f"\n  Injection volume (Σd·a, scaled): {injection_volume_raw:+.6f}  ← primary metric")
    print(f"  Injection score (normalised):    {injection_norm:+.6f}")

    # ── COMMENTED OUT: volume-based methods ──────────────────────────────────
    # # EDT cartilage volume delta (noisy due to biological variability):
    # import jax; gc.collect(); jax.clear_caches()
    # ref_cartilage_aligned_path = out_dir / "ref_cartilage_aligned.stl"
    # smc_align(ref_path=str(sample_cartilage_path), sample_path=str(ref_cartilage_path),
    #           output_path=str(ref_cartilage_aligned_path), seed=0, n_restarts=smc_restarts)
    # ref_cart_aligned = trimesh.load(str(ref_cartilage_aligned_path), force="mesh", process=False)
    # sample_cart = trimesh.load(str(sample_cartilage_path), force="mesh", process=False)
    # ref_cart_aligned.merge_vertices(); sample_cart.merge_vertices()
    # injection_volume = (abs(ref_cart_aligned.volume) - abs(sample_cart.volume)) * volume_scale
    #
    # # Simple V(sample_2) - V(ref_2):
    # ref_2_vol    = abs(trimesh.load(str(out_dir/"ref_2.stl"),    force="mesh").volume) * volume_scale
    # sample_2_vol = abs(trimesh.load(str(out_dir/"sample_2.stl"), force="mesh").volume) * volume_scale
    # simple_injection = sample_2_vol - ref_2_vol
    # ── END commented methods ─────────────────────────────────────────────────

    # Save results
    inj_txt = out_dir / "injection_volume.txt"
    inj_txt.write_text(
        f"# articular surface distance method (area-weighted)\n"
        f"# positive = bone grew into joint in sample\n"
        f"\n"
        f"articular_percentile:           {articular_percentile}\n"
        f"volume_scale:                   {volume_scale}\n"
        f"\n"
        f"ref_area_weighted_dist:         {ref_dist:.6f}\n"
        f"sample_area_weighted_dist:      {sample_dist:.6f}\n"
        f"ref_articular_volume:           {ref_art_vol:.6f}\n"
        f"sample_articular_volume:        {sample_art_vol:.6f}\n"
        f"ref_bone2_length:               {ref_b2_len:.6f}\n"
        f"sample_bone2_length:            {sample_b2_len:.6f}\n"
        f"ref_dist_normalized:            {ref_dist_norm:.6f}\n"
        f"sample_dist_normalized:         {sample_dist_norm:.6f}\n"
        f"\n"
        f"injection_volume_raw:           {injection_volume_raw:+.6f}  # (ref - sample) articular vol × scale\n"
        f"injection_score_normalized:     {injection_norm:+.6f}  # dimensionless distance ratio\n"
    )
    print(f"\n  Saved {inj_txt.name}")

    if visualize:
        print("\n" + "=" * 60)
        print("Visualization  (rerun-sdk)")
        print("=" * 60)
        _visualize_rerun(out_dir, articular_percentile)

    print(f"\nDone.")
    print(f"  {len(ref_matched)} matched pairs  → {out_dir}/")
    print(f"  {len(ref_segs) + len(sample_segs)} total segments → {all_seg_dir}/")
    print(f"  Injection volume (scaled): {injection_volume_raw:+.6f}  → {inj_txt}")
    print(f"  Injection score (norm'd):  {injection_norm:+.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VOX → mesh → align → segment → atlas-guided labelling."
    )
    parser.add_argument("ref",    help="Reference VOX file")
    parser.add_argument("sample", help="Sample VOX file")
    parser.add_argument(
        "atlas", nargs="+",
        help="Atlas STL files in order (e.g. bone_1_atlas.stl bone_2_atlas.stl). "
             "Output index matches this order.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: <ref>_vs_<sample>/)",
    )
    parser.add_argument(
        "--no-smc", action="store_true",
        help="Skip SMC aggregate alignment step (step 7/8)",
    )
    parser.add_argument(
        "--smc-restarts", type=int, default=3, metavar="N",
        help="Number of IS restarts for SMC alignment (default: 3)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, metavar="A",
        help="Alpha value for alpha-shape cartilage volume (default: 0 = convex hull). "
             "Increase for tighter fit around bone surfaces.",
    )
    parser.add_argument(
        "--volume-scale", type=float, default=1.0, metavar="S",
        help="Multiply all cartilage/injection volumes by this factor. "
             "Use 1e-3 for cm³, 1e-9 for m³ (default: 1.0 = raw mm³). "
             "Check the voxel size printed in step 8 to verify STL units.",
    )
    parser.add_argument(
        "--sigma", type=float, default=3.0, metavar="S",
        help="Gaussian pre-smooth σ in voxels for VOX→STL (default: 3.0; 0 to disable)",
    )
    parser.add_argument(
        "--decimate", type=float, default=0.05, metavar="R",
        help="Fraction of faces to keep after decimation (default: 0.05 = keep 5%%)",
    )
    parser.add_argument(
        "--laplacian", type=int, default=100, metavar="N",
        help="Taubin smoothing iterations for VOX→STL (default: 100; 0 to disable)",
    )
    parser.add_argument(
        "--cartilage-threshold", type=float, default=10.0, metavar="T",
        help="Dual-EDT threshold in voxels: a voxel is joint space if its distance "
             "to both bone surfaces is less than this value (default: 10)",
    )
    parser.add_argument(
        "--articular-percentile", type=float, default=20.0, metavar="P",
        help="Bottom P%% of bone-2 vertices (by distance to bone-1) define the "
             "articular region for the injection score (default: 20)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch rerun-sdk viewer showing ref_1, ref_2, sample_1, sample_2 "
             "with bone-2 colored by distance to bone-1 (red=close, blue=far).",
    )
    args = parser.parse_args()
    run_pipeline(args.ref, args.sample, args.atlas, args.output_dir,
                 run_smc=not args.no_smc, smc_restarts=args.smc_restarts,
                 alpha=args.alpha, volume_scale=args.volume_scale,
                 sigma=args.sigma, decimate=args.decimate,
                 taubin_iterations=args.laplacian,
                 cartilage_threshold=args.cartilage_threshold,
                 articular_percentile=args.articular_percentile,
                 visualize=args.visualize)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        print("Usage: python pipeline.py ref.vox sample.vox "
              "bone_1_atlas.stl bone_2_atlas.stl [-o results/]")
        sys.exit(0)
    main()
