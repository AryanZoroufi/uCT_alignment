"""
Align a sample mesh to a reference mesh.

Pipeline:
  1. PCA  — uniform scale + coarse rotation + translation
            Sign of each axis fixed by skewness (asymmetric statistic,
            unlike std which cannot distinguish 0° from 180°)
  2. ICP  — iterative closest point refinement; corrects any residual
            rotation/translation left by PCA, including any 180° flip
            that skewness couldn't resolve (e.g. near-symmetric bones)

Usage:
    python align_mesh.py ref.stl sample.stl
    python align_mesh.py ref.stl sample.stl -o aligned.stl
    python align_mesh.py ref.stl sample.stl --no-icp   # PCA only

Dependencies:
    pip install trimesh numpy
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def _area_weighted_centroid(mesh: trimesh.Trimesh) -> np.ndarray:
    """Surface centroid weighted by triangle area."""
    areas = mesh.area_faces
    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    return (face_centroids * areas[:, None]).sum(axis=0) / areas.sum()


def _pca_axes(mesh: trimesh.Trimesh, centroid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Area-weighted PCA of vertex positions.

    Returns
    -------
    axes   : (3, 3) — rows are principal axes, largest variance first
    stddevs: (3,)   — std along each axis
    """
    verts = mesh.vertices - centroid

    vertex_areas = np.zeros(len(verts))
    np.add.at(vertex_areas, mesh.faces[:, 0], mesh.area_faces / 3)
    np.add.at(vertex_areas, mesh.faces[:, 1], mesh.area_faces / 3)
    np.add.at(vertex_areas, mesh.faces[:, 2], mesh.area_faces / 3)
    w = vertex_areas / vertex_areas.sum()

    cov = (verts * w[:, None]).T @ verts

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    axes    = eigenvectors.T
    stddevs = np.sqrt(np.maximum(eigenvalues, 0))
    return axes, stddevs


def _canonical_axes(verts_centred: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """
    Fix the sign of each principal axis using skewness (third moment).

    Skewness is asymmetric: skew(-x) = -skew(x), so flipping an axis
    changes the sign of the skewness. We choose the direction that gives
    positive skewness, giving a canonical orientation that is stable under
    180° rotations.

    If skewness is near zero (symmetric bone along that axis) the sign
    choice is arbitrary — ICP will correct it in the next step.
    """
    canonical = axes.copy()
    for i in range(3):
        proj = verts_centred @ axes[i]
        skewness = float(np.mean(proj ** 3)) / (float(np.std(proj)) ** 3 + 1e-10)
        if skewness < 0:
            canonical[i] *= -1

    # Guarantee a proper rotation matrix (det = +1, not a reflection)
    if np.linalg.det(canonical) < 0:
        canonical[2] *= -1   # flip the least-variance axis

    return canonical


# ---------------------------------------------------------------------------
# In-memory alignment (trimesh objects in, trimesh object out)
# ---------------------------------------------------------------------------

def align_trimesh(
    ref: trimesh.Trimesh,
    sample: trimesh.Trimesh,
    use_icp: bool = True,
    icp_max_iter: int = 80,
    icp_samples: int = 10_000,
) -> trimesh.Trimesh:
    """
    Align sample onto ref in memory and return the aligned copy.
    Same PCA + ICP pipeline as align_mesh() but without any file I/O.
    """
    ref_centroid    = _area_weighted_centroid(ref)
    sample_centroid = _area_weighted_centroid(sample)

    ref_axes,    ref_stddevs    = _pca_axes(ref,    ref_centroid)
    sample_axes, sample_stddevs = _pca_axes(sample, sample_centroid)

    uniform_scale = ref_stddevs.mean() / sample_stddevs.mean()

    ref_canon    = _canonical_axes(ref.vertices    - ref_centroid,    ref_axes)
    sample_canon = _canonical_axes(sample.vertices - sample_centroid, sample_axes)

    R = ref_canon.T @ sample_canon

    T_pca = np.eye(4)
    T_pca[:3, :3] = R * uniform_scale
    T_pca[:3,  3] = ref_centroid - R @ (sample_centroid * uniform_scale)

    aligned = sample.copy()
    aligned.apply_transform(T_pca)

    if use_icp:
        rng = np.random.default_rng(0)
        n_ref  = min(icp_samples, len(ref.vertices))
        n_samp = min(icp_samples, len(aligned.vertices))
        ref_pts    = ref.vertices[rng.choice(len(ref.vertices),     n_ref,  replace=False)]
        sample_pts = aligned.vertices[rng.choice(len(aligned.vertices), n_samp, replace=False)]
        T_icp, _, _ = trimesh.registration.icp(
            sample_pts, ref_pts,
            max_iterations=icp_max_iter, threshold=1e-6,
            reflection=False, scale=False,
        )
        aligned.apply_transform(T_icp)

    return aligned


# ---------------------------------------------------------------------------
# File-based alignment (reads STL files, saves result to disk)
# ---------------------------------------------------------------------------

def align_mesh(
    ref_path: str,
    sample_path: str,
    output_path: str | None = None,
    use_icp: bool = True,
    icp_max_iter: int = 80,
    icp_samples: int = 10_000,
) -> None:
    """
    Parameters
    ----------
    ref_path, sample_path : input STL files
    output_path           : where to save the aligned sample (default: <sample>_aligned.stl)
    use_icp               : run ICP refinement after PCA (default: True)
    icp_max_iter          : maximum ICP iterations
    icp_samples           : number of surface points sampled for ICP
                            (subsampling keeps ICP fast on large meshes)
    """
    ref_path    = Path(ref_path)
    sample_path = Path(sample_path)
    out_path    = Path(output_path) if output_path else \
                  sample_path.parent / f"{sample_path.stem}_aligned.stl"

    print(f"Loading reference: {ref_path}")
    ref = trimesh.load(str(ref_path), force="mesh", process=False)
    ref.merge_vertices()
    print(f"  Vertices: {len(ref.vertices):,}   Faces: {len(ref.faces):,}")

    print(f"Loading sample:    {sample_path}")
    sample = trimesh.load(str(sample_path), force="mesh", process=False)
    sample.merge_vertices()
    print(f"  Vertices: {len(sample.vertices):,}   Faces: {len(sample.faces):,}")

    # -------------------------------------------------------------------------
    # Step 1 — PCA coarse alignment
    # -------------------------------------------------------------------------
    print("\n[1/2] PCA alignment ...")

    ref_centroid    = _area_weighted_centroid(ref)
    sample_centroid = _area_weighted_centroid(sample)

    ref_axes,    ref_stddevs    = _pca_axes(ref,    ref_centroid)
    sample_axes, sample_stddevs = _pca_axes(sample, sample_centroid)

    uniform_scale = ref_stddevs.mean() / sample_stddevs.mean()
    print(f"  Ref    stddevs (mm): {np.round(ref_stddevs, 3)}")
    print(f"  Sample stddevs (mm): {np.round(sample_stddevs, 3)}")
    print(f"  Uniform scale:       {uniform_scale:.6f}")

    # Fix axis signs with skewness so each mesh has a canonical orientation
    ref_canon    = _canonical_axes(ref.vertices    - ref_centroid,    ref_axes)
    sample_canon = _canonical_axes(sample.vertices - sample_centroid, sample_axes)

    # Rotation that maps sample canonical frame → ref canonical frame
    R = ref_canon.T @ sample_canon

    # Build 4×4: scale → rotate → translate to ref centroid
    T_pca = np.eye(4)
    T_pca[:3, :3] = R * uniform_scale
    T_pca[:3,  3] = ref_centroid - R @ (sample_centroid * uniform_scale)

    aligned = sample.copy()
    aligned.apply_transform(T_pca)

    pca_centroid_err = np.linalg.norm(_area_weighted_centroid(aligned) - ref_centroid)
    print(f"  Centroid error after PCA: {pca_centroid_err:.4f} mm")

    # -------------------------------------------------------------------------
    # Step 2 — ICP refinement (rotation + translation only, no further scaling)
    # -------------------------------------------------------------------------
    if not use_icp:
        print("\n  Skipping ICP (--no-icp)")
    else:
        print(f"\n[2/2] ICP refinement  "
              f"(max_iter={icp_max_iter}, samples={icp_samples:,}) ...")

        # Subsample surface points for speed — ICP result is applied to all vertices
        n_ref  = min(icp_samples, len(ref.vertices))
        n_samp = min(icp_samples, len(aligned.vertices))

        rng = np.random.default_rng(0)
        ref_pts     = ref.vertices[rng.choice(len(ref.vertices),     n_ref,  replace=False)]
        sample_pts  = aligned.vertices[rng.choice(len(aligned.vertices), n_samp, replace=False)]

        # trimesh ICP: finds T such that T @ sample_pts ≈ ref_pts
        T_icp, _, cost = trimesh.registration.icp(
            sample_pts,
            ref_pts,
            max_iterations = icp_max_iter,
            threshold      = 1e-6,
            reflection     = False,
            scale          = False,
        )
        aligned.apply_transform(T_icp)

        final_centroid_err = np.linalg.norm(_area_weighted_centroid(aligned) - ref_centroid)
        print(f"  ICP final cost:          {cost:.6f}")
        print(f"  Centroid error after ICP: {final_centroid_err:.4f} mm")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    aligned.export(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA + ICP alignment of a sample mesh onto a reference mesh."
    )
    parser.add_argument("ref",    help="Reference STL file")
    parser.add_argument("sample", help="Sample STL file to align")
    parser.add_argument(
        "-o", "--output",
        help="Output path (default: <sample>_aligned.stl)",
    )
    parser.add_argument(
        "--no-icp", action="store_true",
        help="Skip ICP refinement, use PCA only",
    )
    parser.add_argument(
        "--icp-iter", type=int, default=80,
        help="Max ICP iterations (default: 80)",
    )
    parser.add_argument(
        "--icp-samples", type=int, default=10_000,
        help="Surface points sampled per mesh for ICP (default: 10000)",
    )

    args = parser.parse_args()
    align_mesh(
        args.ref, args.sample, args.output,
        use_icp      = not args.no_icp,
        icp_max_iter = args.icp_iter,
        icp_samples  = args.icp_samples,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        print("Usage: python align_mesh.py ref.stl sample.stl [-o aligned.stl]")
        sys.exit(0)
    main()
