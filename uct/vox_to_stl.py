"""
Convert a Rigaku CT scanner .vox file to 3D surface mesh(es) (.stl).

File format (Rigaku "Vox1999a"):
  - Text header terminated by ##\\x0c\\n
  - Binary body: uint16 little-endian voxels, shape (Z, Y, X)
  - HU = raw_uint16 * Scale + Offset  (from header Field line)

Pipeline:
  1. Parse Rigaku VOX header + binary body → HU volume (float32)
  2. (Optional) Gaussian smooth to reduce staircase artefacts
  3. Extract bone isosurface with Marching Cubes (HU > bone_threshold)
  4. (Optional) Extract soft tissue band (bg_threshold < HU < bone_threshold)
  5. (Optional) Laplacian smooth each mesh
  6. Write binary STL(s) via numpy-stl

Dependencies:
    pip install numpy scikit-image numpy-stl scipy
"""

import re
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import sparse
from skimage import measure
from skimage.filters import threshold_otsu
from stl import mesh as stl_mesh


# ---------------------------------------------------------------------------
# Rigaku VOX parser
# ---------------------------------------------------------------------------

def load_vox(path: str) -> tuple[np.ndarray, float]:
    """
    Read a Rigaku 'Vox1999a' .vox file.

    Returns
    -------
    grid : float32 ndarray of shape (Z, Y, X) in Hounsfield Units
    voxel_size_mm : isotropic voxel edge length in mm
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        raw = f.read()

    # --- locate header / binary boundary ------------------------------------
    sep = b"##\x0c\n"
    last_sep = raw.rfind(sep)
    if last_sep == -1:
        raise ValueError("Cannot find header terminator (##\\x0c\\n). "
                         "Is this a Rigaku VOX file?")
    header_text = raw[:last_sep + len(sep)].decode("latin-1")
    data_offset = last_sep + len(sep)

    # --- parse header fields ------------------------------------------------
    def _get(pattern, text, cast=str, required=True):
        m = re.search(pattern, text)
        if m is None:
            if required:
                raise ValueError(f"Header field not found: {pattern!r}")
            return None
        return cast(m.group(1))

    endian_char = "<" if _get(r"Endian\s+(\w)", header_text) == "L" else ">"
    sx, sy, sz = map(int, re.search(
        r"VolumeSize\s+(\d+)\s+(\d+)\s+(\d+)", header_text).groups())
    voxel_bits = _get(r"VoxelSize\s+(\d+)", header_text, int)

    # VolumeScale is in cm — convert to mm
    vscale_str = re.search(
        r"VolumeScale\s+([\d.eE+\-]+)", header_text).group(1)
    voxel_size_mm = float(vscale_str) * 10.0   # cm → mm

    # Field: Scale and Offset for HU conversion
    field_m = re.search(
        r'Scale\s+([\d.eE+\-]+)\s+Offset\s+([\d.eE+\-]+)', header_text)
    if field_m:
        hu_scale  = float(field_m.group(1))
        hu_offset = float(field_m.group(2))
    else:
        hu_scale, hu_offset = 1.0, 0.0

    print(f"  Grid size:      {sx} x {sy} x {sz}  (X x Y x Z)")
    print(f"  Voxel size:     {voxel_size_mm:.4f} mm")
    print(f"  HU conversion:  raw × {hu_scale} + ({hu_offset})")

    # --- read binary voxel data ---------------------------------------------
    dtype = np.dtype(f"{endian_char}u{voxel_bits // 8}")
    n_expected = sx * sy * sz
    body = np.frombuffer(raw, dtype=dtype, offset=data_offset)

    if body.size != n_expected:
        raise ValueError(
            f"Expected {n_expected:,} voxels, got {body.size:,}. "
            "Check VolumeSize in header.")

    grid_raw = body.reshape((sz, sy, sx))
    grid = grid_raw.astype(np.float32) * hu_scale + hu_offset

    hu_min, hu_max = float(grid.min()), float(grid.max())
    print(f"  HU range:       {hu_min:.0f} – {hu_max:.0f}")

    return grid, voxel_size_mm


# ---------------------------------------------------------------------------
# Mesh extraction helpers
# ---------------------------------------------------------------------------

def _mesh_from_volume(vol: np.ndarray, level: float, step_size: int,
                      voxel_size_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Run marching cubes on vol and return (verts_mm, faces)."""
    verts, faces, _normals, _ = measure.marching_cubes(
        vol,
        level=level,
        step_size=step_size,
        allow_degenerate=False,
        method="lewiner",
    )
    verts = verts * (voxel_size_mm * step_size)
    return verts, faces


def _save_stl(verts: np.ndarray, faces: np.ndarray, path: str) -> None:
    """Write a binary STL file from vertices and faces."""
    surface = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
    surface.vectors[:] = verts[faces]
    surface.save(path)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  Saved: {path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def vox_to_stl(
    vox_path: str,
    stl_path: str,
    *,
    iso_level: float | None = None,
    soft_tissue_stl: str | None = None,
    soft_tissue_lower: float | None = None,
    gaussian_sigma: float = 1.0,
    taubin_iterations: int = 30,
    decimate: float | None = None,
    step_size: int = 2,
) -> None:
    """
    Full pipeline: VOX → smoothed volume → marching cubes → STL(s).

    Parameters
    ----------
    vox_path             : input .vox file
    stl_path             : output STL for bone (HU > iso_level)
    iso_level            : HU bone threshold. None → Otsu auto-detect.
    soft_tissue_stl      : if given, also extract soft tissue and save here
    soft_tissue_lower    : lower HU bound for soft tissue band.
                           None → second Otsu on sub-bone voxels.
    gaussian_sigma       : Gaussian pre-smooth σ in voxels (0 = off).
    taubin_iterations    : Taubin smoothing passes (0 = off). Default 10.
    decimate             : fraction of faces to keep after decimation, e.g. 0.1.
                           None = no decimation. Applied to bone mesh only.
    step_size            : marching-cubes step; 1=full res, 2=default, 3-4=preview.
    """

    print(f"\n[1/5] Loading {vox_path}")
    grid, voxel_size_mm = load_vox(vox_path)

    # --- auto bone threshold (Otsu on full volume) ---------------------------
    if iso_level is None:
        sub = grid[::4, ::4, ::4].ravel()
        iso_level = float(threshold_otsu(sub))
        print(f"  Bone iso_level  = {iso_level:.1f} HU  (Otsu)")
    else:
        print(f"  Bone iso_level  = {iso_level:.1f} HU")

    # --- auto soft-tissue lower bound (Otsu on sub-bone voxels) -------------
    if soft_tissue_stl is not None and soft_tissue_lower is None:
        sub_below = grid[::4, ::4, ::4].ravel()
        sub_below = sub_below[sub_below < iso_level]
        if sub_below.size > 0:
            soft_tissue_lower = float(threshold_otsu(sub_below))
            print(f"  Soft tissue lower = {soft_tissue_lower:.1f} HU  (Otsu on sub-bone voxels)")
        else:
            soft_tissue_lower = float(grid.min())
            print(f"  Soft tissue lower = {soft_tissue_lower:.1f} HU  (volume minimum)")

    # --- Gaussian pre-smooth ------------------------------------------------
    if gaussian_sigma > 0:
        print(f"\n[2/5] Gaussian smoothing (σ={gaussian_sigma} voxels)")
        grid_smooth = gaussian_filter(grid, sigma=gaussian_sigma)
    else:
        print("\n[2/5] Skipping Gaussian smoothing")
        grid_smooth = grid

    # --- Bone mesh ----------------------------------------------------------
    print(f"\n[3/5] Marching Cubes — bone  (step_size={step_size}, "
          f"res={voxel_size_mm * step_size:.3f} mm)")
    bone_verts, bone_faces = _mesh_from_volume(
        grid_smooth, iso_level, step_size, voxel_size_mm)
    print(f"  Vertices: {len(bone_verts):,}   Faces: {len(bone_faces):,}")

    # --- Soft tissue mesh ---------------------------------------------------
    if soft_tissue_stl is not None:
        print(f"\n[3b] Marching Cubes — soft tissue  "
              f"(band: {soft_tissue_lower:.1f} – {iso_level:.1f} HU)")
        # Build a binary mask for the band, smooth it, then run marching cubes
        band_mask = np.where(
            (grid >= soft_tissue_lower) & (grid < iso_level),
            1.0, 0.0
        ).astype(np.float32)
        if gaussian_sigma > 0:
            band_mask = gaussian_filter(band_mask, sigma=gaussian_sigma)
        st_verts, st_faces = _mesh_from_volume(
            band_mask, 0.5, step_size, voxel_size_mm)
        print(f"  Vertices: {len(st_verts):,}   Faces: {len(st_faces):,}")

    # --- Decimation (bone only) — done first so smoothing acts on final mesh --
    if decimate is not None:
        print(f"\n[4/5] Decimating bone mesh (keeping {decimate*100:.0f}% of faces)")
        before = len(bone_faces)
        bone_verts, bone_faces = _decimate(bone_verts, bone_faces, decimate)
        print(f"  Faces: {before:,} → {len(bone_faces):,}")

    # --- Taubin smoothing (after decimation so we smooth the final mesh) ----
    if taubin_iterations > 0:
        print(f"\n[4b] Taubin smoothing ({taubin_iterations} iterations)")
        print("  bone...")
        bone_verts = _taubin_smooth(bone_verts, bone_faces, taubin_iterations)
        if soft_tissue_stl is not None:
            print("  soft tissue...")
            st_verts = _taubin_smooth(st_verts, st_faces, taubin_iterations)
    else:
        print("\n[4b] Skipping Taubin smoothing")

    # --- Write STL(s) -------------------------------------------------------
    print(f"\n[5/5] Writing STL(s)")
    _save_stl(bone_verts, bone_faces, stl_path)
    if soft_tissue_stl is not None:
        _save_stl(st_verts, st_faces, soft_tissue_stl)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Taubin mesh smoother (sparse matrix — no shrinkage)
# ---------------------------------------------------------------------------

def _taubin_smooth(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """
    Taubin (1995) smoothing: alternates a positive pass (λ) and a negative
    pass (μ) so the mesh doesn't shrink, unlike plain Laplacian smoothing.
    """
    n = len(verts)
    i0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    i1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(i0), dtype=np.float32)
    A = sparse.csr_matrix((data, (i0, i1)), shape=(n, n))

    row_sums = np.asarray(A.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    L = sparse.diags(1.0 / row_sums) @ A   # row-normalised averaging matrix

    v = verts.copy()
    for k in range(iterations):
        factor = lam if k % 2 == 0 else mu
        v = v + factor * ((L @ v) - v)
        if (k + 1) % 10 == 0:
            print(f"    iteration {k + 1}/{iterations}")
    return v


# ---------------------------------------------------------------------------
# Mesh decimation (quadric error — preserves shape, removes spikes)
# ---------------------------------------------------------------------------

def _decimate(verts: np.ndarray, faces: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce mesh complexity via quadric error decimation.
    ratio: fraction of faces to keep, e.g. 0.1 keeps 10%.
    """
    import fast_simplification
    target_reduction = float(np.clip(1.0 - ratio, 0.0, 0.99))
    out_verts, out_faces = fast_simplification.simplify(
        verts.astype(np.float64),
        faces.astype(np.int32),
        target_reduction=target_reduction,
    )
    return out_verts.astype(np.float32), out_faces


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Rigaku CT scanner .vox file to STL mesh(es)."
    )
    parser.add_argument("vox_file", help="Path to input .vox file")
    parser.add_argument("stl_file", nargs="?", default=None,
                        help="Output STL for bone (default: input name + .stl)")
    parser.add_argument(
        "--iso-level", type=float, default=None,
        help="HU bone threshold (default: Otsu auto-detect)",
    )
    parser.add_argument(
        "--soft-tissue-stl", metavar="PATH",
        help="If given, also extract soft tissue band and save to this STL",
    )
    parser.add_argument(
        "--soft-tissue-lower", type=float, default=None,
        metavar="HU",
        help="Lower HU bound for soft tissue band "
             "(default: second Otsu on sub-bone voxels)",
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0,
        help="Gaussian pre-smooth σ in voxels; 0 to disable (default: 1.0)",
    )
    parser.add_argument(
        "--laplacian", type=int, default=10,
        help="Taubin smoothing iterations; 0 to disable (default: 30)",
    )
    parser.add_argument(
        "--decimate", type=float, default=None, metavar="RATIO",
        help="Fraction of bone faces to keep via quadric decimation, e.g. 0.1 "
             "keeps 10%% of triangles. Removes spikes and simplifies mesh. "
             "Default: no decimation.",
    )
    parser.add_argument(
        "--step-size", type=int, default=2,
        help="Marching-cubes step size; 1=full res, 2=default, 3-4=preview",
    )

    args = parser.parse_args()

    vox_path = args.vox_file
    stl_path = args.stl_file or str(Path(vox_path).with_suffix(".stl"))

    vox_to_stl(
        vox_path,
        stl_path,
        iso_level=args.iso_level,
        soft_tissue_stl=args.soft_tissue_stl,
        soft_tissue_lower=args.soft_tissue_lower,
        gaussian_sigma=args.sigma,
        taubin_iterations=args.laplacian,
        decimate=args.decimate,
        step_size=args.step_size,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        print("Usage: python vox_to_stl.py <file.vox> [<output.stl>] [options]")
        print("       python vox_to_stl.py --help")
        sys.exit(0)
    main()
