"""
Compute the joint-space volume trapped between two bone meshes.

Math / algorithm
----------------
  For each column along the X axis (i.e. each (y, z) pair):
    - x_b2_max  = highest x voxel occupied by bone-2 in that column
    - x_b1_min  = lowest  x voxel occupied by bone-1 in that column
    - If x_b2_max < x_b1_min: fill every voxel between them → joint space
    - Otherwise (bones overlap in this column): skip

  Assumption: bone-2 lies generally in the -x direction, bone-1 in +x.
  Swap --bone1 / --bone2 if your geometry is reversed.

Usage:
    python joint_space.py ref_1.stl ref_2.stl -o ref_c.stl
    python joint_space.py ref_1.stl ref_2.stl --pitch 0.5   # finer
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import label as nd_label, distance_transform_edt
from skimage.measure import marching_cubes

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from vox_to_stl import _decimate, _taubin_smooth


# ---------------------------------------------------------------------------
# Bone voxelisation onto a shared grid (works for non-watertight meshes)
# ---------------------------------------------------------------------------

def _bone_mask(mesh: trimesh.Trimesh,
               vmin: np.ndarray,
               shape: tuple,
               pitch: float) -> np.ndarray:
    vox    = trimesh.voxel.creation.voxelize(mesh, pitch)
    mat    = vox.matrix
    origin = vox.transform[:3, 3]
    offset = np.round((origin - vmin) / pitch).astype(int)

    result = np.zeros(shape, dtype=bool)
    s, sh  = np.array(mat.shape), np.array(shape)

    src_lo = np.maximum(0, -offset)
    src_hi = np.minimum(s, sh - offset)
    if np.any(src_hi <= src_lo):
        return result

    dst_lo = offset + src_lo
    dst_hi = offset + src_hi
    result[dst_lo[0]:dst_hi[0],
           dst_lo[1]:dst_hi[1],
           dst_lo[2]:dst_hi[2]] = mat[src_lo[0]:src_hi[0],
                                       src_lo[1]:src_hi[1],
                                       src_lo[2]:src_hi[2]]
    return result


# ---------------------------------------------------------------------------
# X-axis outlier filter
# ---------------------------------------------------------------------------

def _filter_x_outliers(mesh: trimesh.Trimesh, label: str = "") -> trimesh.Trimesh:
    """Remove vertices whose x-coordinate is more than 1 std from the mean x."""
    verts = mesh.vertices
    x      = verts[:, 0]
    mean_x = x.mean()
    std_x  = x.std()
    keep   = np.abs(x - mean_x) <= std_x

    old_to_new          = np.full(len(verts), -1, dtype=int)
    old_to_new[keep]    = np.arange(keep.sum())
    face_mask           = np.all(keep[mesh.faces], axis=1)
    new_mesh = trimesh.Trimesh(
        vertices = verts[keep],
        faces    = old_to_new[mesh.faces[face_mask]],
        process  = False,
    )
    removed = (~keep).sum()
    print(f"  X-outlier filter ({label}): removed {removed:,} verts "
          f"(x outside [{mean_x - std_x:.1f}, {mean_x + std_x:.1f}])")
    return new_mesh


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_joint_space(
    bone1_path: str,
    bone2_path: str,
    output_path: str,
    pitch: float = 0.75,
    padding: float = 3.0,
    decimate: float = 0.05,
    smooth_iterations: int = 100,
) -> None:
    """
    bone1_path        : the bone on the high-x side (ref_1)
    bone2_path        : the bone on the low-x side  (ref_2)
    decimate          : target face fraction after decimation (default 0.1 = keep 10%)
    smooth_iterations : Taubin smoothing passes after decimation (default 20)
    """
    print(f"\nLoading meshes ...")
    b1 = trimesh.load(str(bone1_path), force="mesh", process=False)
    b2 = trimesh.load(str(bone2_path), force="mesh", process=False)
    b1.merge_vertices()
    b2.merge_vertices()
    print(f"  Bone 1 (high-x): {len(b1.vertices):,} verts, {len(b1.faces):,} faces")
    print(f"  Bone 2 (low-x):  {len(b2.vertices):,} verts, {len(b2.faces):,} faces")

    # ── X-axis outlier cleanup ────────────────────────────────────────────────
    b1 = _filter_x_outliers(b1, label="bone 1")
    b2 = _filter_x_outliers(b2, label="bone 2")

    # ── Shared voxel grid ─────────────────────────────────────────────────
    all_verts = np.vstack([b1.vertices, b2.vertices])
    vmin  = all_verts.min(0) - padding
    vmax  = all_verts.max(0) + padding
    shape = tuple(np.ceil((vmax - vmin) / pitch).astype(int))
    print(f"\nVoxel grid: {shape}  ({int(np.prod(shape)):,} voxels,  pitch={pitch} mm)")

    # ── Voxelise bones ────────────────────────────────────────────────────
    print("Voxelising bone 1 ...")
    in_b1 = _bone_mask(b1, vmin, shape, pitch)
    print(f"  Bone 1 voxels: {in_b1.sum():,}")

    print("Voxelising bone 2 ...")
    in_b2 = _bone_mask(b2, vmin, shape, pitch)
    print(f"  Bone 2 voxels: {in_b2.sum():,}")

    # ── Ray-cast along X for each (Y, Z) column ───────────────────────────
    # in_b1 shape: (nx, ny, nz)
    nx, ny, nz = shape
    joint_vox = np.zeros(shape, dtype=bool)
    n_filled = 0

    # For each (y, z) column, argmax over x gives the first True index.
    # We need the LAST x of b2 (max) and FIRST x of b1 (min).
    # np.argmax returns the first True; for last True we flip along x.

    # Precompute: for b2 last-x and b1 first-x along axis 0
    # b2_last[y,z]  = last  x-index in b2 that is True (-1 if none)
    # b1_first[y,z] = first x-index in b1 that is True (-1 if none)

    has_b2 = in_b2.any(axis=0)          # (ny, nz)
    has_b1 = in_b1.any(axis=0)          # (ny, nz)
    valid  = has_b2 & has_b1            # columns with both bones present

    print(f"\nRay-casting along X axis ...")
    print(f"  (Y,Z) columns with both bones present: {valid.sum():,} / {ny*nz:,}")

    # Last x occupied by b2: flip x, take argmax, convert back
    b2_last  = (nx - 1) - np.argmax(in_b2[::-1, :, :], axis=0)   # (ny, nz)
    b1_first =            np.argmax(in_b1,               axis=0)   # (ny, nz)

    # argmax returns 0 when no True exists — mask those out with valid
    filled = valid & (b2_last < b1_first)
    print(f"  Columns with a gap (b2_max < b1_min): {filled.sum():,}")

    ys, zs = np.where(filled)
    for y, z in zip(ys, zs):
        x_lo = int(b2_last[y, z]) + 1
        x_hi = int(b1_first[y, z])
        joint_vox[x_lo:x_hi, y, z] = True
        n_filled += x_hi - x_lo

    print(f"  Joint-space voxels filled: {n_filled:,}")

    if n_filled == 0:
        print("ERROR: no gap found between bones along X.  "
              "Check bone orientation — bone-1 should have higher x than bone-2.")
        sys.exit(1)

    # ── Marching cubes → mesh ─────────────────────────────────────────────
    print("\nRunning marching cubes ...")
    padded = np.pad(joint_vox.astype(np.float32), 1, constant_values=0.0)
    verts_mc, faces_mc, _, _ = marching_cubes(padded, level=0.5)
    verts_world = (verts_mc - 1.0) * pitch + vmin

    mesh_out = trimesh.Trimesh(vertices=verts_world, faces=faces_mc)
    mesh_out.merge_vertices()

    vol_mm3 = n_filled * pitch ** 3
    print(f"  Joint-space volume (voxel):  {vol_mm3:.1f} mm³  = {vol_mm3/1000:.3f} cm³")
    print(f"  Raw mesh: {len(mesh_out.vertices):,} verts, {len(mesh_out.faces):,} faces")

    # ── Simplify: decimate then Taubin smooth ────────────────────────────
    print(f"  Decimating (keeping {decimate*100:.0f}% of faces) ...")
    verts_s, faces_s = _decimate(np.array(mesh_out.vertices, dtype=np.float32),
                                  np.array(mesh_out.faces,    dtype=np.int32),
                                  ratio=decimate)
    print(f"  Smoothing ({smooth_iterations} Taubin iterations) ...")
    verts_s = _taubin_smooth(verts_s, faces_s, smooth_iterations)
    mesh_out = trimesh.Trimesh(vertices=verts_s, faces=faces_s)
    mesh_out.merge_vertices()
    print(f"  Simplified mesh: {len(mesh_out.vertices):,} verts, {len(mesh_out.faces):,} faces")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_out.export(str(out_path))
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Dual-EDT joint space
# ---------------------------------------------------------------------------

def compute_joint_space_edt(
    bone1_path: str,
    bone2_path: str,
    output_path: str,
    pitch: float = 0.75,
    padding: float = 3.0,
    threshold: float = 10.0,
    decimate: float = 0.05,
    smooth_iterations: int = 100,
) -> None:
    """
    Joint-space mesh via dual Euclidean Distance Transform.

    A voxel belongs to the joint space if:
      - it is not inside either bone, AND
      - its distance to bone-1 surface < threshold (voxels), AND
      - its distance to bone-2 surface < threshold (voxels)

    i.e. it is simultaneously close to both articular surfaces.
    Works for any joint orientation — no axis assumption.

    threshold : half-width of the joint space in voxels.  Tune this so the
                resulting mesh covers the visible cartilage gap.
    """
    print(f"\nLoading meshes ...")
    b1 = trimesh.load(str(bone1_path), force="mesh", process=False)
    b2 = trimesh.load(str(bone2_path), force="mesh", process=False)
    b1.merge_vertices();  b2.merge_vertices()
    print(f"  Bone 1: {len(b1.vertices):,} verts")
    print(f"  Bone 2: {len(b2.vertices):,} verts")

    b1 = _filter_x_outliers(b1, "bone 1")
    b2 = _filter_x_outliers(b2, "bone 2")

    # ── Shared voxel grid ─────────────────────────────────────────────────
    all_verts = np.vstack([b1.vertices, b2.vertices])
    vmin  = all_verts.min(0) - padding
    vmax  = all_verts.max(0) + padding
    shape = tuple(np.ceil((vmax - vmin) / pitch).astype(int))
    print(f"\nVoxel grid: {shape}  pitch={pitch}")

    in_b1 = _bone_mask(b1, vmin, shape, pitch)
    in_b2 = _bone_mask(b2, vmin, shape, pitch)
    print(f"  Bone 1 voxels: {in_b1.sum():,}")
    print(f"  Bone 2 voxels: {in_b2.sum():,}")

    # ── Dual EDT ─────────────────────────────────────────────────────────
    print(f"\nComputing distance transforms (threshold={threshold} voxels) ...")
    d1 = distance_transform_edt(~in_b1)   # distance to bone-1 surface
    d2 = distance_transform_edt(~in_b2)   # distance to bone-2 surface

    joint_vox = (d1 < threshold) & (d2 < threshold) & ~in_b1 & ~in_b2
    n_filled  = int(joint_vox.sum())
    print(f"  Joint-space voxels: {n_filled:,}")

    if n_filled == 0:
        print("ERROR: no joint-space voxels found. Try increasing --cartilage-threshold.")
        sys.exit(1)

    # ── Marching cubes → mesh ─────────────────────────────────────────────
    print("\nRunning marching cubes ...")
    padded = np.pad(joint_vox.astype(np.float32), 1, constant_values=0.0)
    verts_mc, faces_mc, _, _ = marching_cubes(padded, level=0.5)
    verts_world = (verts_mc - 1.0) * pitch + vmin

    mesh_out = trimesh.Trimesh(vertices=verts_world, faces=faces_mc)
    mesh_out.merge_vertices()

    vol_mm3 = n_filled * pitch ** 3
    print(f"  Joint-space volume (voxel): {vol_mm3:.1f}  raw units³")
    print(f"  Raw mesh: {len(mesh_out.vertices):,} verts, {len(mesh_out.faces):,} faces")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_out.export(str(out_path))
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Injection overlap
# ---------------------------------------------------------------------------

def compute_injection(
    ref_c_aligned_path: str,
    sample_c_path:      str,
    output_path:        str,
    pitch:              float = 0.75,
    padding:            float = 3.0,
) -> None:
    """
    injection = voxels in ref_c_aligned that are NOT in sample_c.

    This represents the joint-space volume present in the reference (after
    alignment) that is missing in the sample — i.e. bone has grown into that
    space in the sample.

    Parameters
    ----------
    ref_c_aligned_path : aligned reference joint-space STL
    sample_c_path      : sample joint-space STL
    output_path        : where to write injection.stl
    """
    print(f"\nLoading meshes ...")
    ref_c_aligned = trimesh.load(str(ref_c_aligned_path), force="mesh", process=False)
    sample_c      = trimesh.load(str(sample_c_path),      force="mesh", process=False)
    ref_c_aligned.merge_vertices()
    sample_c.merge_vertices()
    print(f"  ref_c_aligned: {len(ref_c_aligned.vertices):,} verts")
    print(f"  sample_c:      {len(sample_c.vertices):,} verts")

    all_verts = np.vstack([ref_c_aligned.vertices, sample_c.vertices])
    vmin  = all_verts.min(0) - padding
    vmax  = all_verts.max(0) + padding
    shape = tuple(np.ceil((vmax - vmin) / pitch).astype(int))
    print(f"\nVoxel grid: {shape}  ({int(np.prod(shape)):,} voxels,  pitch={pitch} mm)")

    print("Voxelising ref_c_aligned ...")
    in_ref = _bone_mask(ref_c_aligned, vmin, shape, pitch)
    print(f"  ref_c_aligned voxels: {in_ref.sum():,}")

    print("Voxelising sample_c ...")
    in_smp = _bone_mask(sample_c, vmin, shape, pitch)
    print(f"  sample_c voxels:      {in_smp.sum():,}")

    injection = in_ref & ~in_smp
    n_inj = int(injection.sum())
    print(f"\nInjection voxels (ref_c_aligned − sample_c): {n_inj:,}")
    if n_inj == 0:
        print("WARNING: no injection voxels found.")
        return

    # Keep only the largest 3-D connected component (removes noise voxels that
    # are isolated or only touching along edges/corners in 2-D slices)
    print("Cleaning: keeping largest 3-D connected component ...")
    labeled, n_components = nd_label(injection)
    print(f"  Components found: {n_components}")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0                          # ignore background label
    largest_label = int(sizes.argmax())
    injection = (labeled == largest_label)
    n_inj = int(injection.sum())
    print(f"  Voxels after cleanup: {n_inj:,}  "
          f"(removed {int(sizes.sum()) - n_inj:,} noise voxels)")

    # Convert voxel indices to world-space point cloud (voxel centres)
    idx = np.argwhere(injection)                          # (N, 3) integer indices
    points = idx.astype(np.float32) * pitch + vmin       # world coords (mm)

    vol_mm3 = n_inj * pitch ** 3
    print(f"\nInjection volume: {vol_mm3:.1f} mm³  = {vol_mm3/1000:.3f} cm³")
    print(f"  Point cloud: {len(points):,} points")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.PointCloud(vertices=points)
    cloud.export(str(out_path))
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Joint-space mesh: fills the X-gap between two bone STLs "
                    "for every (Y,Z) column.")
    parser.add_argument("bone1", help="Bone on the HIGH-x side (e.g. ref_1.stl)")
    parser.add_argument("bone2", help="Bone on the LOW-x  side (e.g. ref_2.stl)")
    parser.add_argument("-o", "--output", default="ref_c.stl",
                        help="Output STL path (default: ref_c.stl)")
    parser.add_argument("--pitch", type=float, default=0.75,
                        help="Voxel size in mm (default: 0.75)")
    parser.add_argument("--padding", type=float, default=3.0,
                        help="Grid padding in mm (default: 3.0)")
    args = parser.parse_args()

    compute_joint_space(
        args.bone1, args.bone2, args.output,
        pitch=args.pitch, padding=args.padding,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        print("Usage: python joint_space.py ref_1.stl ref_2.stl [-o ref_c.stl]")
        sys.exit(0)
    main()
