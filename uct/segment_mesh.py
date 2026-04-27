"""
Segment a mesh into disconnected parts and save each as a separate STL.

Each connected component (a set of faces with no shared edges to other faces)
is saved as its own STL file at its original position in world space.

Usage:
    python segment_mesh.py bone.stl
    python segment_mesh.py bone.stl -o parts/
    python segment_mesh.py bone.stl --min-faces 500   # discard tiny fragments
    python segment_mesh.py bone.stl --top 5           # keep only 5 largest parts

Dependencies:
    pip install trimesh numpy
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import trimesh


def split_bone_xslice(
    bone: trimesh.Trimesh,
    voxel_size: float,
    area_drop_fraction: float = 0.5,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh] | None:
    """
    Sweep X-cross-sections from +x (bottom) toward -x (top) in steps of
    voxel_size.  At each slice the convex-hull area of the YZ cross-section
    is computed (so hollow cross-sections are treated as filled).  On the
    first slice where area drops by more than area_drop_fraction relative to
    the previous slice the mesh is cut there and (lower_part, upper_part) is
    returned.  lower_part has x >= cut_x; upper_part has x < cut_x.
    Returns None if no qualifying drop is found.
    """
    from scipy.spatial import ConvexHull

    x_max = float(bone.vertices[:, 0].max())
    x_min = float(bone.vertices[:, 0].min())
    x_positions = np.arange(x_max, x_min - voxel_size, -voxel_size)
    n_slices = len(x_positions)

    prev_area = None
    cut_x = None

    for idx, x_val in enumerate(x_positions):
        lines = trimesh.intersections.mesh_plane(
            bone,
            plane_normal=np.array([1.0, 0.0, 0.0]),
            plane_origin=np.array([x_val, 0.0, 0.0]),
        )

        if lines is None or len(lines) == 0:
            area = 0.0
        else:
            pts_yz = lines.reshape(-1, 3)[:, 1:]  # (2N, 2) YZ coords
            if len(pts_yz) < 3:
                area = 0.0
            else:
                try:
                    hull = ConvexHull(pts_yz)
                    area = float(hull.volume)  # in 2D, .volume == polygon area
                except Exception:
                    area = 0.0

        if idx % 50 == 0:
            print(f"    slice {idx+1}/{n_slices}  x={x_val:.3f}  ch_area={area:.3f}", end="\r")

        # Detect first >50% area drop (both slices must have real area to
        # avoid false triggers from mesh gaps or the bone boundary)
        if prev_area is not None and prev_area > 1e-12 and area > 1e-12:
            if area < prev_area * (1.0 - area_drop_fraction):
                cut_x = x_val
                break

        if area > 1e-12:
            prev_area = area

    print()  # clear progress line

    if cut_x is None:
        return None

    print(f"    cut at x={cut_x:.3f}")
    lower = trimesh.intersections.slice_mesh_plane(
        bone,
        plane_normal=np.array([1.0, 0.0, 0.0]),
        plane_origin=np.array([cut_x, 0.0, 0.0]),
    )
    upper = trimesh.intersections.slice_mesh_plane(
        bone,
        plane_normal=np.array([-1.0, 0.0, 0.0]),
        plane_origin=np.array([cut_x, 0.0, 0.0]),
    )

    if lower is None or upper is None or len(lower.faces) == 0 or len(upper.faces) == 0:
        return None

    return lower, upper


def split_thin_bridge(
    bone: trimesh.Trimesh,
    voxel_size: float,
    max_bridge_fraction: float = 0.05,
    min_size_ratio: float = 0.20,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh] | None:
    """
    Detect a thin bridge connecting two bone regions and split the mesh there.

    Works in any orientation (not axis-specific): voxelises the mesh, computes
    a distance transform, and erodes inward until the bridge disappears.

    Two criteria must both pass for a split to occur:
      1. Bridge volume < max_bridge_fraction of total mesh volume (bridge is thin).
      2. min(vol_A, vol_B) / max(vol_A, vol_B) > min_size_ratio (both pieces are
         substantial — prevents splitting off small accessory fragments).

    Bridge voxels are assigned to whichever piece they are closest to, so the
    intersection geometry is preserved rather than discarded.

    Returns (mesh_A, mesh_B) or None if no qualifying bridge is found.
    """
    from scipy.ndimage import distance_transform_edt, label as nd_label

    # ------------------------------------------------------------------ step 1
    # Voxelise at CT voxel resolution
    pitch = float(voxel_size)
    vox   = trimesh.voxel.creation.voxelize(bone, pitch)
    grid  = vox.matrix.copy()          # bool (nx, ny, nz) — True inside bone

    if grid.sum() == 0:
        return None

    # ------------------------------------------------------------------ step 2
    # Distance of every interior voxel to the nearest exterior voxel
    dist = distance_transform_edt(grid)   # float array, same shape as grid

    # ------------------------------------------------------------------ step 3
    # Find minimum erosion depth T (in voxels) that splits into 2 components.
    # min_vox_count: ignore components smaller than this (noise fragments).
    min_vox_count = max(8, int(500 * (pitch ** 3)))  # rough face-count analogue

    T_split  = None
    sizes_at_split = None

    for T in range(1, int(dist.max()) + 1):
        eroded   = dist > T
        labeled, n = nd_label(eroded)
        if n < 2:
            continue
        # Count voxels per component (label 0 = background/bridge, skip it)
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        top2 = np.sort(counts)[::-1][:2]
        if top2[1] >= min_vox_count:   # both largest components are non-trivial
            T_split        = T
            sizes_at_split = top2
            labeled_split  = labeled
            break

    if T_split is None:
        return None

    # ------------------------------------------------------------------ step 4
    # Check both split criteria
    bridge_vox    = grid & (dist <= T_split)
    bridge_volume = float(bridge_vox.sum()) * pitch ** 3

    total_volume = float(abs(bone.volume))
    if total_volume < 1e-12:
        total_volume = float(bone.convex_hull.volume)

    vol_A = float(sizes_at_split[0]) * pitch ** 3
    vol_B = float(sizes_at_split[1]) * pitch ** 3

    if bridge_volume / total_volume >= max_bridge_fraction:
        return None   # bridge too thick
    if min(vol_A, vol_B) / max(vol_A, vol_B) <= min_size_ratio:
        return None   # one piece is too small relative to the other

    print(f"    bridge={bridge_volume:.2f}  total={total_volume:.2f}"
          f"  ratio={bridge_volume/total_volume:.3f}"
          f"  size_ratio={min(vol_A,vol_B)/max(vol_A,vol_B):.3f}")

    # ------------------------------------------------------------------ step 5
    # Propagate the two component labels to bridge voxels (nearest labelled voxel).
    # labeled_split has 0 for background + bridge, 1 and 2 for the two components.
    _, idx = distance_transform_edt(labeled_split == 0, return_indices=True)
    full_label = labeled_split[tuple(idx)]   # propagate labels everywhere
    full_label[~grid] = 0                    # zero out non-bone voxels

    # Keep only the two dominant labels (renumber so they are 1 and 2)
    counts = np.bincount(full_label[grid].ravel())
    counts[0] = 0
    dominant = np.argsort(counts)[::-1][:2]
    label_map = np.zeros(full_label.max() + 1, dtype=np.intp)
    label_map[dominant[0]] = 1
    label_map[dominant[1]] = 2
    full_label = label_map[full_label]

    # ------------------------------------------------------------------ step 6
    # Classify each face by the label of the voxel containing its centroid.
    face_centroids = bone.vertices[bone.faces].mean(axis=1)   # (F, 3)
    vox_idx        = vox.points_to_indices(face_centroids)    # (F, 3) int
    nx, ny, nz     = grid.shape
    vox_idx        = np.clip(vox_idx, [0, 0, 0], [nx-1, ny-1, nz-1])
    face_labels    = full_label[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]

    faces_A = bone.faces[face_labels == 1]
    faces_B = bone.faces[face_labels == 2]

    if len(faces_A) == 0 or len(faces_B) == 0:
        return None

    mesh_A = trimesh.Trimesh(vertices=bone.vertices, faces=faces_A, process=True)
    mesh_B = trimesh.Trimesh(vertices=bone.vertices, faces=faces_B, process=True)

    return mesh_A, mesh_B


def segment_mesh(
    input_path: str,
    output_dir: str | None = None,
    min_faces: int = 0,
    top_n: int | None = None,
) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print(f"Loading {input_path} ...")
    mesh = trimesh.load(str(input_path), force="mesh", process=False)
    print(f"  Vertices: {len(mesh.vertices):,}   Faces: {len(mesh.faces):,}")

    # STL files store each triangle independently with no shared vertices.
    # Merge duplicate vertices so adjacency/connectivity can be computed.
    print("  Merging duplicate vertices ...")
    mesh.merge_vertices()
    print(f"  After merge: {len(mesh.vertices):,} vertices")

    # --- split into connected components ------------------------------------
    print("\nFinding connected components ...")
    components = mesh.split(only_watertight=False)
    print(f"  Found {len(components)} component(s)")

    # sort largest first
    components = sorted(components, key=lambda m: len(m.faces), reverse=True)

    # filter by minimum face count
    if min_faces > 0:
        before = len(components)
        components = [m for m in components if len(m.faces) >= min_faces]
        discarded = before - len(components)
        if discarded:
            print(f"  Discarded {discarded} fragment(s) with < {min_faces} faces")

    # keep only top N
    if top_n is not None:
        components = components[:top_n]

    if not components:
        print("No components remaining after filtering.")
        return

    # --- save each component -----------------------------------------------
    n_digits = len(str(len(components)))
    print(f"\nSaving {len(components)} part(s) to {out_dir}/")

    for i, part in enumerate(components):
        out_path = out_dir / f"{stem}_part{str(i + 1).zfill(n_digits)}.stl"
        part.export(str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [{i+1:>{n_digits}}] {len(part.faces):>10,} faces  →  {out_path.name}  ({size_mb:.1f} MB)")

    print(f"\nDone. {len(components)} parts saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Split a mesh into disconnected parts and save each as STL."
    )
    parser.add_argument("input", help="Input STL file")
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory for output STL files (default: same directory as input)",
    )
    parser.add_argument(
        "--min-faces", type=int, default=0, metavar="N",
        help="Discard parts with fewer than N faces (default: 0 = keep all)",
    )
    parser.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="Keep only the N largest parts (default: keep all)",
    )

    args = parser.parse_args()

    segment_mesh(
        args.input,
        output_dir=args.output_dir,
        min_faces=args.min_faces,
        top_n=args.top,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        print("Usage: python segment_mesh.py <mesh.stl> [options]")
        print("       python segment_mesh.py --help")
        sys.exit(0)
    main()
