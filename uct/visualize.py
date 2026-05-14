#!/usr/bin/env python3
"""
visualize.py — post-pipeline visualization utilities for uCT alignment results.

Usage:
    python visualize.py 360 <stl> [<stl2> ...] [--out NAME] [--frames N] [--fps F] [--width W] [--height H]

Examples:
    python visualize.py 360 results_CT_20260201_191553/CT_20260310_191553.stl
    python visualize.py 360 results_CT_20260201_191553/ref_1.stl results_CT_20260201_191553/ref_2.stl --out ref_combined
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import imageio.v3 as iio

# Colours for up to 6 meshes; more meshes reuse the last colour
MESH_COLORS = [
    [240, 225, 200],   # warm ivory
    [100, 180, 230],   # sky blue
    [230, 120, 100],   # coral
    [120, 210, 140],   # mint green
    [210, 160, 230],   # lavender
    [240, 200,  80],   # amber
    [100, 220, 210],   # teal
    [230, 140, 190],   # pink
    [ 90, 160, 100],   # forest green
    [180, 120,  80],   # terracotta
    [140, 180, 240],   # periwinkle
    [240, 180, 120],   # peach
]


# ---------------------------------------------------------------------------
# 360° rotation video
# ---------------------------------------------------------------------------

def render_360(stl_paths: list[str],
               out_name: str | None = None,
               n_frames: int = 180,
               fps: int = 30,
               width: int = 1280,
               height: int = 720,
               cam_elevation: float = 0.0) -> Path:
    """
    Render a 360° rotation video around the longest axis (X, after pipeline
    orientation) for one or more STL meshes shown together.

    The camera orbits in the YZ plane around the combined bounding-box centre,
    keeping X as the 'up' direction so the bones stand vertically.

    Output: <results_dir>/<out_name>_360.mp4  (or <stl_stem>_360.mp4 for a
    single file).  <out_name> defaults to the stem of the first STL.
    """
    stl_paths = [Path(p) for p in stl_paths]
    out_dir   = stl_paths[0].parent
    stem      = out_name if out_name else stl_paths[0].stem
    output_path = out_dir / (stem + "_360.mp4")

    print(f"Loading {len(stl_paths)} mesh(es) ...")
    meshes = []
    for p in stl_paths:
        m = pv.read(str(p))
        meshes.append(m)
        print(f"  {p.name}: {m.n_points:,} verts  |  {m.n_cells:,} faces")

    # Combined bounding box for camera fitting
    all_bounds = np.array([m.bounds for m in meshes])
    combined_min = all_bounds[:, [0, 2, 4]].min(axis=0)
    combined_max = all_bounds[:, [1, 3, 5]].max(axis=0)
    center  = (combined_min + combined_max) / 2.0
    extents = combined_max - combined_min
    print(f"  Combined extents: X={extents[0]:.1f}mm  Y={extents[1]:.1f}mm  Z={extents[2]:.1f}mm")
    print(f"  Output: {output_path}  ({n_frames} frames @ {fps}fps)")

    pl = pv.Plotter(off_screen=True, window_size=[width, height])
    pl.background_color = [12, 12, 18]

    fallback_colors = MESH_COLORS + [MESH_COLORS[-1]] * max(0, len(meshes) - len(MESH_COLORS))
    for mesh, fallback_color in zip(meshes, fallback_colors):
        # Use stored vertex colors if the mesh has them (e.g. articular PLY files)
        if 'RGBA' in mesh.point_data:
            rgb = mesh.point_data['RGBA'][:, :3]   # drop alpha
            mesh.point_data['RGB'] = rgb
            pl.add_mesh(mesh, scalars='RGB', rgb=True,
                        smooth_shading=True, ambient=0.25, diffuse=0.75)
        elif 'RGB' in mesh.point_data:
            pl.add_mesh(mesh, scalars='RGB', rgb=True,
                        smooth_shading=True, ambient=0.25, diffuse=0.75)
        else:
            pl.add_mesh(mesh, color=fallback_color,
                        specular=0.5, specular_power=20,
                        smooth_shading=True, ambient=0.25, diffuse=0.75)

    # Auto-fit zoom from the +Y side with X as up, then extract orbit radius
    pl.camera.position    = (center[0], center[1] + 1.0, center[2])
    pl.camera.focal_point = tuple(center)
    pl.camera.up          = (1.0, 0.0, 0.0)
    pl.reset_camera()

    cam_pos    = np.array(pl.camera.position)
    cam_radius = float(np.linalg.norm(cam_pos[[1, 2]] - center[[1, 2]]))
    print(f"  Camera orbit radius (auto-fit): {cam_radius:.1f}mm")

    pl.add_light(pv.Light(
        position=(center[0], center[1], center[2] + cam_radius * 1.5),
        color='white', intensity=0.9))
    pl.add_light(pv.Light(
        position=(center[0], center[1], center[2] - cam_radius * 0.8),
        color=[200, 210, 255], intensity=0.4))

    # cam_elevation shifts the camera along X (the long/up axis).
    # Negative = camera lower than centre → looks upward toward the joint.
    cam_x = center[0] + cam_elevation

    frames = []
    for i in range(n_frames):
        angle = 2.0 * np.pi * i / n_frames
        cam_y = center[1] + cam_radius * np.cos(angle)
        cam_z = center[2] + cam_radius * np.sin(angle)

        pl.camera.position    = (cam_x, cam_y, cam_z)
        pl.camera.focal_point = tuple(center)
        pl.camera.up          = (1.0, 0.0, 0.0)

        pl.render()
        frames.append(pl.screenshot(return_img=True))

        if (i + 1) % (n_frames // 6) == 0:
            print(f"  {i+1}/{n_frames} frames rendered")

    pl.close()

    print("Encoding video ...")
    iio.imwrite(str(output_path), frames, fps=fps, codec="libx264",
                output_params=["-crf", "18", "-pix_fmt", "yuv420p"])

    size_b   = output_path.stat().st_size
    size_str = f"{size_b/1e6:.1f} MB" if size_b >= 1e6 else f"{size_b/1e3:.0f} KB"
    print(f"Done → {output_path}  ({size_str})")
    return output_path


# ---------------------------------------------------------------------------
# Lineup video  (segments arranged in a row, all rotating together)
# ---------------------------------------------------------------------------

def _rx_about_point(cx: float, cy: float, cz: float, theta: float) -> np.ndarray:
    """4×4 homogeneous matrix: rotate theta radians around the X axis at (cx,cy,cz)."""
    c, s = np.cos(theta), np.sin(theta)
    # T(center) @ Rx(theta) @ T(-center)
    return np.array([
        [1,  0,  0,  0                   ],
        [0,  c, -s,  cy*(1 - c) + cz*s   ],
        [0,  s,  c,  cz*(1 - c) - cy*s   ],
        [0,  0,  0,  1                   ],
    ], dtype=np.float64)


def _align_longest_to_x(mesh: pv.PolyData) -> pv.PolyData:
    """Rotate mesh so its longest bounding-box axis aligns with X."""
    b      = np.array(mesh.bounds)
    spans  = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
    longest = int(np.argmax(spans))
    c      = np.array(mesh.center)
    if longest == 0:
        return mesh                                          # already X
    elif longest == 1:                                       # Y → rotate -90° around Z
        return mesh.rotate_z(-90, point=c, inplace=False)
    else:                                                    # Z → rotate  90° around Y
        return mesh.rotate_y(90, point=c, inplace=False)


_REFLECT_NORMALS = {'x': (1,0,0), 'y': (0,1,0), 'z': (0,0,1)}


def _mesh_to_obb(mesh: pv.PolyData) -> pv.PolyData:
    """
    Replace mesh with its minimum oriented bounding box found via PCA.

    The three principal components (eigenvectors of the vertex covariance
    matrix, sorted by descending eigenvalue) define the box axes.  PC1 —
    longest spread — is placed along X so the spin-around-X animation shows
    the box rotating around its own long axis, matching the bone behaviour.
    """
    pts = np.array(mesh.points)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    cov = (centered.T @ centered) / len(pts)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)   # ascending order
    idx = np.argsort(eigenvalues)[::-1]                # descending: PC1 first
    axes = eigenvectors[:, idx]                        # columns = PC axes

    projected = centered @ axes
    lo = projected.min(axis=0)
    hi = projected.max(axis=0)

    obb_center_local = (lo + hi) / 2.0
    extents = hi - lo

    # Build an axis-aligned box in PCA space, then rotate+translate to world
    box = pv.Box(bounds=(-extents[0]/2, extents[0]/2,
                          -extents[1]/2, extents[1]/2,
                          -extents[2]/2, extents[2]/2)).triangulate()

    T = np.eye(4)
    T[:3, :3] = axes                              # rotation: align axes to PCs
    T[:3, 3]  = centroid + axes @ obb_center_local  # translate to world centre
    box.transform(T, inplace=True)
    return box

def _make_row(meshes: list, gap_frac: float, x_center: float,
              align_x: bool = False,
              prerotate_deg: tuple[float, float, float] | None = None,
              reflect_axis: str | None = None) -> tuple[list, list]:
    """
    Optionally align, reflect, or pre-rotate each mesh, then lay them out
    along Z with gaps and shift the whole row to x_center in X.
    Returns (positioned_meshes, pivot_centers).
    """
    aligned = []
    for m in meshes:
        if align_x:
            m = _align_longest_to_x(m)
        if reflect_axis is not None:
            normal = _REFLECT_NORMALS[reflect_axis.lower()]
            m = m.reflect(normal, point=m.center, inplace=False)
        if prerotate_deg is not None:
            rx, ry, rz = prerotate_deg
            c = np.array(m.center)
            if rx: m = m.rotate_x(rx, point=c, inplace=False)
            if ry: m = m.rotate_y(ry, point=c, inplace=False)
            if rz: m = m.rotate_z(rz, point=c, inplace=False)
        aligned.append(m)

    max_z_ext = max(m.bounds[5] - m.bounds[4] for m in aligned)
    gap       = max_z_ext * gap_frac

    positioned = []
    cursor_z   = 0.0
    for m in aligned:
        c     = np.array(m.center)
        z_ext = m.bounds[5] - m.bounds[4]
        shift = np.array([-c[0] + x_center, -c[1], -c[2] + cursor_z + z_ext / 2])
        positioned.append(m.translate(shift, inplace=False))
        cursor_z += z_ext + gap

    # centre the row in Z
    total_z  = cursor_z - gap
    z_offset = -total_z / 2
    positioned = [m.translate([0, 0, z_offset], inplace=False) for m in positioned]
    pivots     = [np.array(m.center) for m in positioned]
    return positioned, pivots


def render_lineup(stl_paths: list[str],
                  stl_paths_row2: list[str] | None = None,
                  out_name: str | None = None,
                  n_frames: int = 180,
                  fps: int = 30,
                  width: int = 1920,
                  height: int = 720,
                  gap_frac: float = 0.3,
                  row1_rotate: tuple[float, float, float] | None = None,
                  row2_rotate: tuple[float, float, float] | None = None,
                  row1_reflect: str | None = None,
                  row2_reflect: str | None = None,
                  use_obb: bool = False) -> Path:
    """
    Lay out one or two rows of meshes (row 1 on top, row 2 below) along Z,
    then render a 360° video where every mesh spins around its own centre of
    mass (X axis).  Camera is fixed.

    Each mesh gets a distinct colour.  Output: <out_name>_lineup_360.mp4.
    """
    all_paths = list(stl_paths) + (list(stl_paths_row2) if stl_paths_row2 else [])
    out_dir   = Path(all_paths[0]).parent
    stem      = out_name if out_name else Path(all_paths[0]).stem
    output_path = out_dir / (stem + "_lineup_360.mp4")

    def load(paths, label):
        print(f"Loading {len(paths)} mesh(es) [{label}] ...")
        out = []
        for p in paths:
            m = pv.read(str(p))
            out.append(m)
            print(f"  {Path(p).name}: {m.n_points:,} verts  |  {m.n_cells:,} faces")
        return out

    meshes1 = load(stl_paths, "row 1")
    meshes2 = load(stl_paths_row2, "row 2") if stl_paths_row2 else []

    if use_obb:
        print("Computing oriented bounding boxes (PCA) ...")
        meshes1 = [_mesh_to_obb(m) for m in meshes1]
        meshes2 = [_mesh_to_obb(m) for m in meshes2]

    # X extents of each row (to compute the vertical separation)
    x_ext1 = max(m.bounds[1] - m.bounds[0] for m in meshes1)
    x_ext2 = max(m.bounds[1] - m.bounds[0] for m in meshes2) if meshes2 else 0.0
    row_gap = max(x_ext1, x_ext2) * 0.4   # vertical gap between rows

    # Row 1 sits above X=0, row 2 sits below
    x_top    =  x_ext1 / 2 + row_gap / 2
    x_bottom = -(x_ext2 / 2 + row_gap / 2) if meshes2 else 0.0

    positioned1, pivots1 = _make_row(meshes1, gap_frac, x_center=x_top,    align_x=False, prerotate_deg=row1_rotate, reflect_axis=row1_reflect)
    positioned2, pivots2 = _make_row(meshes2, gap_frac, x_center=x_bottom, align_x=False, prerotate_deg=row2_rotate, reflect_axis=row2_reflect) if meshes2 else ([], [])

    all_positioned = positioned1 + positioned2
    all_pivots     = pivots1     + pivots2

    all_bounds   = np.array([m.bounds for m in all_positioned])
    cmin         = all_bounds[:, [0, 2, 4]].min(axis=0)
    cmax         = all_bounds[:, [1, 3, 5]].max(axis=0)
    scene_center = (cmin + cmax) / 2.0
    extents      = cmax - cmin
    print(f"  Layout extents: X={extents[0]:.1f}mm  Y={extents[1]:.1f}mm  Z={extents[2]:.1f}mm")
    print(f"  Output: {output_path}  ({n_frames} frames @ {fps}fps)")

    pl = pv.Plotter(off_screen=True, window_size=[width, height])
    pl.background_color = [12, 12, 18]

    # assign colours: row 1 from start of palette, row 2 continues from where row 1 left off
    palette = MESH_COLORS * (len(all_positioned) // len(MESH_COLORS) + 1)
    actors  = []
    for mesh, color in zip(all_positioned, palette):
        actor = pl.add_mesh(mesh, color=color,
                            specular=0.5, specular_power=20,
                            smooth_shading=True, ambient=0.25, diffuse=0.75)
        actors.append(actor)

    pl.camera.position    = (scene_center[0], scene_center[1] + 1.0, scene_center[2])
    pl.camera.focal_point = tuple(scene_center)
    pl.camera.up          = (1.0, 0.0, 0.0)
    pl.reset_camera()
    cam_dist = np.linalg.norm(np.array(pl.camera.position) - scene_center)
    print(f"  Camera distance (fixed): {cam_dist:.1f}mm")

    pl.add_light(pv.Light(
        position=(scene_center[0], scene_center[1], scene_center[2] + cam_dist),
        color='white', intensity=0.9))
    pl.add_light(pv.Light(
        position=(scene_center[0], scene_center[1], scene_center[2] - cam_dist * 0.6),
        color=[200, 210, 255], intensity=0.4))

    frames = []
    for i in range(n_frames):
        angle = 2.0 * np.pi * i / n_frames
        for actor, piv in zip(actors, all_pivots):
            actor.user_matrix = _rx_about_point(piv[0], piv[1], piv[2], angle)

        pl.render()
        frames.append(pl.screenshot(return_img=True))

        if (i + 1) % (n_frames // 6) == 0:
            print(f"  {i+1}/{n_frames} frames rendered")

    pl.close()

    print("Encoding video ...")
    iio.imwrite(str(output_path), frames, fps=fps, codec="libx264",
                output_params=["-crf", "18", "-pix_fmt", "yuv420p"])

    size_b   = output_path.stat().st_size
    size_str = f"{size_b/1e6:.1f} MB" if size_b >= 1e6 else f"{size_b/1e3:.0f} KB"
    print(f"Done → {output_path}  ({size_str})")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-pipeline visualization utilities for uCT alignment results."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p360 = sub.add_parser("360", help="360° rotation video around the longest axis")
    p360.add_argument("stl", nargs="+", help="One or more STL files to display together")
    p360.add_argument("--out",    default=None,
                      help="Output filename stem (default: first STL stem)")
    p360.add_argument("--frames", type=int, default=180,
                      help="Number of frames (default: 180 → 6 s at 30 fps)")
    p360.add_argument("--fps",    type=int, default=30)
    p360.add_argument("--width",  type=int, default=1280)
    p360.add_argument("--height", type=int, default=720)
    p360.add_argument("--cam-elevation", type=float, default=0.0,
                      help="Shift camera along the long (X) axis in mm. "
                           "Negative = lower camera, looks up toward the joint.")

    plu = sub.add_parser("lineup", help="Segments arranged in a row, rotating individually")
    plu.add_argument("stl", nargs="+", help="Row-1 STL files")
    plu.add_argument("--row2", nargs="+", default=None,
                     help="Optional second row of STL files displayed below the first")
    plu.add_argument("--row1-rotate", nargs=3, type=float, default=None,
                     metavar=("RX", "RY", "RZ"),
                     help="Pre-rotate all row-1 meshes by RX RY RZ degrees")
    plu.add_argument("--row2-rotate", nargs=3, type=float, default=None,
                     metavar=("RX", "RY", "RZ"),
                     help="Pre-rotate all row-2 meshes by RX RY RZ degrees")
    plu.add_argument("--row1-reflect", default=None, choices=["x", "y", "z"],
                     help="Reflect row-1 meshes across this axis (x, y, or z)")
    plu.add_argument("--row2-reflect", default=None, choices=["x", "y", "z"],
                     help="Reflect row-2 meshes across this axis (x, y, or z)")
    plu.add_argument("--out",    default=None)
    plu.add_argument("--frames", type=int, default=180)
    plu.add_argument("--fps",    type=int, default=30)
    plu.add_argument("--width",  type=int, default=1920)
    plu.add_argument("--height", type=int, default=720)
    plu.add_argument("--gap",    type=float, default=0.3,
                     help="Gap between meshes as fraction of widest Z extent (default 0.3)")
    plu.add_argument("--obb", action="store_true",
                     help="Replace each mesh with its oriented bounding box (PCA cuboid)")

    args = parser.parse_args()

    if args.cmd == "360":
        render_360(args.stl,
                   out_name=args.out,
                   n_frames=args.frames,
                   fps=args.fps,
                   width=args.width,
                   height=args.height,
                   cam_elevation=args.cam_elevation)

    elif args.cmd == "lineup":
        render_lineup(args.stl,
                      stl_paths_row2=args.row2,
                      out_name=args.out,
                      n_frames=args.frames,
                      fps=args.fps,
                      width=args.width,
                      height=args.height,
                      gap_frac=args.gap,
                      row1_rotate=tuple(args.row1_rotate) if args.row1_rotate else None,
                      row2_rotate=tuple(args.row2_rotate) if args.row2_rotate else None,
                      row1_reflect=args.row1_reflect,
                      row2_reflect=args.row2_reflect,
                      use_obb=args.obb)


if __name__ == "__main__":
    main()
