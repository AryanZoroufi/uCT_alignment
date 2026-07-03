"""
Visualize the bridge valley bin on the combined segment (comp 1 = part_2+part_3).

Colors:
  RED   — faces to the left of the valley bin  (will become the smaller bone)
  BLACK — faces inside the valley bin           (the bridge region)
  BLUE  — faces to the right of the valley bin (will become the larger bone)

Saves a colored PLY and renders a 360 video.
"""
import trimesh, numpy as np, sys
sys.path.insert(0, '.')
from pipeline import _compute_bridge_profile

import pyvista as pv
import imageio.v3 as iio
from pathlib import Path

OUT_DIR = Path('results_CT_20260201_193908')

# ── Load the combined segment ────────────────────────────────────────────────
sample = trimesh.load(str(OUT_DIR / 'CT_20260310_193908_aligned.stl'),
                      force='mesh', process=False)
sample.merge_vertices()
comps = sorted(sample.split(only_watertight=False),
               key=lambda m: len(m.faces), reverse=True)
seg = comps[1]
print(f"Segment: {len(seg.faces):,} faces")

# ── Bridge profile ───────────────────────────────────────────────────────────
p        = _compute_bridge_profile(seg)
axis     = p['axis']
vi       = p['valley_idx']
bins     = p['bins']
b_lo     = float(bins[vi])
b_hi     = float(bins[vi + 1])
cut_pos  = p['cut_pos']

face_centroids = seg.vertices[seg.faces].mean(axis=1)
face_proj      = face_centroids @ axis

print(f"Cut axis: {axis.round(3)}  valley bin: [{b_lo:.2f}, {b_hi:.2f}]  "
      f"ratio={p['ratio']:.3f}")

left_mask   = face_proj <  b_lo
bridge_mask = (face_proj >= b_lo) & (face_proj <= b_hi)
right_mask  = face_proj >  b_hi
print(f"LEFT={left_mask.sum():,}  BRIDGE={bridge_mask.sum():,}  RIGHT={right_mask.sum():,}")

# ── Per-vertex colors (scatter face labels onto vertices) ────────────────────
RED   = np.array([220,  50,  50], dtype=np.uint8)
BLACK = np.array([  0,   0,   0], dtype=np.uint8)
BLUE  = np.array([ 50,  80, 220], dtype=np.uint8)

vert_colors = np.full((len(seg.vertices), 3), 128, dtype=np.uint8)
for mask, color in [(left_mask, RED), (bridge_mask, BLACK), (right_mask, BLUE)]:
    vert_colors[seg.faces[mask, 0]] = color
    vert_colors[seg.faces[mask, 1]] = color
    vert_colors[seg.faces[mask, 2]] = color

# ── Save colored PLY ─────────────────────────────────────────────────────────
seg.visual = trimesh.visual.ColorVisuals(vertex_colors=np.hstack(
    [vert_colors, np.full((len(seg.vertices), 1), 255, dtype=np.uint8)]))
ply_path = OUT_DIR / '_bridge_debug_colored.ply'
seg.export(str(ply_path))
print(f"Saved: {ply_path}")

# ── 360 video ────────────────────────────────────────────────────────────────
mesh_pv = pv.read(str(ply_path))
rgb = mesh_pv.point_data['RGBA'][:, :3]
mesh_pv.point_data['RGB'] = rgb

width, height, n_frames, fps = 1280, 720, 180, 30
pl = pv.Plotter(off_screen=True, window_size=[width, height])
pl.background_color = [12, 12, 18]
pl.add_mesh(mesh_pv, scalars='RGB', rgb=True,
            smooth_shading=True, ambient=0.3, diffuse=0.7)

# also draw the cut plane as a thin slab
b = np.array(seg.bounds)
center = seg.centroid
pl.camera.position    = (center[0], center[1] + 1.0, center[2])
pl.camera.focal_point = tuple(center)
pl.camera.up          = (1.0, 0.0, 0.0)
pl.reset_camera()
cam_pos    = np.array(pl.camera.position)
cam_radius = float(np.linalg.norm(cam_pos[[1, 2]] - np.array(center)[[1, 2]]))

pl.add_light(pv.Light(position=(center[0], center[1], center[2] + cam_radius * 1.5),
                      color='white', intensity=0.9))
pl.add_light(pv.Light(position=(center[0], center[1], center[2] - cam_radius * 0.8),
                      color=[200, 210, 255], intensity=0.4))

frames = []
for i in range(n_frames):
    angle = 2.0 * np.pi * i / n_frames
    pl.camera.position    = (center[0],
                              center[1] + cam_radius * np.cos(angle),
                              center[2] + cam_radius * np.sin(angle))
    pl.camera.focal_point = tuple(center)
    pl.camera.up          = (1.0, 0.0, 0.0)
    pl.render()
    frames.append(pl.screenshot(return_img=True))
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{n_frames}")
pl.close()

vid_path = OUT_DIR / '_bridge_debug_360.mp4'
iio.imwrite(str(vid_path), frames, fps=fps, codec='libx264',
            output_params=['-crf', '18', '-pix_fmt', 'yuv420p'])
print(f"Saved: {vid_path}")
