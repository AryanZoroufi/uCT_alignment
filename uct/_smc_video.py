"""
Render an SMC-alignment-progress video for an existing results directory.

For each IS stage end (4 per restart × N restarts for aggregate, 3 per restart
× N restarts per bone for per-bone), captures a pyvista render of the current
sample-bone positions vs fixed ref bones and holds it for HOLD_FRAMES frames
while the camera slowly orbits.

Output: <OUT_DIR>/smc_progress.mp4
"""
import sys, os, tempfile
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
sys.path.insert(0, '.')

import numpy as np
import trimesh
import pyvista as pv
import imageio.v3 as iio
from pathlib import Path
from smc_align import smc_align_aggregate, smc_align_per_bone

OUT_DIR       = Path('results_CT_20260201_191553')
VIDEO_OUT     = OUT_DIR / 'smc_progress.mp4'
WIDTH, HEIGHT = 1280, 720
FPS           = 30
HOLD_FRAMES   = 24    # video frames to hold each stage-end state
N_RESTARTS    = 5

# colours: ref = muted (shown at low opacity), sample = vivid
COLORS_REF    = [[220, 200, 175], [100, 180, 230]]   # ivory, sky-blue
COLORS_SAMPLE = [[240,  70,  50], [ 60, 200, 110]]   # red-coral, mint

ref_paths    = [str(OUT_DIR / 'ref_1.stl'),    str(OUT_DIR / 'ref_2.stl')]
sample_paths = [str(OUT_DIR / 'sample_1.stl'), str(OUT_DIR / 'sample_2.stl')]

# ── Load meshes ──────────────────────────────────────────────────────────────
ref_meshes  = [trimesh.load(p, force='mesh', process=False) for p in ref_paths]
init_meshes = [trimesh.load(p, force='mesh', process=False) for p in sample_paths]
for m in ref_meshes + init_meshes:
    m.merge_vertices()

# ── Build PyVista scene ──────────────────────────────────────────────────────
pl = pv.Plotter(off_screen=True, window_size=[WIDTH, HEIGHT])
pl.background_color = [12, 12, 18]

for m, c in zip(ref_meshes, COLORS_REF):
    faces_pad = np.hstack([np.full((len(m.faces), 1), 3, dtype=np.int_), m.faces])
    pl.add_mesh(pv.PolyData(m.vertices.astype(np.float32), faces_pad),
                color=c, opacity=0.30,
                smooth_shading=True, ambient=0.25, diffuse=0.75)

sample_pv = []
for m, c in zip(init_meshes, COLORS_SAMPLE):
    faces_pad = np.hstack([np.full((len(m.faces), 1), 3, dtype=np.int_), m.faces])
    pd = pv.PolyData(m.vertices.astype(np.float32), faces_pad)
    pl.add_mesh(pd, color=c,
                smooth_shading=True, ambient=0.30, diffuse=0.70,
                specular=0.4, specular_power=15)
    sample_pv.append(pd)

# Camera centred on ref bones (fixed target), wide enough to see anywhere
ref_verts = np.vstack([m.vertices for m in ref_meshes])
center     = ref_verts.mean(0)
pl.camera.position    = (center[0], center[1] + 1.0, center[2])
pl.camera.focal_point = tuple(center)
pl.camera.up          = (1.0, 0.0, 0.0)
pl.reset_camera()
cam_radius = float(np.linalg.norm(
    np.array(pl.camera.position)[[1, 2]] - center[[1, 2]])) * 1.3  # widen 30%

pl.add_light(pv.Light(
    position=(center[0], center[1], center[2] + cam_radius * 1.5),
    color='white', intensity=0.9))
pl.add_light(pv.Light(
    position=(center[0], center[1], center[2] - cam_radius * 0.8),
    color=[200, 210, 255], intensity=0.4))

# ── Frame capture callback ───────────────────────────────────────────────────
frames      = []
cam_angle   = [0.0]
text_actor  = [None]
ORBIT_STEP  = 2.0 * np.pi / (HOLD_FRAMES * 50)   # full orbit over 50 states

def frame_cb(label, all_verts, iou=None):
    for pd, verts in zip(sample_pv, all_verts):
        pd.points[:] = verts.astype(np.float32)

    if text_actor[0] is not None:
        pl.remove_actor(text_actor[0])
    iou_str = f"IoU = {iou:.4f}" if iou is not None else ""
    text_actor[0] = pl.add_text(
        f"{label}\n{iou_str}",
        position='upper_left', font_size=14, color='white', font='courier',
    )

    for _ in range(HOLD_FRAMES):
        a = cam_angle[0]
        pl.camera.position    = (center[0],
                                 center[1] + cam_radius * np.cos(a),
                                 center[2] + cam_radius * np.sin(a))
        pl.camera.focal_point = tuple(center)
        pl.camera.up          = (1.0, 0.0, 0.0)
        pl.render()
        frames.append(pl.screenshot(return_img=True))
        cam_angle[0] += ORBIT_STEP

    print(f"  [frame {len(frames):4d}] {label}  IoU={iou:.4f}" if iou is not None
          else f"  [frame {len(frames):4d}] {label}")

# ── Run SMC (outputs go to temp dir so we don't touch the originals) ─────────
tmp = Path(tempfile.mkdtemp(prefix='smc_video_'))

print("=" * 60)
print("Phase 1 — SMC aggregate alignment")
print("=" * 60)
smc_align_aggregate(
    ref_paths    = ref_paths,
    sample_paths = sample_paths,
    output_dir   = str(tmp),
    seed         = 0,
    n_restarts   = N_RESTARTS,
    frame_cb     = frame_cb,
)

print("\n" + "=" * 60)
print("Phase 2 — SMC per-bone refinement")
print("=" * 60)
smc_align_per_bone(
    ref_paths    = ref_paths,
    sample_paths = [str(tmp / Path(p).name) for p in sample_paths],
    output_dir   = str(tmp),
    seed         = 0,
    n_restarts   = N_RESTARTS,
    frame_cb     = frame_cb,
)

pl.close()

# ── Encode video ─────────────────────────────────────────────────────────────
print(f"\nEncoding {len(frames)} frames → {VIDEO_OUT} ...")
iio.imwrite(str(VIDEO_OUT), frames, fps=FPS, codec='libx264',
            output_params=['-crf', '18', '-pix_fmt', 'yuv420p'])
size_mb = VIDEO_OUT.stat().st_size / 1e6
print(f"Done → {VIDEO_OUT}  ({size_mb:.1f} MB,  {len(frames)/FPS:.0f}s @ {FPS}fps)")

# clean up temp dir
import shutil
shutil.rmtree(tmp, ignore_errors=True)
