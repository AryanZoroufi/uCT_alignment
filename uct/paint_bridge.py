"""
3D voxel-selection tool (Dragonfly-style) for the tibia bridge/growth.

Solid bone at the per-scan bone threshold (Otsu). A spherical brush selects bone
VOXELS in 3D within its radius. The brush is sized in SCREEN PIXELS (constant
on-screen size, converted to a world radius from the camera distance each frame)
so it does NOT change size when you zoom -- zooming just lets you select finer
detail. Selected voxels show as red; volume is measured live and saved on exit.
Auto-resumes prior selection.

Controls
  p        toggle SELECT mode (on=left-drag selects; off=left-drag rotates)
  b / n    bigger / smaller brush (screen pixels)
  scroll   zoom (both modes)
  t        toggle bone transparency (to see selected voxels inside)
  e        eraser toggle      c  clear      m  measure      s  save
  q / close window            finish & save
Saves /tmp/painted_bridge.npy (crop mask) + /tmp/painted_bridge_meta.npy.
"""
from pathlib import Path
import numpy as np
import trimesh
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

OCC = Path(__file__).parent / "../bones_to_recon/B256M1_occ"
STEP, GT = 2, 366110

cd = np.load("/tmp/sl_crop.npz")
hu = cd["hu"].astype(np.float32); nlo = cd["nlo"].astype(int); vmm = float(cd["vmm"])
c1, c4 = cd["c1"], cd["c4"]
THR_HU = float(cd["thr"]) if "thr" in cd.files else -736.0   # per-scan Otsu
bone3d = hu > THR_HU
shape = bone3d.shape
painted3d = np.zeros(shape, bool)

_pp = Path("/tmp/painted_bridge.npy")
if _pp.exists():
    try:
        prev = np.load(_pp)
        if prev.shape == shape and prev.any():
            painted3d |= prev
            print(f"resumed {int(painted3d.sum()):,} vox ('c' clears)", flush=True)
    except Exception as ex:
        print("resume failed:", ex, flush=True)

full = trimesh.load(str(OCC / "sl.stl"), force="mesh", process=False)
V = (full.vertices / STEP).astype(np.float64)
F = full.faces.astype(np.int64)
mesh = pv.PolyData(V, np.hstack([np.full((len(F), 1), 3, np.int64), F]).ravel())

screen_px = [38]; paint_mode = [False]; erase = [False]; painting = [False]
transp = [False]; last_pos = [((c1 + c4) / 2) * vmm]

pl = pv.Plotter(window_size=(1300, 950))
bone_actor = pl.add_mesh(mesh, color=[0.80, 0.69, 0.55], smooth_shading=True, name="bone")
try:
    pl.enable_eye_dome_lighting()
except Exception:
    pass

_unit = pv.Sphere(radius=1.0, theta_resolution=18, phi_resolution=18)
_tpts = _unit.points.copy()
brush = pv.PolyData(_unit.points.copy(), _unit.faces)
brush_actor = pl.add_mesh(brush, color="cyan", style="wireframe", line_width=2, name="brush")
sel_name = "sel"


def world_radius():
    pos = np.asarray(last_pos[0], float)
    cam = pl.camera
    d = np.linalg.norm(np.asarray(cam.position) - pos)
    vh = max(pl.window_size[1], 10)
    wpp = 2.0 * d * np.tan(np.radians(cam.view_angle) / 2.0) / vh
    return max(screen_px[0] * wpp, vmm)


def update_brush():
    brush.points = (_tpts * world_radius() + np.asarray(last_pos[0], float)).astype(np.float32)


def update_sel():
    pl.remove_actor(sel_name, render=False)
    if painted3d.any():
        pts = (np.argwhere(painted3d) + nlo).astype(np.float32) * vmm
        pl.add_mesh(pv.PolyData(pts), color="red", point_size=7,
                    render_points_as_spheres=True, name=sel_name)


update_brush(); update_sel()
picker = vtk.vtkCellPicker(); picker.SetTolerance(0.0008)
iren = pl.iren.interactor
style_cam = vtk.vtkInteractorStyleTrackballCamera()
style_user = vtk.vtkInteractorStyleUser()
iren.SetInteractorStyle(style_cam)
txt = pl.add_text("", position="upper_left", font_size=10, name="hud")


def hud():
    n = int(painted3d.sum())
    mode = ("SELECT" + ("/ERASE" if erase[0] else "")) if paint_mode[0] else "ROTATE"
    txt.set_text("upper_left",
        f"[{mode}]  brush={screen_px[0]}px   selected={n:,} vox = {n*vmm**3:.4f} mm^3"
        f"  (GT~{GT*vmm**3:.3f})\n"
        f"p=select/rotate  b/n=brush  scroll=zoom  t=transparency  e=erase  c=clear  q=quit")


def paint_at(pos):
    p = np.asarray(pos, float)
    R = world_radius(); r = max(1, int(round(R / vmm)))
    ni = np.round(p / vmm).astype(int) - nlo
    z0, z1 = max(ni[0]-r, 0), min(ni[0]+r+1, shape[0])
    y0, y1 = max(ni[1]-r, 0), min(ni[1]+r+1, shape[1])
    x0, x1 = max(ni[2]-r, 0), min(ni[2]+r+1, shape[2])
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    region = ((zz-ni[0])**2 + (yy-ni[1])**2 + (xx-ni[2])**2 <= r*r) \
        & bone3d[z0:z1, y0:y1, x0:x1]
    if erase[0]:
        painted3d[z0:z1, y0:y1, x0:x1] &= ~region
    else:
        painted3d[z0:z1, y0:y1, x0:x1] |= region


def on_move(o, e):
    x, y = iren.GetEventPosition()
    picker.Pick(x, y, 0, pl.renderer)
    if picker.GetCellId() < 0:
        return
    last_pos[0] = np.asarray(picker.GetPickPosition())
    update_brush()
    if paint_mode[0] and painting[0]:
        paint_at(last_pos[0])
    pl.render()


def on_press(o, e):
    painting[0] = True
    if paint_mode[0]:
        on_move(o, e)


def on_release(o, e):
    painting[0] = False
    if paint_mode[0]:
        update_sel()
    hud(); pl.render()


def on_wheel_f(o, e):
    if paint_mode[0]:
        pl.camera.Zoom(1.1); update_brush(); pl.render()


def on_wheel_b(o, e):
    if paint_mode[0]:
        pl.camera.Zoom(1 / 1.1); update_brush(); pl.render()


iren.AddObserver("MouseMoveEvent", on_move)
iren.AddObserver("LeftButtonPressEvent", on_press)
iren.AddObserver("LeftButtonReleaseEvent", on_release)
iren.AddObserver("MouseWheelForwardEvent", on_wheel_f)
iren.AddObserver("MouseWheelBackwardEvent", on_wheel_b)


def toggle_paint():
    paint_mode[0] = not paint_mode[0]
    iren.SetInteractorStyle(style_user if paint_mode[0] else style_cam)
    print("SELECT" if paint_mode[0] else "ROTATE", flush=True); hud(); pl.render()


def toggle_erase():
    erase[0] = not erase[0]
    brush_actor.GetProperty().SetColor((1, 0.6, 0) if erase[0] else (0, 1, 1))
    hud(); pl.render()


def toggle_transp():
    transp[0] = not transp[0]
    bone_actor.GetProperty().SetOpacity(0.3 if transp[0] else 1.0)
    pl.render()


def bigger():
    screen_px[0] = min(screen_px[0] + 6, 200); update_brush(); hud(); pl.render()


def smaller():
    screen_px[0] = max(screen_px[0] - 6, 6); update_brush(); hud(); pl.render()


def clear_all():
    painted3d[:] = False; update_sel(); hud(); pl.render()


def measure():
    n = int(painted3d.sum()); print(f"{n:,} vox = {n*vmm**3:.4f} mm^3", flush=True)


def save():
    np.save("/tmp/painted_bridge.npy", painted3d)
    np.save("/tmp/painted_bridge_meta.npy", np.array([*nlo, vmm], dtype=np.float64))
    n = int(painted3d.sum()); print(f"SAVED {n:,} vox = {n*vmm**3:.4f} mm^3", flush=True)


for keys, fn in [(["b", "plus", "equal"], bigger), (["n", "minus"], smaller),
                 (["p"], toggle_paint), (["e"], toggle_erase), (["t"], toggle_transp),
                 (["c"], clear_all), (["m"], measure), (["s"], save)]:
    for k in keys:
        pl.add_key_event(k, fn)

pl.camera.focal_point = tuple(((c1 + c4) / 2) * vmm)
hud()
print("ready. 'p'=SELECT mode, left-drag to select bone voxels; 'p' to rotate. "
      "b/n brush, scroll zoom, t transparency. Close to finish.", flush=True)
pl.show()
save()
print("done.", flush=True)
