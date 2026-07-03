"""
2D slice painter (Dragonfly-style) for the tibia bridge/growth.

Scroll through cross-section slices (sliced along the axis where the two tibia
parts are most separated, so the gap/bridge sits in the middle slices). Paint
with a circular brush; only bone above the per-scan bone threshold (Otsu) is marked.
The brush is a 3D ball, so each stroke also fills +-radius neighbouring slices.
Volume is measured live and saved on exit; auto-resumes prior paint.

Controls
  left-drag      paint            right-drag / hold E   erase
  scroll, j / k, arrows   change slice (J/K = jump 5)
  [ / ]          brush smaller / bigger
  c   clear      m  measure       s  save        q / close  finish & save
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
for _bk in ("Qt5Agg", "QtAgg", "TkAgg"):
    try:
        matplotlib.use(_bk); break
    except Exception:
        continue
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

GT = 366110
cd = np.load("/tmp/sl_crop.npz")
hu = cd["hu"].astype(np.float32); nlo = cd["nlo"].astype(int); vmm = float(cd["vmm"])
c1, c4 = cd["c1"], cd["c4"]
THR_HU = float(cd["thr"]) if "thr" in cd.files else -736.0   # per-scan Otsu

SAX = int(np.argmax(np.abs(c4 - c1)))            # slice axis = max part separation
vol = np.moveaxis(hu, SAX, 0)                    # (S, H, W)
bone = vol > THR_HU
S, H, W = vol.shape
painted = np.zeros((S, H, W), bool)

_pp = Path("/tmp/painted_bridge.npy")
if _pp.exists():
    try:
        prev = np.moveaxis(np.load(_pp), SAX, 0)
        if prev.shape == painted.shape and prev.any():
            painted |= prev
            print(f"resumed {int(painted.sum()):,} vox", flush=True)
    except Exception as ex:
        print("resume failed:", ex, flush=True)

cmid = (c1 + c4) / 2
cur = [int(round(cmid[SAX] - nlo[SAX]))]         # start at gap midpoint slice
cur[0] = int(np.clip(cur[0], 0, S - 1))
R = [6]                                          # brush radius (voxels)
erasing = [False]; dragging = [False]

fig, ax = plt.subplots(figsize=(9, 9))
plt.subplots_adjust(bottom=0.06, top=0.93)
im_base = ax.imshow(vol[cur[0]], cmap="gray", vmin=-950, vmax=-300, interpolation="nearest")
ov = np.zeros((H, W, 4), np.float32)
im_over = ax.imshow(ov, interpolation="nearest")
brush_c = Circle((W / 2, H / 2), R[0], fill=False, color="cyan", lw=1.5)
ax.add_patch(brush_c)
ax.set_xticks([]); ax.set_yticks([])


def overlay(s):
    o = np.zeros((H, W, 4), np.float32)
    m = painted[s]
    o[m] = [1, 0.1, 0.1, 0.55]
    return o


def title():
    n = int(painted.sum())
    ax.set_title(f"slice {cur[0]+1}/{S} (axis {SAX})   brush={R[0]}vox   "
                 f"{'ERASE' if erasing[0] else 'PAINT'}   "
                 f"painted={n:,} vox = {n*vmm**3:.4f} mm^3  (GT {GT*vmm**3:.3f})",
                 fontsize=10)


def refresh(slice_changed=False):
    if slice_changed:
        im_base.set_data(vol[cur[0]])
    im_over.set_data(overlay(cur[0]))
    title(); fig.canvas.draw_idle()


def paint_at(row, col):
    s = cur[0]
    r = R[0]
    z0, z1 = max(s-r, 0), min(s+r+1, S)
    y0, y1 = max(row-r, 0), min(row+r+1, H)
    x0, x1 = max(col-r, 0), min(col+r+1, W)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    ball = (zz-s)**2 + (yy-row)**2 + (xx-col)**2 <= r*r
    region = ball & bone[z0:z1, y0:y1, x0:x1]
    if erasing[0]:
        painted[z0:z1, y0:y1, x0:x1] &= ~region
    else:
        painted[z0:z1, y0:y1, x0:x1] |= region


def on_press(e):
    if e.inaxes != ax or e.xdata is None:
        return
    dragging[0] = True
    erasing[0] = erasing[0] or (e.button == 3)
    paint_at(int(round(e.ydata)), int(round(e.xdata))); refresh()


def on_release(e):
    dragging[0] = False
    if e.button == 3:
        erasing[0] = False


def on_motion(e):
    if e.inaxes != ax or e.xdata is None:
        return
    brush_c.center = (e.xdata, e.ydata); brush_c.set_radius(R[0])
    if dragging[0]:
        paint_at(int(round(e.ydata)), int(round(e.xdata))); refresh()
    else:
        fig.canvas.draw_idle()


def set_slice(d):
    cur[0] = int(np.clip(cur[0] + d, 0, S - 1)); refresh(slice_changed=True)


def on_scroll(e):
    set_slice(1 if e.step > 0 else -1)


def save():
    np.save("/tmp/painted_bridge.npy", np.moveaxis(painted, 0, SAX))
    np.save("/tmp/painted_bridge_meta.npy", np.array([*nlo, vmm], dtype=np.float64))
    n = int(painted.sum())
    print(f"SAVED {n:,} vox = {n*vmm**3:.4f} mm^3", flush=True)


def on_key(e):
    k = e.key
    if k in ("j", "down", "left"): set_slice(-1)
    elif k in ("k", "up", "right"): set_slice(1)
    elif k == "J": set_slice(-5)
    elif k == "K": set_slice(5)
    elif k == "]": R[0] = min(R[0] + 1, 60); brush_c.set_radius(R[0]); title(); fig.canvas.draw_idle()
    elif k == "[": R[0] = max(R[0] - 1, 1); brush_c.set_radius(R[0]); title(); fig.canvas.draw_idle()
    elif k == "e": erasing[0] = not erasing[0]; title(); fig.canvas.draw_idle()
    elif k == "c": painted[:] = False; refresh()
    elif k == "m": n = int(painted.sum()); print(f"{n:,} vox = {n*vmm**3:.4f} mm^3", flush=True)
    elif k == "s": save()


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("key_press_event", on_key)
refresh(slice_changed=True)
print(f"ready: {S} slices along axis {SAX}, gap near slice {cur[0]+1}. "
      "left-drag to paint bone; scroll/jk to change slice; [ ] brush; e erase; "
      "c clear; close window to finish.", flush=True)
plt.show()
save()
print("done.", flush=True)
