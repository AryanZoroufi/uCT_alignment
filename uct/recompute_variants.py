"""
Generate one-sided-lump variants at the per-scan Otsu threshold, varying W (vox)
and angular arc (deg), so the user can pick the closest to their paint.
Saves each mask + logs all as toggleable rerun layers on the see-through bone.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.ndimage import label as cc_label, binary_closing
import rerun as rr

OCC = Path(__file__).parent / "../bones_to_recon/B256M1_occ"
GT = 366110

cd = np.load("/tmp/sl_crop.npz")
hu = cd["hu"].astype(np.float32); nlo = cd["nlo"].astype(int); vmm = float(cd["vmm"])
c1, c4, a = cd["c1"], cd["c4"], cd["a"]; sep = float(cd["sep"]); center = sep / 2
THR_HU = float(cd["thr"]) if "thr" in cd.files else -736.0   # per-scan Otsu
bone = hu > THR_HU
bxyz = np.argwhere(bone) + nlo
t = (bxyz - c1) @ a
rel = (bxyz - c1) - np.outer(t, a)
tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
u = tmp - (tmp @ a) * a; u /= np.linalg.norm(u); v = np.cross(a, u)
ang = np.arctan2(rel @ v, rel @ u)


def dom_arc(in_band, arc_rad):
    aa = ang[in_band]; nb = 72
    h, edges = np.histogram(aa, bins=nb, range=(-np.pi, np.pi))
    w = max(1, int(round(arc_rad / (2 * np.pi) * nb)))
    hh = np.concatenate([h, h])
    bi = int(np.argmax([hh[i:i + w].sum() for i in range(nb)]))
    lo = edges[bi]
    return ((ang - lo) % (2 * np.pi)) <= arc_rad


def make(W, arc_deg):
    band = np.abs(t - center) <= W
    sel = band & dom_arc(band, np.deg2rad(arc_deg))
    g = np.zeros(bone.shape, bool); idx = bxyz[sel] - nlo
    g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    g = binary_closing(g, iterations=2) & bone
    gl, _ = cc_label(g)
    if gl.max() > 0:
        s = np.bincount(gl.ravel()); s[0] = 0; g = gl == int(np.argmax(s))
    return g


VARIANTS = [("W20_180", 20, 180), ("W30_120", 30, 120), ("W30_180", 30, 180),
            ("W30_240", 30, 240), ("W40_180", 40, 180)]
COLORS = {"W20_180": [60, 160, 255], "W30_120": [255, 220, 0], "W30_180": [235, 20, 20],
          "W30_240": [30, 220, 30], "W40_180": [200, 60, 235]}

full = trimesh.load(str(OCC / "sl.stl"), force="mesh", process=False)
fv = (full.vertices / 2).astype(np.float32)
rng = np.random.default_rng(0); fv = fv[rng.choice(len(fv), 150000, replace=False)]

rr.init("variants", spawn=True); print("init ok", flush=True)
rr.log("bone_seethrough", rr.Points3D(fv,
       colors=np.tile([180, 180, 180, 55], (len(fv), 1)).astype(np.uint8), radii=0.012))
for name, W, arc in VARIANTS:
    g = make(W, arc); n = int(g.sum())
    np.save(f"/tmp/var_{name}.npy", g)
    pts = (np.argwhere(g) + nlo).astype(np.float32) * vmm
    rr.log(f"v_{name}", rr.Points3D(pts, colors=np.tile(COLORS[name], (len(pts), 1)), radii=vmm * 1.2))
    print(f"{name}: W={W} arc={arc}deg -> {n:,} vox = {n*vmm**3:.4f} mm^3 ({n/GT:.2f}x GT)", flush=True)
print("done: toggle v_W20_180(blue) v_W30_120(yellow) v_W30_180(red) "
      "v_W30_240(green) v_W40_180(purple)", flush=True)
