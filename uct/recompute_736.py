"""
Redo the growth measurement at the per-scan bone threshold.
Threshold = per-scan Otsu, stored in the crop npz by precompute_crop.py;
validated 2026-07-02 to match the human's Dragonfly pick within +-2 HU
(mean bias 0) across 8 scans. (Was hardcoded -736 = Otsu for B256M1 SL.)

Two region definitions, each swept to hit GT (366,110 vox):
  A) midpoint band, full cross-section
  B) midpoint band + dominant 180-deg angular sector (one-sided lump)
Saves both masks for visual comparison; reports voxels & mm^3.
"""
import numpy as np
from scipy.ndimage import label as cc_label, binary_closing

GT_VOX, GT_MM3 = 366110, 0.101363

d = np.load("/tmp/sl_crop.npz")
hu = d["hu"].astype(np.float32)
nlo = d["nlo"].astype(int); vmm = float(d["vmm"])
c1, c4, a = d["c1"], d["c4"], d["a"]; sep = float(d["sep"]); center = sep / 2

THR_HU = float(d["thr"]) if "thr" in d.files else -736.0   # per-scan Otsu
bone = hu > THR_HU
print(f"bone(>{THR_HU:.0f}HU, per-scan Otsu) = {int(bone.sum()):,} vox in crop",
      flush=True)

bxyz = np.argwhere(bone) + nlo
t = (bxyz - c1) @ a
rel = (bxyz - c1) - np.outer(t, a)
rad = np.linalg.norm(rel, axis=1)
# orthonormal basis perpendicular to a for angle
tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
u = tmp - (tmp @ a) * a; u /= np.linalg.norm(u)
v = np.cross(a, u)
ang = np.arctan2(rel @ v, rel @ u)        # -pi..pi


def build(sel, close=True):
    g = np.zeros(bone.shape, bool)
    idx = bxyz[sel] - nlo
    g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if close:
        g = binary_closing(g, iterations=2) & bone
    gl, _ = cc_label(g)
    if gl.max() > 0:
        szs = np.bincount(gl.ravel()); szs[0] = 0
        g = gl == int(np.argmax(szs))
    return g


def dominant_sector(in_band, arc=np.pi):
    """boolean (over bxyz) selecting the densest `arc`-wide angular wedge among band voxels."""
    aa = ang[in_band]
    # histogram of angle, find arc-window with most voxels
    nb = 72
    h, edges = np.histogram(aa, bins=nb, range=(-np.pi, np.pi))
    w = int(round(arc / (2 * np.pi) * nb))
    hh = np.concatenate([h, h])
    best_i = int(np.argmax([hh[i:i + w].sum() for i in range(nb)]))
    lo = edges[best_i]; hi = lo + arc
    a2 = ang.copy()
    # wrap so [lo,hi] is contiguous
    in_arc = ((a2 - lo) % (2 * np.pi)) <= (hi - lo)
    return in_arc


print(f"\n{'W':>4} | {'A full-xsec':>20} | {'B one-sided(180deg)':>24}", flush=True)
bestA = bestB = None
for W in (10, 15, 20, 25, 30, 40, 50, 60):
    band = np.abs(t - center) <= W
    gA = build(band); nA = int(gA.sum())
    sec = dominant_sector(band)
    gB = build(band & sec); nB = int(gB.sum())
    print(f"{W:4d} | {nA:8,} {nA*vmm**3:7.4f} {nA/GT_VOX:5.2f}x | "
          f"{nB:8,} {nB*vmm**3:7.4f} {nB/GT_VOX:5.2f}x", flush=True)
    if bestA is None or abs(nA - GT_VOX) < abs(bestA[1] - GT_VOX): bestA = (W, nA, gA)
    if bestB is None or abs(nB - GT_VOX) < abs(bestB[1] - GT_VOX): bestB = (W, nB, gB)

np.save("/tmp/cand736_fullxsec.npy", bestA[2])
np.save("/tmp/cand736_oneside.npy", bestB[2])
print(f"\nbest A full-xsec: W={bestA[0]} -> {bestA[1]:,} vox = {bestA[1]*vmm**3:.4f} mm^3", flush=True)
print(f"best B one-sided: W={bestB[0]} -> {bestB[1]:,} vox = {bestB[1]*vmm**3:.4f} mm^3", flush=True)
print(f"GT = {GT_VOX:,} vox = {GT_MM3} mm^3", flush=True)
print("saved /tmp/cand736_fullxsec.npy and /tmp/cand736_oneside.npy", flush=True)
