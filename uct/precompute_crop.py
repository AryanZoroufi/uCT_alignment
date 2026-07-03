"""
Precompute the cropped bone data for B256M1 SL (and CL) ONCE so refinement
strategies can iterate on small files instead of re-loading the 1.3GB VOX.

Also prints diagnostics on the W=80 midpoint band to reveal WHERE the
over-capture is: axial (band reaching into the parts' facing surfaces) vs
lateral (full annular cross-section vs a localized plate).
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).parent))
from vox_to_stl import load_vox
from skimage.filters import threshold_otsu
import growth_config as cfg

# Which pair to crop: `python precompute_crop.py [PAIR]` (default B256M1).
PAIR = sys.argv[1] if len(sys.argv) > 1 else "B256M1"
_conf = cfg.get(PAIR)
OCC = Path(_conf["occ"])
VOX = {"cl": _conf["cl_vox"], "sl": _conf["sl_vox"]}
STEP = _conf["step"]
GT_VOX = _conf["gt_vox"]
print(f"pair={PAIR}  occ={OCC}", flush=True)


def prep(tag, save_hu=False):
    grid, vmm = load_vox(str(VOX[tag]))
    # Per-scan bone threshold = Otsu on the FULL-volume downsample. Validated
    # 2026-07-02 to match the human's Dragonfly pick within +-2 HU (mean bias 0)
    # across 8 scans. Must be computed on the full grid, NOT the crop (whose
    # bone-heavy histogram would shift Otsu).
    thr = float(threshold_otsu(grid[::4, ::4, ::4]))
    p1 = trimesh.load(str(OCC / f"{tag}_part1.stl"), force="mesh", process=False)
    p4 = trimesh.load(str(OCC / f"{tag}_part4.stl"), force="mesh", process=False)
    n1 = p1.vertices / (vmm * STEP); n4 = p4.vertices / (vmm * STEP)
    c1, c4 = n1.mean(0), n4.mean(0)
    a = (c4 - c1); sep = float(np.linalg.norm(a)); a = a / sep
    mid = (c1 + c4) / 2
    half = 0.85 * np.abs(c4 - c1) + 45
    nlo = np.maximum(np.floor(mid - half).astype(int), 0)
    nhi = np.minimum(np.ceil(mid + half).astype(int), np.array(grid.shape))
    hu_crop = grid[nlo[0]:nhi[0], nlo[1]:nhi[1], nlo[2]:nhi[2]].astype(np.float32)
    bone = hu_crop > thr
    del grid
    out = f"/tmp/{tag}_crop.npz"
    fields = dict(bone=bone, nlo=nlo, vmm=vmm, n1=n1, n4=n4,
                  c1=c1, c4=c4, sep=sep, a=a, thr=thr)
    if save_hu:
        fields["hu"] = hu_crop.astype(np.float16)
    np.savez_compressed(out, **fields)
    print(f"[{tag}] saved {out}  bone crop {bone.shape}  "
          f"{int(bone.sum()):,} vox  otsu_thr={thr:.0f}HU  sep={sep:.0f}  "
          f"hu={save_hu}", flush=True)
    return bone, nlo, vmm, n1, n4, c1, c4, sep, a


def diag_sl(bone, nlo, vmm, n1, n4, c1, c4, sep, a):
    bxyz = np.argwhere(bone) + nlo
    t = (bxyz - c1) @ a
    center = sep / 2
    # axis-perp distance from the c1->c4 line
    rel = (bxyz - c1) - np.outer(t, a)
    rad = np.linalg.norm(rel, axis=1)
    W = 80
    band = np.abs(t - center) <= W
    gt_s = f"{GT_VOX:,}" if GT_VOX else "?"
    print(f"\n[SL] W={W} band: {int(band.sum()):,} vox (GT {gt_s})", flush=True)
    print("  axial profile |t-center| (vox) -> bone count per 10-vox shell:", flush=True)
    bt = np.abs(t[band] - center)
    for lo in range(0, W, 10):
        c = int(((bt >= lo) & (bt < lo + 10)).sum())
        print(f"    {lo:3d}-{lo+10:3d}: {c:8,} {'#'*(c//1500)}", flush=True)
    print("  lateral profile radius-from-axis (vox) within band:", flush=True)
    br = rad[band]
    for lo in range(0, int(br.max()) + 15, 15):
        c = int(((br >= lo) & (br < lo + 15)).sum())
        print(f"    {lo:3d}-{lo+15:3d}: {c:8,} {'#'*(c//1500)}", flush=True)
    print(f"  radius: median={np.median(br):.0f} p90={np.percentile(br,90):.0f} "
          f"max={br.max():.0f} vox ({br.max()*vmm:.2f}mm)", flush=True)


if __name__ == "__main__":
    sl = prep("sl", save_hu=True)
    prep("cl", save_hu=True)
    diag_sl(*sl)
