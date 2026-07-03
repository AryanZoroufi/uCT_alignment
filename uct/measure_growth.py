"""
Reusable, tag-parameterized growth measurement.

Factored out of recompute_736.py with EXACT logic preserved:
  - Per-scan bone threshold = crop "thr" field (per-scan Otsu), -736 fallback.
  - bone = hu > thr.
  - Axial band about the gap centre: |t - center| <= W.
  - A = full cross-section = largest connected component of
        binary_closing(band, iters=2) & bone.
  - B = band & dominant 180-deg angular sector, same closing + largest-CC.
  - Sweep W in (10,15,20,25,30,40,50,60); pick W minimizing |vox - GT|.

Coordinate frame (matches the verified pipeline):
  bxyz = argwhere(bone) + nlo;  t = (bxyz - c1) @ a in [0, sep];
  center = sep/2;  rel = (bxyz - c1) - outer(t, a);  ang = atan2(rel@v, rel@u).
  Volume mm^3 = vox * vmm**3.

measure(tag, gt_vox=366110) -> {"A": {...}, "B": {...}, "thr": float}

Run directly to reproduce the SL regression and print an SL-vs-CL summary.
"""
import numpy as np
from scipy.ndimage import label as cc_label, binary_closing

GT_VOX_DEFAULT = 366110
W_SWEEP = (10, 15, 20, 25, 30, 40, 50, 60)


def _crop_path(tag):
    return f"/tmp/{tag}_crop.npz"


def measure(tag, gt_vox=GT_VOX_DEFAULT, save_masks=False):
    """Measure the gap-slab growth for a crop tagged `tag` (e.g. 'sl','cl').

    Returns {"A": {best_W,vox,mm3,ratio}, "B": {...}, "thr": float}.
    Ratio is vox/gt_vox. save_masks writes /tmp/measure_{tag}_{A,B}.npy.
    """
    d = np.load(_crop_path(tag))
    hu = d["hu"].astype(np.float32)
    nlo = d["nlo"].astype(int)
    vmm = float(d["vmm"])
    c1, c4, a = d["c1"], d["c4"], d["a"]
    sep = float(d["sep"])
    center = sep / 2

    thr = float(d["thr"]) if "thr" in d.files else -736.0   # per-scan Otsu
    bone = hu > thr

    bxyz = np.argwhere(bone) + nlo
    t = (bxyz - c1) @ a
    rel = (bxyz - c1) - np.outer(t, a)
    # orthonormal basis perpendicular to a for the angle
    tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = tmp - (tmp @ a) * a
    u /= np.linalg.norm(u)
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
            szs = np.bincount(gl.ravel())
            szs[0] = 0
            g = gl == int(np.argmax(szs))
        return g

    def dominant_sector(in_band, arc=np.pi):
        """boolean (over bxyz) selecting the densest arc-wide angular wedge."""
        aa = ang[in_band]
        nb = 72
        h, edges = np.histogram(aa, bins=nb, range=(-np.pi, np.pi))
        w = int(round(arc / (2 * np.pi) * nb))
        hh = np.concatenate([h, h])
        best_i = int(np.argmax([hh[i:i + w].sum() for i in range(nb)]))
        lo = edges[best_i]
        hi = lo + arc
        in_arc = ((ang - lo) % (2 * np.pi)) <= (hi - lo)
        return in_arc

    bestA = bestB = None   # (W, vox, mask)
    sweep = {}
    for W in W_SWEEP:
        band = np.abs(t - center) <= W
        gA = build(band)
        nA = int(gA.sum())
        sec = dominant_sector(band)
        gB = build(band & sec)
        nB = int(gB.sum())
        sweep[W] = {"A": nA, "B": nB}
        if bestA is None or abs(nA - gt_vox) < abs(bestA[1] - gt_vox):
            bestA = (W, nA, gA)
        if bestB is None or abs(nB - gt_vox) < abs(bestB[1] - gt_vox):
            bestB = (W, nB, gB)

    if save_masks:
        np.save(f"/tmp/measure_{tag}_A.npy", bestA[2])
        np.save(f"/tmp/measure_{tag}_B.npy", bestB[2])

    def pack(best):
        W, vox, _ = best
        return {"best_W": W, "vox": vox, "mm3": vox * vmm ** 3,
                "ratio": vox / gt_vox}

    return {"A": pack(bestA), "B": pack(bestB), "thr": thr, "vmm": vmm,
            "sweep": sweep, "gt_vox": gt_vox}


def qc_localization(cl_sweep, thresh=0.05):
    """Automatic gap-localization QC from the CONTROL (CL) gap-band sweep.

    A correctly-localized gap is near-EMPTY at the tight window and only fills as
    the band widens into the parts (B256M1 CL: W10=122 vox = 0.03% of W60). If the
    tight-window CL band is already a large fraction of the wide-window band, the
    gap centre is landing INSIDE bone -> parts 1/4 mislocalized (B256M7 CL: W10 is
    20% of W60). Returns dict(w_tight,w_wide,cl_tight,cl_wide,fill,localized).
    """
    ws = sorted(cl_sweep)
    wt, ww = ws[0], ws[-1]
    ct, cw = cl_sweep[wt]["A"], cl_sweep[ww]["A"]
    fill = ct / max(cw, 1)
    return dict(w_tight=wt, w_wide=ww, cl_tight=ct, cl_wide=cw,
                fill=fill, localized=fill < thresh)


if __name__ == "__main__":
    GT = GT_VOX_DEFAULT

    sl = measure("sl", gt_vox=GT, save_masks=True)
    cl = measure("cl", gt_vox=GT, save_masks=True)
    vmm = sl["vmm"]
    slA, slB = sl["A"], sl["B"]

    print(f"[SL] thr={sl['thr']:.1f}HU   [CL] thr={cl['thr']:.1f}HU   (per-scan Otsu)")
    print(f"[SL] A full-xsec best: W={slA['best_W']:2d} -> {slA['vox']:,} vox "
          f"= {slA['mm3']:.4f} mm^3 ({slA['ratio']:.2f}x GT)")
    print(f"[SL] B one-sided best: W={slB['best_W']:2d} -> {slB['vox']:,} vox "
          f"= {slB['mm3']:.4f} mm^3 ({slB['ratio']:.2f}x GT)")
    print(f"[SL] GT (paint): {GT:,} vox", flush=True)

    REG_EXPECT = 362775
    reg_ok = abs(slA["vox"] - REG_EXPECT) <= 2000
    print(f"[SL] REGRESSION A: expect ~{REG_EXPECT:,}, got {slA['vox']:,} "
          f"-> {'PASS' if reg_ok else 'FAIL'}", flush=True)

    # --- HONEST biological control: evaluate CL at SL's OWN gap window (fixed W).
    #     Do NOT use CL's argmin-vs-GT: with no growth, CL bone grows
    #     monotonically with W and its argmin spuriously lands at W=60, where the
    #     wide band engulfs the native tibia cross-sections (looks bigger than SL,
    #     which is a metric artifact, not biology). ---
    WA = slA["best_W"]
    print(f"\n=== CONTROL: SL vs CL at the SAME window (fixed W={WA}) ===")
    print(f"{'shape':11} | {'SL vox':>9} | {'CL vox':>9} | {'SL/CL':>7}")
    for lbl, key in (("A full-xsec", "A"), ("B one-sided", "B")):
        s = sl["sweep"][WA][key]
        c = cl["sweep"][WA][key]
        r = f"{s/c:.0f}x" if c else "inf"
        print(f"{lbl:11} | {s:9,} | {c:9,} | {r:>7}")
    print(" CL has no ectopic growth -> the mid-gap band is near-empty; the large\n"
          " SL/CL ratio IS the growth signal.", flush=True)

    print(f"\n=== full A-volume sweep over W (vox) — fixed-W, honest baseline ===")
    print(f"{'W':>4} | {'SL-A':>10} | {'CL-A':>10} | {'SL/CL':>7}")
    for W in W_SWEEP:
        s = sl["sweep"][W]["A"]
        c = cl["sweep"][W]["A"]
        r = f"{s/c:.0f}x" if c else "inf"
        print(f"{W:4d} | {s:10,} | {c:10,} | {r:>7}")
    print("NOTE: lock ONE window for the whole cohort; never let CL pick its own "
          "argmin (it fills into the parts as W grows).", flush=True)

    # --- automatic gap-localization QC gate (control must be near-empty at W_tight) ---
    q = qc_localization(cl["sweep"])
    verdict = "PASS (gap localized)" if q["localized"] else \
        "FAIL (gap likely MISLOCALIZED -> manual QC)"
    print(f"\n=== QC GAP-LOCALIZATION GATE ===", flush=True)
    print(f"CL-A(W={q['w_tight']}) = {q['cl_tight']:,} vox = {100*q['fill']:.2f}% of "
          f"CL-A(W={q['w_wide']}) = {q['cl_wide']:,}   (pass if < 5%)  ->  {verdict}",
          flush=True)
    if not q["localized"]:
        print("  The contralateral gap band is NOT near-empty at the tight window: the "
              "gap centre is inside bone (parts 1/4 mislocalized). Do NOT trust this "
              "pair's growth number -- re-segment or hand-correct parts 1 & 4.", flush=True)
