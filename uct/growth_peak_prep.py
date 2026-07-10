"""
Native-voxel precompute for the growth-peak procedure. Everything is done in ONE common
SL native frame, in PHYSICAL mm (native_idx * vmm).

Per pair:
  * Load SL & CL VOX, Otsu threshold, crop, pore-fill (close), keep tibia component(s),
    label each native bone voxel part1/part4 by the nearest part-mesh.
  * SL frame: axis a = part1->part4 (height increases toward part4), lateral u,w; origin c1.
  * Overlay CL -> SL (mirror-aware 7-DOF chamfer) so both legs share the XZ grid.
  * Two growth heatmaps per XZ cell (positive = growth):
        bone1 top    = SL_top1 - CL_top1      (part 1, gap-facing top surface)
        bone4 bottom = CL_bot4 - SL_bot4      (part 4, gap-facing bottom surface)
    each cell carries its 3D surface position (on the SURGICAL front) and lateral (u,w).
  * Connection point = centre of the closest SL part1<->part4 interface.
  * Native connectivity (part1 & part4 in one component / closest approach).

Writes peak_data.pkl  ->  consumed by growth_peak.py (the fast, tunable procedure).
"""
import sys
import os
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
import trimesh
import fast_simplification
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage import measure
sys.path.insert(0, str(Path(__file__).parent))
from vox_to_stl import load_vox
from atlas_register import register_atlas_7dof_chamfer
import growth_config as cfg

PAIRS = ["B256M1", "B256M7"]
S = Path("/tmp/claude-1000/-home-aryan-Projects-uct-backup-uCT-alignment/"
         "2243f74f-59e2-4926-a697-e6fdd62c17ea/scratchpad")
STRUCT = np.ones((3, 3, 3))
CLOSE = 3          # pore-fill (~19 µm), not the 0.5 mm gap
SMOOTH_ITERS = 90  # Taubin smoothing on the gap-facing surfaces (user-chosen level)
DECIM_FACES = 55000  # decimate marching-cubes surface to ~atlas-mesh resolution before Taubin
# Everything is in OCC-MESH coordinates (= native_idx * vmm * STEP = physical_mm * STEP),
# matching the working envelope overlay. physical mm = mesh / STEP (STEP = 2).
CELL = 0.24        # XZ cell size in mesh units (= 0.12 mm)

# Atlas (align_7dof) — the native parts are segmented from the SAME atlas as the display, NOT the
# occ part-meshes (which disagree near the junction and mislocate the connection).
SEG_VER = 3        # bump to invalidate leg caches when the segmentation/crop changes
_ATL = np.load(str(Path(__file__).parent / "atlas.npz"), allow_pickle=True)
_solid = _ATL["solid"]; _pitch = float(_ATL["pitch"]); _origin = _ATL["origin"].astype(np.float64)
_avm, _af, _an, _av = measure.marching_cubes(np.pad(_solid, 1).astype(np.float32), 0.5)
_avm = (_avm - 1) * _pitch + _origin
_aidx = np.clip(np.round((_avm - _origin) / _pitch).astype(int), 0, np.array(_solid.shape) - 1)
_vlab = _ATL["bone_labels"][_aidx[:, 0], _aidx[:, 1], _aidx[:, 2]].astype(int)
_Asurf = _ATL["surface_points"].astype(np.float64)


def euler_R(r):
    rx, ry, rz = r
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def refine_7dof(src, tgt, ncand=1000, iters=2, seed=0):
    r = np.random.default_rng(seed); c = src.mean(0); tt = cKDTree(tgt)
    def app(P, p): M = euler_R(p[3:6]) * np.exp(p[6]); return (P - c) @ M.T + c + p[0:3]
    def score(p):
        T = app(src, p); return 0.5 * (tt.query(T)[0].mean() + cKDTree(T).query(tgt)[0].mean())
    bp = np.zeros(7); bs = score(bp)
    RG = {"t": [2., 2, 2], "r": [.4, .4, .4], "s": [.4], "j": [1., 1, 1, .17, .17, .17, .1]}
    for _ in range(iters):
        for st, ix in (("t", [0, 1, 2]), ("r", [3, 4, 5]), ("s", [6]), ("j", list(range(7)))):
            P = np.tile(bp, (ncand, 1)); P[:, ix] = bp[ix] + r.uniform(-1, 1, (ncand, len(ix))) * np.array(RG[st])
            for p in P:
                s = score(p)
                if s < bs: bs, bp = s, p.copy()
    return bp, c


def apply4(T, V): return (V @ T[:3, :3].T) + T[:3, 3]


def comp_seed(lab, seeds, shape):
    s = np.round(seeds).astype(int); s = s[(s >= 0).all(1) & (s < np.array(shape)).all(1)]
    L = lab[s[:, 0], s[:, 1], s[:, 2]]; L = L[L > 0]
    return Counter(L).most_common(1)[0][0] if len(L) else -1


def smooth_surf(mask, shape, nlo, vmm, step):
    """Binary voxel set -> marching-cubes surface -> Taubin-smoothed vertices (MESH coords)."""
    b = np.zeros(shape, bool); b[mask[:, 0], mask[:, 1], mask[:, 2]] = True
    b = ndimage.binary_closing(b, structure=STRUCT, iterations=1)      # weld the voxel set
    v, f, _, _ = measure.marching_cubes(np.pad(b, 1).astype(np.float32), 0.5); v -= 1
    m = trimesh.Trimesh(v, f, process=False); m.merge_vertices()
    if len(m.faces) > DECIM_FACES:
        dv, df = fast_simplification.simplify(m.vertices.astype(np.float64), m.faces.astype(np.int32),
                                              target_reduction=1 - DECIM_FACES / len(m.faces))
        m = trimesh.Trimesh(dv, df, process=False); m.merge_vertices()
    trimesh.smoothing.filter_taubin(m, iterations=SMOOTH_ITERS)
    return (m.vertices + nlo) * (vmm * step)                          # MESH coords


def atlas_seg(mesh):
    """Segment a bone MESH into part1/part4 vertex sets via the atlas (7-DOF chamfer + refine),
    identical to the display pipeline. Returns {1: verts, 4: verts} in the mesh's own frame."""
    r = np.random.default_rng(0)
    N = mesh.vertices[r.choice(len(mesh.vertices), min(4000, len(mesh.vertices)), replace=False)].astype(np.float64)
    T, _info = register_atlas_7dof_chamfer(_Asurf, N)
    av7 = apply4(T, _avm)
    ss = apply4(T, _Asurf[r.choice(len(_Asurf), 3000, replace=False)])
    dp, c7 = refine_7dof(ss, N[r.choice(len(N), 3000, replace=False)])
    M = euler_R(dp[3:6]) * np.exp(dp[6]); avf = (av7 - c7) @ M.T + c7 + dp[0:3]
    m = _vlab > 0; _, nn = cKDTree(avf[m]).query(mesh.vertices); vl = _vlab[m][nn].astype(int)
    out = {}
    for L in (1, 4):
        ff = mesh.faces[(vl[mesh.faces] == L).sum(1) >= 2]
        out[L] = mesh.vertices[np.unique(ff)] if len(ff) else mesh.vertices[vl == L]
    return out


def load_leg(pair, tag):
    conf = cfg.get(pair); OCC = Path(conf["occ"]); STEP = conf["step"]
    grid, vmm = load_vox(conf[f"{tag}_vox"])
    thr = float(threshold_otsu(grid[::4, ::4, ::4]))
    mesh = trimesh.load(str(OCC / f"{tag}.stl"), force="mesh", process=False); mesh.merge_vertices()
    seg = atlas_seg(mesh); p1m = seg[1]; p4m = seg[4]              # atlas-segmented parts (not occ meshes)
    n1 = p1m / (vmm * STEP); n4 = p4m / (vmm * STEP)                # part meshes -> native idx
    alln = np.vstack([n1, n4])                                      # crop to the atlas parts' full
    nlo = np.maximum(np.floor(alln.min(0)).astype(int) - 45, 0)     # bbox + margin (was an
    nhi = np.minimum(np.ceil(alln.max(0)).astype(int) + 45, np.array(grid.shape))  # anisotropic
    #                                                                sep-based crop that clipped width)
    bone = grid[nlo[0]:nhi[0], nlo[1]:nhi[1], nlo[2]:nhi[2]] > thr; del grid
    bone = ndimage.binary_closing(bone, structure=STRUCT, iterations=CLOSE)
    lab, _ = ndimage.label(bone, structure=STRUCT)
    m1 = comp_seed(lab, n1 - nlo, bone.shape); m4 = comp_seed(lab, n4 - nlo, bone.shape)
    keep = np.isin(lab, [x for x in (m1, m4) if x > 0])
    vox = np.argwhere(keep)                                            # crop-local native idx
    pts = (vox + nlo).astype(np.float64) * (vmm * STEP)               # MESH coords
    lp = cKDTree(p1m).query(pts)[0] <= cKDTree(p4m).query(pts)[0]      # nearest part-mesh (mesh coords)
    p1s = smooth_surf(vox[lp], bone.shape, nlo, vmm, STEP)            # Taubin-smoothed gap-facing surfaces
    p4s = smooth_surf(vox[~lp], bone.shape, nlo, vmm, STEP)
    return dict(pts=pts, p1=pts[lp], p4=pts[~lp], p1s=p1s, p4s=p4s, keep=keep, nlo=nlo,
                vmm=vmm, step=STEP, connected=bool(m1 == m4 and m1 > 0))


def _sig():                                                            # cache-invalidation signature
    return np.array([SMOOTH_ITERS, DECIM_FACES, CLOSE, SEG_VER], dtype=np.int64)


def load_leg_cached(pair, tag, force=False):
    """load_leg with an on-disk cache. The VOX load + morphology + Taubin smoothing is the
    expensive part (~2 min/leg); cache it keyed by the params that affect it so re-runs (peak /
    connection / heatmap tuning) skip straight to the cached surfaces."""
    cf = S / f"leg_cache_{pair}_{tag}.npz"
    if cf.exists() and not force:
        try:
            z = np.load(cf)
            if np.array_equal(z["sig"], _sig()):
                return dict(p1=z["p1"], p4=z["p4"], p1s=z["p1s"], p4s=z["p4s"], keep=z["keep"],
                            nlo=z["nlo"], vmm=float(z["vmm"]), step=int(z["step"]),
                            connected=bool(z["connected"]))
        except Exception:
            pass
    d = load_leg(pair, tag)
    np.savez_compressed(cf, sig=_sig(), p1=d["p1"], p4=d["p4"], p1s=d["p1s"], p4s=d["p4s"],
                        keep=d["keep"], nlo=d["nlo"], vmm=d["vmm"], step=d["step"],
                        connected=d["connected"])
    return d


def _warm(task):                                                      # parallel cache-warming worker
    load_leg_cached(*task); return task


def _free_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // (1024 * 1024)
    except Exception:
        pass
    return 8


NECK_SEP = 0.30    # mm: core-separation above which the join is a long thin bridge (use its neck)


def core_conn(keep, nlo, n1seed, n4seed, vmm, step):
    """Erode the fused bone until the two cores separate, then pick the connection ADAPTIVELY:
      * long thin reaching bridge (cores far apart after little erosion) -> the narrowest
        cross-section between the core centroids (a plain closest-pair floats in the eroded void);
      * thick direct join (cores nearly touching) -> the closest core-pair midpoint.
    Result is snapped onto the nearest bone voxel. MESH coords."""
    s1 = n1seed - nlo; s4 = n4seed - nlo; er = keep
    kv = np.argwhere(keep).astype(np.float64); tk = cKDTree(kv)        # crop-local bone voxels
    for it in range(1, 60):
        er = ndimage.binary_erosion(er, structure=STRUCT)             # incremental (one voxel/step)
        lab_e, _ = ndimage.label(er, structure=STRUCT)
        m1 = comp_seed(lab_e, s1, er.shape); m4 = comp_seed(lab_e, s4, er.shape)
        if m1 > 0 and m4 > 0 and m1 != m4:                            # split into two cores
            c1v = np.argwhere(lab_e == m1).astype(np.float64); c4v = np.argwhere(lab_e == m4).astype(np.float64)
            d, i = cKDTree(c1v).query(c4v); j = int(d.argmin()); coresep = float(d[j]) * (vmm * step)
            if coresep > NECK_SEP:                                     # long thin bridge -> neck
                cc1 = c1v.mean(0); ax = c4v.mean(0) - cc1; ax /= np.linalg.norm(ax)
                proj = (kv - cc1) @ ax; hi = float((c4v.mean(0) - cc1) @ ax)
                band = (proj > 0) & (proj < hi); idx = np.floor(proj[band] / 3.0).astype(int)
                cnt = np.bincount(idx); inte = np.arange(len(cnt))
                good = (inte >= 1) & (inte <= inte.max() - 1) & (cnt > 0)
                mid = kv[band][idx == inte[good][np.argmin(cnt[good])]].mean(0)   # thinnest slab centre
            else:                                                     # thick direct join
                mid = (c1v[i[j]] + c4v[j]) / 2
            mid = kv[tk.query(mid[None])[1][0]]                        # snap onto nearest bone voxel
            return (mid + nlo) * (vmm * step), it, coresep
    return None, -1, -1.0


def env(pts, o, a, u, w, mode):
    d = pts - o; h = d @ a; uu = d @ u; vv = d @ w
    iu = np.floor(uu / CELL).astype(np.int64); iv = np.floor(vv / CELL).astype(np.int64)
    key = np.stack([iu, iv], 1); uk, inv = np.unique(key, axis=0, return_inverse=True)
    ext = np.full(len(uk), -np.inf if mode == "max" else np.inf)
    (np.maximum if mode == "max" else np.minimum).at(ext, inv, h)
    cnt = np.bincount(inv); cu = np.zeros(len(uk)); cw = np.zeros(len(uk))
    np.add.at(cu, inv, uu); np.add.at(cw, inv, vv); cu /= cnt; cw /= cnt
    return {tuple(uk[i]): (ext[i], cu[i], cw[i]) for i in range(len(uk))}


def process(pair):
    SL = load_leg_cached(pair, "sl"); CL = load_leg_cached(pair, "cl")
    vmm, STEP = SL["vmm"], SL["step"]
    c1 = SL["p1"].mean(0); c4 = SL["p4"].mean(0); a = c4 - c1; sep = float(np.linalg.norm(a)); a = a / sep
    tmp = np.array([1., 0, 0]) if abs(a[0]) < 0.9 else np.array([0., 1, 0])
    u = np.cross(a, tmp); u /= np.linalg.norm(u); w = np.cross(a, u); o = c1

    # Constrained CL->SL alignment: force the CL axis (part1->part4) onto the SL axis so it
    # CANNOT flip part1<->part4; sweep only the lateral spin + mirror by chamfer, then refine.
    OCC = Path(cfg.get(pair)["occ"]); rng = np.random.default_rng(0)
    slm = trimesh.load(str(OCC / "sl.stl"), force="mesh", process=False); slm.merge_vertices()
    clm = trimesh.load(str(OCC / "cl.stl"), force="mesh", process=False); clm.merge_vertices()
    SLsurf = slm.vertices[rng.choice(len(slm.vertices), 5000, replace=False)]
    CLsurf = clm.vertices[rng.choice(len(clm.vertices), 5000, replace=False)]
    cc1 = CL["p1"].mean(0); cc4 = CL["p4"].mean(0); csep = float(np.linalg.norm(cc4 - cc1)); ca = (cc4 - cc1) / csep
    ct = np.array([1., 0, 0]) if abs(ca[0]) < 0.9 else np.array([0., 1, 0])
    cu = np.cross(ca, ct); cu /= np.linalg.norm(cu); cw = np.cross(ca, cu); sc = sep / csep

    def coarse(pts, mir, th):
        d = pts - cc1; lh = d @ ca; lu = mir * (d @ cu); lw = d @ cw
        cs, sn = np.cos(th), np.sin(th)
        return c1 + (lh * sc)[:, None] * a + ((cs * lu - sn * lw) * sc)[:, None] * u + ((sn * lu + cs * lw) * sc)[:, None] * w
    tree = cKDTree(SLsurf); best = None
    for mir in (1., -1.):
        for th in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            res = tree.query(coarse(CLsurf, mir, th))[0].mean()
            if best is None or res < best[0]: best = (res, mir, th)
    _, mir, th = best
    SLfull = slm.vertices; tgt_tree = cKDTree(SLfull)                # similarity-ICP target

    def umeyama(X, Y):                                               # s,R,t : s*R*X+t ~ Y
        mx = X.mean(0); my = Y.mean(0); Xc = X - mx; Yc = Y - my
        U, Dg, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
        W = np.diag([1., 1., np.sign(np.linalg.det(U @ Vt))]); R = U @ W @ Vt
        return (np.trace(np.diag(Dg) @ W) / ((Xc ** 2).sum() / len(X))), R, None
    cur = coarse(CLsurf, mir, th); sA, RA, tA = 1., np.eye(3), np.zeros(3)
    for _ in range(15):
        nn = tgt_tree.query(cur)[1]; s, R, _ = umeyama(cur, SLfull[nn])
        t = SLfull[nn].mean(0) - s * R @ cur.mean(0)
        cur = s * (cur @ R.T) + t; sA, RA, tA = s * sA, R @ RA, s * (R @ tA) + t
    def cl2sl(P): p = coarse(P, mir, th); return sA * (p @ RA.T) + tA
    CLp1 = cl2sl(CL["p1s"]); CLp4 = cl2sl(CL["p4s"])                  # smoothed gap-facing surfaces
    info = {"mirror": mir < 0, "res": float(tgt_tree.query(cl2sl(CLsurf))[0].mean())}

    SLtop1 = env(SL["p1s"], o, a, u, w, "max"); CLtop1 = env(CLp1, o, a, u, w, "max")
    SLbot4 = env(SL["p4s"], o, a, u, w, "min"); CLbot4 = env(CLp4, o, a, u, w, "min")

    def heat(SLe, CLe, sign):
        rows = []
        for c in set(SLe) & set(CLe):
            sh, cu, cw = SLe[c]; ch = CLe[c][0]
            pos = o + sh * a + cu * u + cw * w                       # 3D point on the SURGICAL front
            rows.append([c[0], c[1], sign * (sh - ch), pos[0], pos[1], pos[2], cu, cw])
        return np.array(rows) if rows else np.zeros((0, 8))
    h1 = heat(SLtop1, CLtop1, +1.0)                                  # bone1 top  = SL - CL
    h4 = heat(SLbot4, CLbot4, -1.0)                                  # bone4 bottom = CL - SL

    dd, ii = cKDTree(SL["p1"]).query(SL["p4"]); dmin = float(dd.min())  # raw part adjacency
    # Connection = midpoint of the closest CORE1<->CORE4 pair after eroding the bridge (option A).
    n1seed = np.round(SL["p1"] / (vmm * STEP)).astype(int)
    n4seed = np.round(SL["p4"] / (vmm * STEP)).astype(int)
    np.savez_compressed(S / f"conn_cache_{pair}.npz", keep=SL["keep"], nlo=SL["nlo"],
                        n1seed=n1seed, n4seed=n4seed, vmm=vmm, step=STEP, o=o, a=a, sep=sep)
    conn, erit, coresep = core_conn(SL["keep"], SL["nlo"], n1seed, n4seed, vmm, STEP)
    if conn is None:                                                 # fallback: seam centroid
        ct = dd <= dmin + 0.06; conn = ((SL["p4"][ct] + SL["p1"][ii[ct]]) / 2).mean(0)
        erit, coresep = -1, dmin
    connected = bool(SL["connected"] or dmin < 0.10)                 # 0.05 mm in mesh units

    def desc(h):
        return (f"cells={len(h)} val[{h[:,2].min():.2f},{h[:,2].max():.2f}] "
                f"pos={100*(h[:,2]>0).mean():.0f}%") if len(h) else "cells=0 (EMPTY!)"
    print(f"{pair}: mirror={info['mirror']} align-res={info['res']/STEP*1000:.0f}µm connected={connected} "
          f"erode={erit}(core-sep={coresep/STEP*1000:.0f}µm)  h1 {desc(h1)}  h4 {desc(h4)}  "
          f"conn={np.round(conn,2)}", flush=True)
    p1sub = SL["p1"][:: max(1, len(SL["p1"]) // 20000)]
    # voxel-model overlay for the artifact: raw native voxels, CL aligned into the SL frame
    slv = np.vstack([SL["p1"], SL["p4"]]); clv = cl2sl(np.vstack([CL["p1"], CL["p4"]]))
    def _sub(P, n=15000): return P[:: max(1, len(P) // n)].astype(np.float32)
    return dict(h1=h1, h4=h4, conn=conn, connected=connected, min_dist=dmin,
                p1=p1sub, o=o, a=a, u=u, w=w, cell=CELL, vmm=vmm, step=STEP, sep=sep,
                vox_sl=_sub(slv), vox_cl=_sub(clv))


if __name__ == "__main__":
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    tasks = [(p, t) for p in PAIRS for t in ("sl", "cl")]
    NPROC = max(1, min(len(tasks), _free_gb() // 4))                  # ~3 GB peak per leg-load
    warm = sum((S / f"leg_cache_{p}_{t}.npz").exists() for p, t in tasks)
    print(f"warming {len(tasks)-warm}/{len(tasks)} cold leg caches with {NPROC} procs "
          f"(MemAvailable={_free_gb()} GB)...", flush=True)
    if NPROC > 1:
        with ctx.Pool(NPROC) as pool:
            pool.map(_warm, tasks)                                     # parallel VOX load + smoothing
    else:
        for t in tasks:
            _warm(t)
    # legs now cached -> process (align + heatmaps + connection) runs cheap, in parallel over pairs
    if len(PAIRS) > 1:
        with ctx.Pool(min(len(PAIRS), max(1, _free_gb() // 3))) as pool:
            data = dict(zip(PAIRS, pool.map(process, PAIRS)))
    else:
        data = {p: process(p) for p in PAIRS}
    with open(S / "peak_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"\nsaved {S/'peak_data.pkl'}", flush=True)
