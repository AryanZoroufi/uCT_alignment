"""
Clean 7-DOF (rigid + UNIFORM scale) registration of a surgical bone mesh to the
atlas mesh. NO segmentation, no SMC.

  7 DOF = 3 rotation + 3 translation + 1 uniform scale (Umeyama similarity).
  Global init: {mirror 2} x {axis-sign 4} x {spin 16} = 128 starts, each refined
  by similarity-ICP, and the winner is chosen by SYMMETRIC CHAMFER (mm) -- the
  honest overlap metric -- NOT the ICP nearest-point residual (which was
  misleadingly tiny in the earlier run).

Reports the best scale / chamfer / voxel-IoU and overlays atlas(7DOF, magenta)
on the surgical bone (grey) in rerun.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from skimage import measure
sys.path.insert(0, str(Path(__file__).parent))
import growth_config as cfg

PAIR, TAG = "B256M1", "sl"
occ = Path(cfg.get(PAIR)["occ"])
ATLAS = Path(__file__).parent / "atlas.npz"
rng = np.random.default_rng(0)
N_SPIN = 16
N_PTS = 4000


def pca(P):
    c = P.mean(0); Q = P - c
    w, V = np.linalg.eigh(Q.T @ Q / len(Q))
    o = np.argsort(w)[::-1]
    return c, V[:, o], np.sqrt(np.maximum(w[o], 1e-9))


def axis_rot(axis, th):
    a = axis / np.linalg.norm(axis); x, y, z = a
    c, s, C = np.cos(th), np.sin(th), 1 - np.cos(th)
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def umeyama(X, Y):
    """similarity X->Y : returns s, R, t with Y ~= s*R@X + t."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    S = (Yc.T @ Xc) / len(X)
    U, D, Vt = np.linalg.svd(S)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var = (Xc ** 2).sum() / len(X)
    s = float(np.trace(np.diag(D) @ W) / var)
    t = my - s * R @ mx
    return s, R, t


def sim_icp(src, tgt, s, R, t, iters=40, tol=1e-4):
    tree = cKDTree(tgt)
    prev = np.inf
    for _ in range(iters):
        P = s * (src @ R.T) + t
        d, idx = tree.query(P)
        s, R, t = umeyama(src, tgt[idx])
        m = d.mean()
        if abs(prev - m) < tol:
            break
        prev = m
    return s, R, t


def chamfer(A, B):
    return 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())


def solid_iou(m1, m2, n=90):
    lo = np.minimum(m1.bounds[0], m2.bounds[0])
    hi = np.maximum(m1.bounds[1], m2.bounds[1])
    pitch = float((hi - lo).max()) / n
    shp = (np.floor((hi - lo) / pitch).astype(int) + 3)

    def occ_grid(m):
        try:
            pts = m.voxelized(pitch=pitch).fill().points
        except Exception:
            pts = m.sample(300000)
        idx = np.clip(np.floor((pts - lo) / pitch).astype(int), 0, shp - 1)
        g = np.zeros(shp, bool); g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return g
    g1, g2 = occ_grid(m1), occ_grid(m2)
    return float((g1 & g2).sum()) / max(int((g1 | g2).sum()), 1)


# ---------- meshes ----------
scan = trimesh.load(str(occ / f"{TAG}.stl"), force="mesh", process=False)
scan.merge_vertices()
a = np.load(str(ATLAS), allow_pickle=True)
solid, pitch, origin = a["solid"], float(a["pitch"]), a["origin"].astype(np.float64)
av, af, _, _ = measure.marching_cubes(np.pad(solid, 1).astype(np.float32), 0.5)
av = (av - 1) * pitch + origin
atlas_mesh = trimesh.Trimesh(av, af, process=False)
print(f"surgical {PAIR} {TAG}: {len(scan.vertices):,}v  bbox span "
      f"{np.round(scan.extents,1)}", flush=True)
print(f"atlas: {len(av):,}v  bbox span {np.round(atlas_mesh.extents,0)}", flush=True)

tgt = scan.vertices[rng.choice(len(scan.vertices), min(N_PTS, len(scan.vertices)), replace=False)].astype(np.float64)
atl = a["surface_points"].astype(np.float64)
src0 = atl[rng.choice(len(atl), min(N_PTS, len(atl)), replace=False)]

tc, tV, tstd = pca(tgt)
SIGNS = [np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
spins = np.linspace(0, 2 * np.pi, N_SPIN, endpoint=False)

best = None  # (chamfer, s, R, t, tag)
for mir in (False, True):
    src = src0.copy()
    if mir:
        src[:, 0] = 2 * src[:, 0].mean() - src[:, 0]
    sc, sV, sstd = pca(src)
    s0 = float(tstd[0] / sstd[0])
    for si, Sg in enumerate(SIGNS):
        base = tV @ Sg @ sV.T
        for th in spins:
            R0 = axis_rot(tV[:, 0], th) @ base
            t0 = tc - s0 * R0 @ sc
            s, R, t = sim_icp(src, tgt, s0, R0, t0)
            ch = chamfer(s * (src @ R.T) + t, tgt)
            if best is None or ch < best[0]:
                best = (ch, s, R, t, f"mir={int(mir)} sign={si} spin={np.degrees(th):.0f}")

ch, s, R, t, tag = best
atlas_7dof = trimesh.Trimesh((s * (atlas_mesh.vertices @ R.T) + t), af, process=False)
iou = solid_iou(atlas_7dof, scan)
print(f"\n===== 7-DOF SIMILARITY RESULT ({tag}) =====", flush=True)
print(f"  uniform scale s = {s:.4f}", flush=True)
print(f"  symmetric chamfer = {ch:.3f} mm   (ICP earlier 0.600, SMC 0.514)", flush=True)
print(f"  solid IoU (full atlas vs bone) = {iou:.4f}   (SMC IoU was 0.274)", flush=True)
print(f"  atlas span after 7-DOF: {np.round(atlas_7dof.extents,1)}  vs bone "
      f"{np.round(scan.extents,1)}", flush=True)

import rerun as rr
rr.init("align_7dof", spawn=True)
print("\ninit ok", flush=True)
rr.log("surgical_bone", rr.Mesh3D(
    vertex_positions=scan.vertices.astype(np.float32), triangle_indices=scan.faces,
    vertex_colors=np.tile([150, 150, 150], (len(scan.vertices), 1))))
rr.log("atlas_7DOF", rr.Mesh3D(
    vertex_positions=atlas_7dof.vertices.astype(np.float32), triangle_indices=af,
    vertex_colors=np.tile([235, 40, 220], (len(atlas_7dof.vertices), 1))))
print("rerun: grey=surgical bone, MAGENTA=atlas after 7-DOF similarity.", flush=True)
