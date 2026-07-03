"""
Chamfer-driven SMC-style refinement, SEEDED from the 7-DOF similarity fit.

Fixes the two failure modes we found in the pipeline SMC:
  (1) SEED: start from the chamfer-selected 7-DOF pose (align_7dof), so the atlas
      is already correctly scaled/posed -- the sampler only has to refine, and
      the widened scale window is now centred on the right value.
  (2) LIKELIHOOD: score particles by symmetric CHAMFER (KD-tree, mm) instead of
      the 32^3 IoU (which is capped ~0.35 by the 9-bone foot atlas and gives the
      sampler almost no gradient).

Refinement = factorised importance-sampling stages (translation / rotation /
anisotropic-scale / joint), best-of-N per stage, iterated. Anisotropic scale is
ALLOWED (widened +-40%) so it can also correct the residual z-flatness.

Reports chamfer + IoU for: 7-DOF seed  vs  7-DOF + chamfer-SMC.
Overlay: grey=bone, MAGENTA=7-DOF seed, CYAN=after chamfer-SMC.
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
N_SPIN, N_PTS = 16, 4000
N_CAND, N_SCORE = 1500, 3500         # IS candidates / scoring points per stage
N_ITER = 2                           # passes over the 4 stages


# ---------- 7-DOF helpers (same as align_7dof) ----------
def pca(P):
    c = P.mean(0); Q = P - c
    w, V = np.linalg.eigh(Q.T @ Q / len(Q))
    o = np.argsort(w)[::-1]
    return c, V[:, o], np.sqrt(np.maximum(w[o], 1e-9))


def axis_rot(axis, th):
    a = axis / np.linalg.norm(axis); x, y, z = a
    c, s, C = np.cos(th), np.sin(th), 1 - np.cos(th)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X-mx, Y-my
    U, D, Vt = np.linalg.svd((Yc.T @ Xc) / len(X))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = float(np.trace(np.diag(D) @ W) / ((Xc**2).sum() / len(X)))
    return s, R, my - s * R @ mx


def sim_icp(src, tgt, s, R, t, iters=40, tol=1e-4):
    tree = cKDTree(tgt); prev = np.inf
    for _ in range(iters):
        d, idx = tree.query(s * (src @ R.T) + t)
        s, R, t = umeyama(src, tgt[idx])
        if abs(prev - d.mean()) < tol:
            break
        prev = d.mean()
    return s, R, t


def euler_R(r):
    rx, ry, rz = r
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def sym_chamfer(A, B):
    return 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())


def solid_iou(m1, m2, n=90):
    lo = np.minimum(m1.bounds[0], m2.bounds[0]); hi = np.maximum(m1.bounds[1], m2.bounds[1])
    pitch = float((hi - lo).max()) / n; shp = np.floor((hi - lo) / pitch).astype(int) + 3

    def g(m):
        try:
            pts = m.voxelized(pitch=pitch).fill().points
        except Exception:
            pts = m.sample(300000)
        idx = np.clip(np.floor((pts - lo) / pitch).astype(int), 0, shp - 1)
        gg = np.zeros(shp, bool); gg[idx[:, 0], idx[:, 1], idx[:, 2]] = True; return gg
    a, b = g(m1), g(m2)
    return float((a & b).sum()) / max(int((a | b).sum()), 1)


# ---------- load ----------
scan = trimesh.load(str(occ / f"{TAG}.stl"), force="mesh", process=False); scan.merge_vertices()
a = np.load(str(ATLAS), allow_pickle=True)
solid, pitch, origin = a["solid"], float(a["pitch"]), a["origin"].astype(np.float64)
av, af, _, _ = measure.marching_cubes(np.pad(solid, 1).astype(np.float32), 0.5)
av = (av - 1) * pitch + origin
atlas_mesh = trimesh.Trimesh(av, af, process=False)
tgt = scan.vertices[rng.choice(len(scan.vertices), min(N_PTS, len(scan.vertices)), replace=False)].astype(np.float64)
atl = a["surface_points"].astype(np.float64)
src0 = atl[rng.choice(len(atl), min(N_PTS, len(atl)), replace=False)]

# ---------- STAGE 0: 7-DOF seed (chamfer-selected sweep) ----------
tc, tV, tstd = pca(tgt)
SIGNS = [np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
spins = np.linspace(0, 2 * np.pi, N_SPIN, endpoint=False)
m0 = float(src0[:, 0].mean())               # mirror plane (atlas x-mean)


def reflect(P):
    Q = P.copy(); Q[:, 0] = 2 * m0 - Q[:, 0]; return Q


best7 = None                                 # (ch, s, R, t, mir)
for mir in (False, True):
    src = reflect(src0) if mir else src0.copy()
    sc, sV, sstd = pca(src); s0 = float(tstd[0] / sstd[0])
    for Sg in SIGNS:
        base = tV @ Sg @ sV.T
        for th in spins:
            R0 = axis_rot(tV[:, 0], th) @ base
            s, R, t = sim_icp(src, tgt, s0, R0, tc - s0 * R0 @ sc)
            ch = sym_chamfer(s * (src @ R.T) + t, tgt)
            if best7 is None or ch < best7[0]:
                best7 = (ch, s, R, t, mir)
ch7, s7, R7, t7, mir7 = best7


def atlas_in_scan(P):                        # mirror (if it won) THEN 7-DOF
    return s7 * ((reflect(P) if mir7 else P) @ R7.T) + t7


atlas7_v = atlas_in_scan(atlas_mesh.vertices)     # full mesh, 7-DOF (mirror baked)
atlas7_surf = atlas_in_scan(src0)
mesh7 = trimesh.Trimesh(atlas7_v, af, process=False)
print(f"[SEED 7-DOF] scale={s7:.4f}  mirror={mir7}  chamfer={ch7:.3f}mm  "
      f"IoU={solid_iou(mesh7, scan):.4f}", flush=True)

# ---------- STAGE 1-4: chamfer-scored IS refinement (delta about atlas7 centroid) ----------
c7 = atlas7_surf.mean(0)
# DENSE, FIXED, SYMMETRIC chamfer. Both earlier attempts were gameable:
#   - sparse symmetric (1200 pts) -> density-biased noise -> mild wrong shrink.
#   - dense ONE-directional (atlas->scan) -> collapses the atlas onto a surface
#     patch (leaves the bone uncovered) since it doesn't penalise coverage.
# Symmetric penalises BOTH "atlas off the bone" (atlas->scan) AND "bone not
# covered by atlas" (scan->atlas), so it cannot be gamed by shrinking; enough
# points (3500) keep it from being noisy.
scan_sc = scan.vertices.astype(np.float64)
scan_sc = scan_sc[rng.choice(len(scan_sc), min(N_SCORE, len(scan_sc)), replace=False)]
atl_sc = atlas_in_scan(atl[rng.choice(len(atl), N_SCORE, replace=False)])
scan_tree = cKDTree(scan_sc)


def apply_delta(P, p):
    M = euler_R(p[3:6]) @ np.diag(p[6:9])
    return (P - c7) @ M.T + c7 + p[0:3]


def score(p):
    T = apply_delta(atl_sc, p)
    return 0.5 * (scan_tree.query(T)[0].mean() + cKDTree(T).query(scan_sc)[0].mean())


# stage sampling half-widths (WIDENED, anisotropic scale allowed)
RANGES = {
    "trans": np.array([3.0, 3.0, 3.0]),                    # mm
    "rot":   np.array([0.70, 0.70, 0.70]),                 # rad (~40 deg)
    "scale": np.array([0.40, 0.40, 0.40]),                 # +-40% anisotropic
    "joint": np.array([1.0, 1.0, 1.0, 0.17, 0.17, 0.17, 0.10, 0.10, 0.10]),
}
best_p = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1.0])
best_s = score(best_p)


def sample_stage(which):
    global best_p, best_s
    P = np.tile(best_p, (N_CAND, 1))
    if which == "trans":
        P[:, 0:3] = best_p[0:3] + rng.uniform(-1, 1, (N_CAND, 3)) * RANGES["trans"]
    elif which == "rot":
        P[:, 3:6] = best_p[3:6] + rng.uniform(-1, 1, (N_CAND, 3)) * RANGES["rot"]
    elif which == "scale":
        P[:, 6:9] = best_p[6:9] + rng.uniform(-1, 1, (N_CAND, 3)) * RANGES["scale"]
    else:
        P = best_p + rng.uniform(-1, 1, (N_CAND, 9)) * RANGES["joint"]
    for p in P:
        sc = score(p)
        if sc < best_s:
            best_s, best_p = sc, p.copy()


for it in range(N_ITER):
    for st in ("trans", "rot", "scale", "joint"):
        sample_stage(st)
    print(f"  [iter {it+1}] chamfer={best_s:.3f}mm  "
          f"scale=({best_p[6]:.3f},{best_p[7]:.3f},{best_p[8]:.3f})  "
          f"rot=({np.degrees(best_p[3]):.0f},{np.degrees(best_p[4]):.0f},"
          f"{np.degrees(best_p[5]):.0f})deg", flush=True)

atlas_final_v = apply_delta(atlas7_v, best_p)
mesh_final = trimesh.Trimesh(atlas_final_v, af, process=False)
ch_final = sym_chamfer(apply_delta(atlas7_surf, best_p), tgt)
iou_final = solid_iou(mesh_final, scan)

print(f"\n===== RESULT =====", flush=True)
print(f"7-DOF seed         : chamfer={ch7:.3f}mm  IoU={solid_iou(mesh7, scan):.4f}", flush=True)
print(f"+ chamfer-SMC      : chamfer={ch_final:.3f}mm  IoU={iou_final:.4f}", flush=True)
print(f"  anisotropic scale found = ({best_p[6]:.3f},{best_p[7]:.3f},{best_p[8]:.3f})  "
      f"(z-stretch fixes the flat-atlas mismatch if >1)", flush=True)
print(f"  atlas span after : {np.round(mesh_final.extents,1)}  vs bone {np.round(scan.extents,1)}",
      flush=True)

import rerun as rr
rr.init("smc_chamfer", spawn=True); print("\ninit ok", flush=True)
rr.log("surgical_bone", rr.Mesh3D(vertex_positions=scan.vertices.astype(np.float32),
       triangle_indices=scan.faces, vertex_colors=np.tile([150, 150, 150], (len(scan.vertices), 1))))
rr.log("atlas_7DOF_seed", rr.Mesh3D(vertex_positions=atlas7_v.astype(np.float32),
       triangle_indices=af, vertex_colors=np.tile([235, 40, 220], (len(atlas7_v), 1))))
rr.log("atlas_chamferSMC", rr.Mesh3D(vertex_positions=atlas_final_v.astype(np.float32),
       triangle_indices=af, vertex_colors=np.tile([40, 220, 230], (len(atlas_final_v), 1))))
print("rerun: grey=bone, MAGENTA=7-DOF seed, CYAN=after chamfer-SMC.", flush=True)
