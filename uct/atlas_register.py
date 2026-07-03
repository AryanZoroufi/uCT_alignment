"""
Chamfer-selected, mirror-correct 7-DOF (rigid + uniform scale) registration of
the atlas surface onto a scan surface. Returns a 4x4 atlas->scan affine (with the
mirror baked into the matrix), to be used as the initial transform for the
downstream SimpleITK affine refinement in _segment_via_atlas.

Replaces the pipeline's PCA + ICP-residual selection + IoU-SMC, which:
  - selected the pose by ICP nearest-point residual (misleadingly small) and so
    landed in the wrong mirror/scale basin, and
  - could not recover scale (IoU-SMC scale window was +-20% around a bad init).
This version sweeps {mirror x sign x spin} similarity-ICP starts and picks the
winner by SYMMETRIC CHAMFER. On B256M1 SL it lifts atlas->bone IoU 0.27 -> 0.43.
"""
import numpy as np
from scipy.spatial import cKDTree


def _pca(P):
    c = P.mean(0); Q = P - c
    w, V = np.linalg.eigh(Q.T @ Q / len(Q))
    o = np.argsort(w)[::-1]
    return c, V[:, o], np.sqrt(np.maximum(w[o], 1e-9))


def _axis_rot(axis, th):
    a = axis / np.linalg.norm(axis); x, y, z = a
    c, s, C = np.cos(th), np.sin(th), 1 - np.cos(th)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def _umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X-mx, Y-my
    U, D, Vt = np.linalg.svd((Yc.T @ Xc) / len(X))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = float(np.trace(np.diag(D) @ W) / ((Xc**2).sum() / len(X)))
    return s, R, my - s * R @ mx


def _sim_icp(src, tgt_tree, tgt, s, R, t, iters=40, tol=1e-4):
    prev = np.inf
    for _ in range(iters):
        d, idx = tgt_tree.query(s * (src @ R.T) + t)
        s, R, t = _umeyama(src, tgt[idx])
        if abs(prev - d.mean()) < tol:
            break
        prev = d.mean()
    return s, R, t


def _sym_chamfer(A, B, Btree):
    return 0.5 * (Btree.query(A)[0].mean() + cKDTree(A).query(B)[0].mean())


def register_atlas_7dof_chamfer(A_surf, N_surf, seed=0, n_spin=16, n_pts=4000):
    """A_surf (atlas surface pts), N_surf (scan surface pts) -> (A_total 4x4, info)."""
    rng = np.random.default_rng(seed)
    src0 = A_surf[rng.choice(len(A_surf), min(n_pts, len(A_surf)), replace=False)].astype(np.float64)
    tgt = N_surf[rng.choice(len(N_surf), min(n_pts, len(N_surf)), replace=False)].astype(np.float64)
    tc, tV, tstd = _pca(tgt)
    tgt_tree = cKDTree(tgt)
    m0 = float(src0[:, 0].mean())

    def reflect(P):
        Q = P.copy(); Q[:, 0] = 2 * m0 - Q[:, 0]; return Q

    SIGNS = [np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
    spins = np.linspace(0, 2 * np.pi, n_spin, endpoint=False)
    best = None                                   # (ch, s, R, t, mir)
    for mir in (False, True):
        src = reflect(src0) if mir else src0.copy()
        sc, sV, sstd = _pca(src); s0 = float(tstd[0] / sstd[0])
        for Sg in SIGNS:
            base = tV @ Sg @ sV.T
            for th in spins:
                R0 = _axis_rot(tV[:, 0], th) @ base
                s, R, t = _sim_icp(src, tgt_tree, tgt, s0, R0, tc - s0 * R0 @ sc)
                ch = _sym_chamfer(s * (src @ R.T) + t, tgt, tgt_tree)
                if best is None or ch < best[0]:
                    best = (ch, s, R, t, mir)
    ch, s7, R7, t7, mir7 = best

    # Bake mirror (reflection about x=m0) into the 4x4:
    #   x -> s7 R7 (D x + tr) + t7,  D=diag(-1,1,1), tr=[2 m0,0,0] if mirror.
    D = np.diag([-1.0, 1.0, 1.0]) if mir7 else np.eye(3)
    tr = np.array([2 * m0, 0.0, 0.0]) if mir7 else np.zeros(3)
    A = np.eye(4)
    A[:3, :3] = s7 * R7 @ D
    A[:3, 3] = s7 * R7 @ tr + t7
    return A, dict(chamfer=float(ch), scale=float(s7), mirror=bool(mir7))
