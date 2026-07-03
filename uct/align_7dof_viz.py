"""
7-DOF alignment ONLY (no segmentation) of the atlas to each of the 4 bones, to
check the atlas stays PROPORTIONAL when the extra DOF are removed.

  ICP:  chamfer-selected 7-DOF similarity (atlas_register, uniform scale, mirror)
  SMC:  chamfer-scored importance-sampling refinement, ALSO 7-DOF (uniform scale)
NO anisotropic scale, NO shear -> the atlas cannot be fattened/sheared.

Visualize: grey = high-res scan bone; colours = aligned atlas by label (1=red &
4=blue tibia). Grid TOP=B256M1, BOTTOM=B256M7; LEFT=CL, RIGHT=SL.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from skimage import measure
import rerun as rr
sys.path.insert(0, str(Path(__file__).parent))
from atlas_register import register_atlas_7dof_chamfer

RECON = Path(__file__).parent / "../bones_to_recon"
ATLAS = Path(__file__).parent / "atlas.npz"
SPACING = 22.0
LCOL = {1: [230, 25, 75], 2: [60, 180, 75], 3: [255, 215, 20], 4: [0, 130, 200],
        5: [245, 130, 48], 6: [145, 30, 180], 7: [70, 240, 240], 8: [240, 50, 230],
        9: [170, 110, 40]}
BONES = [("B256M1", "cl", 0, 0), ("B256M1", "sl", 1, 0),
         ("B256M7", "cl", 0, 1), ("B256M7", "sl", 1, 1)]


def apply4(T, V):
    return (V @ T[:3, :3].T) + T[:3, 3]


def euler_R(r):
    rx, ry, rz = r
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def submesh(V, F, vl, L):
    keep = (vl[F] == L).sum(1) >= 2
    ff = F[keep]
    if not len(ff):
        return None, None
    used = np.unique(ff)
    remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
    return V[used].astype(np.float32), remap[ff]


def refine_7dof(atl_surf, scan_surf, ncand=1000, iters=2, seed=0):
    """chamfer-scored IS refinement, 7-DOF (uniform scale). Returns delta params."""
    rng = np.random.default_rng(seed)
    c = atl_surf.mean(0)
    stree = cKDTree(scan_surf)

    def apply_delta(P, p):
        M = euler_R(p[3:6]) * np.exp(p[6])            # uniform scale = exp(p6)
        return (P - c) @ M.T + c + p[0:3]

    def score(p):
        T = apply_delta(atl_surf, p)
        return 0.5 * (stree.query(T)[0].mean() + cKDTree(T).query(scan_surf)[0].mean())

    best_p = np.zeros(7); best_s = score(best_p)
    RANGES = {"t": np.array([2., 2, 2]), "r": np.array([.4, .4, .4]),
              "s": np.array([.4]), "j": np.array([1., 1, 1, .17, .17, .17, .1])}
    for _ in range(iters):
        for st, idxs in (("t", [0, 1, 2]), ("r", [3, 4, 5]), ("s", [6]), ("j", list(range(7)))):
            P = np.tile(best_p, (ncand, 1))
            P[:, idxs] = best_p[idxs] + rng.uniform(-1, 1, (ncand, len(idxs))) * RANGES[st]
            for p in P:
                sc = score(p)
                if sc < best_s:
                    best_s, best_p = sc, p.copy()
    return best_p, c, best_s


# ---- atlas mesh + per-vertex labels (original atlas frame) ----
a = np.load(str(ATLAS), allow_pickle=True)
solid = a["solid"]; pitch = float(a["pitch"]); origin = a["origin"].astype(np.float64)
av0, af0, _, _ = measure.marching_cubes(np.pad(solid, 1).astype(np.float32), 0.5)
av0 = (av0 - 1) * pitch + origin
ai = np.clip(np.round((av0 - origin) / pitch).astype(int), 0, np.array(solid.shape) - 1)
vlab0 = a["bone_labels"][ai[:, 0], ai[:, 1], ai[:, 2]].astype(int)
A_surf = a["surface_points"].astype(np.float64)
rng = np.random.default_rng(0)

rr.init("align_7dof_viz", spawn=True)
print("init ok", flush=True)
for pair, tag, col, row in BONES:
    occ = RECON / f"{pair}_occ"
    bone = trimesh.load(str(occ / f"{tag}.stl"), force="mesh", process=False)
    N_surf = bone.vertices[rng.choice(len(bone.vertices),
             min(4000, len(bone.vertices)), replace=False)].astype(np.float64)

    T, info = register_atlas_7dof_chamfer(A_surf, N_surf)      # ICP 7-DOF
    av7 = apply4(T, av0)                                       # atlas mesh -> scan
    atl_surf7 = apply4(T, A_surf[rng.choice(len(A_surf), 3000, replace=False)])
    dp, c7, ch = refine_7dof(atl_surf7, N_surf[rng.choice(len(N_surf), 3000, replace=False)])
    M = euler_R(dp[3:6]) * np.exp(dp[6])
    avf = (av7 - c7) @ M.T + c7 + dp[0:3]                      # + 7-DOF SMC refine

    off = (np.array([col * SPACING, row * SPACING, 0.0]) - bone.vertices.mean(0)).astype(np.float32)
    base = f"{pair}/{tag.upper()}"
    rr.log(f"{base}/bone_GREY", rr.Mesh3D(
        vertex_positions=(bone.vertices + off).astype(np.float32),
        triangle_indices=bone.faces,
        vertex_colors=np.tile([150, 150, 150], (len(bone.vertices), 1))))
    for L in sorted(np.unique(vlab0)):
        if L == 0:
            continue
        sv, sf = submesh(avf, af0, vlab0, L)
        if sv is None:
            continue
        name = f"atlas{L}_TIBIA" if L in (1, 4) else f"atlas{L}"
        rr.log(f"{base}/{name}", rr.Mesh3D(vertex_positions=sv + off, triangle_indices=sf,
               vertex_colors=np.tile(LCOL.get(L, [200, 200, 200]), (len(sv), 1))))
    print(f"{base}: ICP scale={info['scale']:.4f} mirror={info['mirror']}  "
          f"refined uniform-scale x{np.exp(dp[6]):.3f}  chamfer={ch:.3f}mm", flush=True)

print("\ndone. grey=bone, colours=7-DOF-aligned atlas by label (uniform scale, no "
      "shear). Compare proportions vs the distorted full-registration atlas.", flush=True)
