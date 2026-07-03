"""
Export the 7-DOF-aligned atlas-on-bone scene (4 bones) as a compact JSON of
decimated meshes (base64 typed arrays) for a self-contained WebGL mobile viewer.
Same 7-DOF alignment as align_7dof_viz.py (ICP + uniform-scale chamfer refine).
"""
import sys
import json
import base64
from pathlib import Path
import numpy as np
import trimesh
import fast_simplification
from scipy.spatial import cKDTree
from skimage import measure
sys.path.insert(0, str(Path(__file__).parent))
from atlas_register import register_atlas_7dof_chamfer

RECON = Path(__file__).parent / "../bones_to_recon"
ATLAS = Path(__file__).parent / "atlas.npz"
OUT = Path("/tmp/claude-1000/-home-aryan-Projects-uct-backup-uCT-alignment/"
           "2243f74f-59e2-4926-a697-e6fdd62c17ea/scratchpad/align7dof_scene.json")
SPACING = 22.0
BONE_FACES, ATLAS_FACES = 26000, 5000     # bone = detailed surface (uint16-safe: <65k verts)
LCOL = {1: [230, 25, 75], 2: [60, 180, 75], 3: [255, 215, 20], 4: [0, 130, 200],
        5: [245, 130, 48], 6: [145, 30, 180], 7: [70, 240, 240], 8: [240, 50, 230],
        9: [170, 110, 40]}
BONES = [("B256M1", "cl", 0, 0), ("B256M1", "sl", 1, 0),
         ("B256M7", "cl", 0, 1), ("B256M7", "sl", 1, 1)]


def euler_R(r):
    rx, ry, rz = r
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def refine_7dof(atl_surf, scan_surf, ncand=1000, iters=2, seed=0):
    rng = np.random.default_rng(seed); c = atl_surf.mean(0); stree = cKDTree(scan_surf)

    def app(P, p):
        M = euler_R(p[3:6]) * np.exp(p[6]); return (P - c) @ M.T + c + p[0:3]

    def score(p):
        T = app(atl_surf, p)
        return 0.5 * (stree.query(T)[0].mean() + cKDTree(T).query(scan_surf)[0].mean())
    best_p = np.zeros(7); best_s = score(best_p)
    RG = {"t": [2., 2, 2], "r": [.4, .4, .4], "s": [.4], "j": [1., 1, 1, .17, .17, .17, .1]}
    for _ in range(iters):
        for st, idxs in (("t", [0, 1, 2]), ("r", [3, 4, 5]), ("s", [6]), ("j", list(range(7)))):
            P = np.tile(best_p, (ncand, 1))
            P[:, idxs] = best_p[idxs] + rng.uniform(-1, 1, (ncand, len(idxs))) * np.array(RG[st])
            for p in P:
                s = score(p)
                if s < best_s:
                    best_s, best_p = s, p.copy()
    return best_p, c


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def submesh(V, F, vl, L):
    """faces whose majority vertex-label == L -> compact submesh."""
    keep = (vl[F] == L).sum(1) >= 2
    ff = F[keep]
    if not len(ff):
        return None, None
    used = np.unique(ff)
    remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
    return V[used], remap[ff]


def pack(mesh, colors_u8, name, group, label=0):
    m = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    assert len(m.vertices) < 65536, len(m.vertices)   # uint16 indices
    return dict(name=name, group=group, label=int(label),
                pos=b64(m.vertices.astype(np.float32)),
                col=b64(colors_u8.astype(np.uint8)),
                idx=b64(m.faces.astype(np.uint16)),
                nverts=len(m.vertices), nfaces=len(m.faces))


def decimate(V, F, target):
    if len(F) <= target:
        return V.astype(np.float64), F.astype(np.int32)
    return fast_simplification.simplify(V.astype(np.float64), F.astype(np.int32),
                                        target_reduction=float(1 - target / len(F)))


# atlas mesh + labels
a = np.load(str(ATLAS), allow_pickle=True)
solid = a["solid"]; pitch = float(a["pitch"]); origin = a["origin"].astype(np.float64)
av0, af0, _, _ = measure.marching_cubes(np.pad(solid, 1).astype(np.float32), 0.5)
av0 = (av0 - 1) * pitch + origin
ai = np.clip(np.round((av0 - origin) / pitch).astype(int), 0, np.array(solid.shape) - 1)
vlab0 = a["bone_labels"][ai[:, 0], ai[:, 1], ai[:, 2]].astype(int)
A_surf = a["surface_points"].astype(np.float64)
rng = np.random.default_rng(0)

meshes = []
for gi, (pair, tag, col, row) in enumerate(BONES):
    occ = RECON / f"{pair}_occ"
    bone = trimesh.load(str(occ / f"{tag}.stl"), force="mesh", process=False)
    bone.merge_vertices()      # index the surface so decimation shares vertices (uint16-safe)
    N_surf = bone.vertices[rng.choice(len(bone.vertices), min(4000, len(bone.vertices)),
             replace=False)].astype(np.float64)
    T, info = register_atlas_7dof_chamfer(A_surf, N_surf)
    av7 = (av0 @ T[:3, :3].T) + T[:3, 3]
    ssurf = (A_surf[rng.choice(len(A_surf), 3000, replace=False)] @ T[:3, :3].T) + T[:3, 3]
    dp, c7 = refine_7dof(ssurf, N_surf[rng.choice(len(N_surf), 3000, replace=False)])
    M = euler_R(dp[3:6]) * np.exp(dp[6])
    avf = (av7 - c7) @ M.T + c7 + dp[0:3]

    off = np.array([col * SPACING, row * SPACING, 0.0]) - bone.vertices.mean(0)
    spec = f"{pair} {tag.upper()}"

    # detailed bone (decimated); transfer the 7-DOF-aligned atlas labels to each
    # bone vertex (nearest aligned-atlas vertex), then SPLIT the bone surface into
    # one separate mesh per label -- the bone's shape/detail is unchanged.
    bv, bf = decimate(bone.vertices, bone.faces, BONE_FACES)
    _m = vlab0 > 0                                  # only LABELLED atlas verts
    _, nn = cKDTree(avf[_m]).query(bv)
    vlab = vlab0[_m][nn].astype(int)
    bv = bv + off
    pieces = []
    for L in sorted(set(int(x) for x in np.unique(vlab))):
        sv, sf = submesh(bv, bf, vlab, L)
        if sv is None:
            continue
        meshes.append(pack(trimesh.Trimesh(sv, sf, process=False),
                           np.tile(LCOL.get(L, [200, 200, 200]), (len(sv), 1)),
                           f"{spec} · part {L}", gi, label=L))
        pieces.append((L, len(sf)))
    print(f"{spec}: {len(bf):,}f bone -> pieces {pieces}  scale={info['scale']:.4f}",
          flush=True)

scene = dict(spacing=SPACING, groups=[f"{p} {t.upper()}" for p, t, _, _ in BONES],
             meshes=meshes)
OUT.write_text(json.dumps(scene))
print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB, {len(meshes)} meshes)", flush=True)
