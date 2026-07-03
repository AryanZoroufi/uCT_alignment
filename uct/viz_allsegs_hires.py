"""
rerun: FULL segmentation at FULL RESOLUTION.

The coarse atlas per-bone meshes are only used as a label source: every vertex of
the HIGH-RES bone surface is assigned to its nearest atlas segment, then the
detailed bone is split into per-segment submeshes. So each segment is shown at the
same detail as the full bone (~200k faces) -- just divided by label.

Tibia = bones 1 (red) & 4 (blue). Grid: TOP=B256M1, BOTTOM=B256M7; LEFT=CL, RIGHT=SL.
Toggle each segment in the tree ({pair}/{CL,SL}/seg{L}).
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import rerun as rr

RECON = Path(__file__).parent / "../bones_to_recon"
SPACING = 22.0
LCOL = {1: [230, 25, 75], 2: [60, 180, 75], 3: [255, 215, 20], 4: [0, 130, 200],
        5: [245, 130, 48], 6: [145, 30, 180], 7: [70, 240, 240], 8: [240, 50, 230],
        9: [170, 110, 40]}
BONES = [("B256M1", "cl", 0, 0), ("B256M1", "sl", 1, 0),
         ("B256M7", "cl", 0, 1), ("B256M7", "sl", 1, 1)]


def submesh(V, F, vlab, L):
    keep = (vlab[F] == L).sum(1) >= 2          # faces whose majority label == L
    ff = F[keep]
    if not len(ff):
        return None, None
    used = np.unique(ff)
    remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
    return V[used].astype(np.float32), remap[ff]


rr.init("allsegs_hires", spawn=True)
print("init ok", flush=True)
for pair, tag, col, row in BONES:
    occ = RECON / f"{pair}_occ"
    bone = trimesh.load(str(occ / f"{tag}.stl"), force="mesh", process=False)
    V, F = bone.vertices, bone.faces

    # label source: coarse atlas per-bone meshes
    segfiles = sorted((occ / "allsegs").glob(f"{tag}_bone_*.stl"))
    seg_pts, seg_lab = [], []
    for f in segfiles:
        L = int(f.stem.split("_")[-1])
        m = trimesh.load(str(f), force="mesh", process=False)
        seg_pts.append(m.vertices); seg_lab.append(np.full(len(m.vertices), L, int))
    seg_pts = np.vstack(seg_pts); seg_lab = np.concatenate(seg_lab)

    # assign each HIGH-RES bone vertex to nearest atlas segment
    _, idx = cKDTree(seg_pts).query(V)
    vlab = seg_lab[idx]

    off = (np.array([col * SPACING, row * SPACING, 0.0]) - V.mean(0)).astype(np.float32)
    present = []
    for L in sorted(np.unique(vlab)):
        sv, sf = submesh(V, F, vlab, L)
        if sv is None:
            continue
        name = f"seg{L}_TIBIA" if L in (1, 4) else f"seg{L}"
        rr.log(f"{pair}/{tag.upper()}/{name}", rr.Mesh3D(
            vertex_positions=sv + off, triangle_indices=sf,
            vertex_colors=np.tile(LCOL.get(L, [150, 150, 150]), (len(sv), 1))))
        present.append((int(L), int((vlab == L).sum())))
    print(f"{pair} {tag}: bone {len(F):,}f  segments(vtx) {present}", flush=True)

print("\ndone. FULL-RES bone split by segment; TIBIA = seg 1 (red) & 4 (blue).", flush=True)
print("TOP=B256M1, BOTTOM=B256M7; LEFT=CL, RIGHT=SL. Toggle segments in the tree.",
      flush=True)
