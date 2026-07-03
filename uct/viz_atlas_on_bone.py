"""
rerun: the REGISTERED ATLAS overlaid on each high-res bone, to see visually where
the atlas lands (and hence why the segmentation labels fall where they do).

  grey  = high-res scan bone (reality)
  color = the atlas template transformed into the scan frame, split by atlas
          label (1..9); tibia = 1 (red) & 4 (blue).

Uses {occ}/{tag}_atlasdbg.npz (atlas-in-scan-frame + per-vertex atlas label) from
seg_debug.py -- same default registration as the segmentation, so this IS the
alignment the labels came from. Toggle bone vs atlas / individual labels in the tree.
Grid: TOP=B256M1, BOTTOM=B256M7; LEFT=CL, RIGHT=SL.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
import rerun as rr

RECON = Path(__file__).parent / "../bones_to_recon"
SPACING = 22.0
LCOL = {1: [230, 25, 75], 2: [60, 180, 75], 3: [255, 215, 20], 4: [0, 130, 200],
        5: [245, 130, 48], 6: [145, 30, 180], 7: [70, 240, 240], 8: [240, 50, 230],
        9: [170, 110, 40]}
BONES = [("B256M1", "cl", 0, 0), ("B256M1", "sl", 1, 0),
         ("B256M7", "cl", 0, 1), ("B256M7", "sl", 1, 1)]


def submesh(V, F, vl, L):
    keep = (vl[F] == L).sum(1) >= 2
    ff = F[keep]
    if not len(ff):
        return None, None
    used = np.unique(ff)
    remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
    return V[used].astype(np.float32), remap[ff]


rr.init("atlas_on_bone", spawn=True)
print("init ok", flush=True)
for pair, tag, col, row in BONES:
    occ = RECON / f"{pair}_occ"
    bone = trimesh.load(str(occ / f"{tag}.stl"), force="mesh", process=False)
    dbgp = occ / f"{tag}_atlasdbg.npz"
    if not dbgp.exists():
        print(f"{pair} {tag}: MISSING {dbgp.name}", flush=True)
        continue
    d = np.load(str(dbgp))
    av, af, vl = d["atlas_verts"], d["atlas_faces"], d["atlas_vlabels"].astype(int)

    off = (np.array([col * SPACING, row * SPACING, 0.0]) - bone.vertices.mean(0)).astype(np.float32)
    base = f"{pair}/{tag.upper()}"
    rr.log(f"{base}/bone_GREY", rr.Mesh3D(
        vertex_positions=(bone.vertices + off).astype(np.float32),
        triangle_indices=bone.faces,
        vertex_colors=np.tile([150, 150, 150], (len(bone.vertices), 1))))
    labs = []
    for L in sorted(np.unique(vl)):
        if L == 0:
            continue
        sv, sf = submesh(av, af, vl, L)
        if sv is None:
            continue
        name = f"atlas{L}_TIBIA" if L in (1, 4) else f"atlas{L}"
        rr.log(f"{base}/{name}", rr.Mesh3D(
            vertex_positions=sv + off, triangle_indices=sf,
            vertex_colors=np.tile(LCOL.get(L, [200, 200, 200]), (len(sv), 1))))
        labs.append(int(L))
    print(f"{pair} {tag}: atlas labels overlaid {labs}", flush=True)

print("\ndone. grey=scan bone, colours=registered atlas labels (1=red tibia, "
      "4=blue tibia). Toggle bone vs atlas in the tree to compare.", flush=True)
