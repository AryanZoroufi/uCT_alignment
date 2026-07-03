"""
rerun: FULL segmentation of the 4 bones -- every atlas bone shown in its own
colour (from {occ}/allsegs/{tag}_bone_{L}.stl, produced by seg_allparts.py).
Tibia = bones 1 (red) & 4 (blue). Grid: TOP=B256M1, BOTTOM=B256M7; LEFT=CL, RIGHT=SL.
Toggle individual bones in the tree.
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

rr.init("allsegs", spawn=True)
print("init ok", flush=True)
for pair, tag, col, row in BONES:
    seg = RECON / f"{pair}_occ" / "allsegs"
    files = sorted(seg.glob(f"{tag}_bone_*.stl"))
    if not files:
        print(f"{pair} {tag}: no allsegs files yet", flush=True)
        continue
    meshes = {int(f.stem.split("_")[-1]):
              trimesh.load(str(f), force="mesh", process=False) for f in files}
    c = np.vstack([m.vertices for m in meshes.values()]).mean(0)
    off = (np.array([col * SPACING, row * SPACING, 0.0]) - c).astype(np.float32)
    for L, m in sorted(meshes.items()):
        name = f"bone{L}_TIBIA" if L in (1, 4) else f"bone{L}"
        rr.log(f"{pair}/{tag.upper()}/{name}", rr.Mesh3D(
            vertex_positions=(m.vertices + off).astype(np.float32),
            triangle_indices=m.faces,
            vertex_colors=np.tile(LCOL.get(L, [150, 150, 150]), (len(m.vertices), 1))))
    print(f"{pair} {tag}: labels {sorted(meshes)}  "
          f"(sizes {[len(meshes[L].vertices) for L in sorted(meshes)]})", flush=True)
print("\ndone. each colour = one atlas bone; TIBIA = bones 1 (red) & 4 (blue).", flush=True)
print("TOP=B256M1, BOTTOM=B256M7; LEFT=CL, RIGHT=SL. Check every scan piece got a "
      "sensible label and the tibia (1&4) sits on the two ossification centres.",
      flush=True)
