"""
rerun: segmentation RESULTS for the 4 bones (working default-ICP pipeline).

Each high-res tibia is painted by its segmented parts: vertices near part 1 -> BLUE,
near part 4 -> ORANGE, everything else (other bones in the FOV) -> grey. This shows
the actual segmentation on the true surface (not the coarse atlas part-meshes).

Grid:  TOP row = B256M1 (QC PASS, 177x control),  BOTTOM row = B256M7 (QC FAIL,
gap mislocalized);  LEFT = CL,  RIGHT = SL.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import rerun as rr

RECON = Path(__file__).parent / "../bones_to_recon"
SPACING = 22.0
THR = 0.8                               # raw-mm: paint bone vertices within THR of a part
P1 = [70, 150, 255]                     # blue  = part1
P4 = [255, 160, 50]                     # orange = part4
GREY = [155, 155, 155]

BONES = [("B256M1", "cl", 0, 0, "PASS"), ("B256M1", "sl", 1, 0, "PASS"),
         ("B256M7", "cl", 0, 1, "FAIL"), ("B256M7", "sl", 1, 1, "FAIL")]

rr.init("seg_result", spawn=True)
print("init ok", flush=True)

for pair, tag, col, row, qc in BONES:
    occ = RECON / f"{pair}_occ"
    bone = trimesh.load(str(occ / f"{tag}.stl"), force="mesh", process=False)
    p1 = trimesh.load(str(occ / f"{tag}_part1.stl"), force="mesh", process=False)
    p4 = trimesh.load(str(occ / f"{tag}_part4.stl"), force="mesh", process=False)

    bv = bone.vertices
    d1 = cKDTree(p1.vertices).query(bv)[0]
    d4 = cKDTree(p4.vertices).query(bv)[0]
    colors = np.tile(GREY, (len(bv), 1)).astype(np.uint8)
    n1 = (d1 < THR) & (d1 <= d4)
    n4 = (d4 < THR) & (d4 < d1)
    colors[n1] = P1
    colors[n4] = P4

    c = bv.mean(0)
    off = (np.array([col * SPACING, row * SPACING, 0.0]) - c).astype(np.float32)
    base = f"{pair}__QC_{qc}/{tag.upper()}"
    rr.log(base, rr.Mesh3D(vertex_positions=(bv + off).astype(np.float32),
           triangle_indices=bone.faces, vertex_colors=colors))
    print(f"{base}: bone {len(bone.faces):,}f  part1-painted {int(n1.sum()):,}v  "
          f"part4-painted {int(n4.sum()):,}v", flush=True)

print("\ndone. grey=tibia surface, BLUE=part1, ORANGE=part4.", flush=True)
print("TOP=B256M1 (QC PASS: parts cap the two centres with a clear gap), "
      "BOTTOM=B256M7 (QC FAIL: parts mislocalized -> gap lands in bone).", flush=True)
