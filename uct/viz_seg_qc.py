"""
rerun QC viewer for the atlas segmentation (detailed + alignment debugging).

For each bone (B256M1 + B256M7, CL & SL) shows, in a 2x2 grid:
  * scan_bone   -- the FULL high-res tibia surface mesh (grey)  [= bone post-align]
  * atlas_Lk    -- the registered ATLAS template, split by atlas label 1-4 and
                   colour-coded (muted)                          [= mesh for atlas]
  * seg_part1/4 -- the actual segmented parts 1 (bright blue) & 4 (bright orange)

So you can see how well the atlas template sits on the real bone (B256M7's poor
IoU ~0.35 should show the atlas 1/4 regions NOT matching the bone / each other),
and where parts 1 & 4 ended up.

All meshes are in raw mm (verified: {tag}_atlasdbg.npz scan bbox == full-STL raw
bbox). Each bone is recentred on its own bone centroid, then placed in the grid.

Requires {tag}_atlasdbg.npz in each *_occ dir (from seg_debug.py).
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
import rerun as rr

RECON = Path(__file__).parent / "../bones_to_recon"
SPACING = 22.0                              # mm between bones in the grid
BONE_COL = [150, 150, 150]
ATLAS_COL = {1: [30, 60, 170], 2: [20, 150, 150], 3: [40, 150, 40], 4: [170, 95, 20]}
SEG1_COL = [90, 160, 255]                   # bright blue  = segmented part1
SEG4_COL = [255, 175, 70]                   # bright orange = segmented part4

BONES = [("B256M1", "cl", 0, 0), ("B256M1", "sl", 1, 0),
         ("B256M7", "cl", 0, 1), ("B256M7", "sl", 1, 1)]


def submesh(V, F, vl, L):
    """Faces whose majority vertex-label == L, remapped to a compact submesh."""
    keep = (vl[F] == L).sum(1) >= 2
    ff = F[keep]
    if not len(ff):
        return None, None
    used = np.unique(ff)
    remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
    return V[used].astype(np.float32), remap[ff]


rr.init("seg_qc", spawn=True)
print("init ok", flush=True)

for pair, tag, col, row in BONES:
    occ = RECON / f"{pair}_occ"
    bone = trimesh.load(str(occ / f"{tag}.stl"), force="mesh", process=False)
    p1 = trimesh.load(str(occ / f"{tag}_part1.stl"), force="mesh", process=False)
    p4 = trimesh.load(str(occ / f"{tag}_part4.stl"), force="mesh", process=False)
    dbg = np.load(str(occ / f"{tag}_atlasdbg.npz"))
    av, af, vl = dbg["atlas_verts"], dbg["atlas_faces"], dbg["atlas_vlabels"]

    c = bone.vertices.mean(0)
    off = np.array([col * SPACING, row * SPACING, 0.0]) - c    # recentre + grid
    base = f"{pair}/{tag.upper()}"

    rr.log(f"{base}/scan_bone", rr.Mesh3D(
        vertex_positions=(bone.vertices + off).astype(np.float32),
        triangle_indices=bone.faces,
        vertex_colors=np.tile(BONE_COL, (len(bone.vertices), 1))))

    for L in (1, 2, 3, 4):
        sv, sf = submesh(av, af, vl, L)
        if sv is None:
            continue
        rr.log(f"{base}/atlas_L{L}", rr.Mesh3D(
            vertex_positions=sv + off.astype(np.float32), triangle_indices=sf,
            vertex_colors=np.tile(ATLAS_COL[L], (len(sv), 1))))

    rr.log(f"{base}/seg_part1_BLUE", rr.Mesh3D(
        vertex_positions=(p1.vertices + off).astype(np.float32),
        triangle_indices=p1.faces,
        vertex_colors=np.tile(SEG1_COL, (len(p1.vertices), 1))))
    rr.log(f"{base}/seg_part4_ORANGE", rr.Mesh3D(
        vertex_positions=(p4.vertices + off).astype(np.float32),
        triangle_indices=p4.faces,
        vertex_colors=np.tile(SEG4_COL, (len(p4.vertices), 1))))

    labs = sorted(set(int(x) for x in np.unique(vl) if x in (1, 2, 3, 4)))
    print(f"{base}: bone {len(bone.faces):,}f  atlas labels present {labs}  "
          f"part1 {len(p1.faces)}f  part4 {len(p4.faces)}f", flush=True)

print("\ndone. GREY=scan bone (high-res), muted blue/cyan/green/orange = atlas "
      "labels 1/2/3/4 (registered template), BRIGHT blue/orange = segmented "
      "part1/part4.", flush=True)
print("layout: TOP=B256M1 (validated), BOTTOM=B256M7 (suspect); LEFT=CL, RIGHT=SL. "
      "Toggle entities in the tree. Look: does atlas label-1 (dark blue) & label-4 "
      "(dark orange) sit ON the two ossification centres?", flush=True)
