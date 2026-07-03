"""
Build a purpose-built 2-part TIBIA atlas by extracting labels 1 & 4 (the two
tibial ossification centres) from the generic 9-bone foot atlas (atlas.npz).

Keeps labels 1 & 4 (so downstream part IDs are unchanged), rebuilds solid +
surface_points from just those two bones, and writes atlas_tibia.npz. Registering
THIS (a tibia) to a tibia scan should both fit well AND label the parts
correctly -- unlike the foot atlas, whose good global fit still mislabels parts.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion

HERE = Path(__file__).parent
a = np.load(HERE / "atlas.npz", allow_pickle=True)
lbl = a["bone_labels"]
pitch = float(a["pitch"])
origin = a["origin"].astype(np.float32)

tibia = np.isin(lbl, [1, 4])
new_lbl = np.where(tibia, lbl, 0).astype(np.uint8)
new_solid = tibia
struct = np.ones((3, 3, 3), bool)
boundary = new_solid & ~binary_erosion(new_solid, structure=struct)
surf = (np.argwhere(boundary).astype(np.float32) * pitch + origin).astype(np.float32)

np.savez(HERE / "atlas_tibia.npz",
         bone_labels=new_lbl, solid=new_solid,
         origin=origin, pitch=np.float32(pitch),
         surface_points=surf, n_bones=np.int64(2),
         bone_ids=np.array([1, 4], np.int32), anatomy="tibia_2parts")

for L in (1, 4):
    m = lbl == L
    c = (np.argwhere(m).mean(0) * pitch + origin) if m.any() else None
    print(f"  label {L}: {int(m.sum()):,} vox  centroid(atlas mm)="
          f"{np.round(c, 1) if c is not None else '?'}", flush=True)
print(f"tibia atlas: labels {np.unique(new_lbl).tolist()}  solid {int(new_solid.sum()):,} vox  "
      f"surf {len(surf):,} pts  -> atlas_tibia.npz", flush=True)
