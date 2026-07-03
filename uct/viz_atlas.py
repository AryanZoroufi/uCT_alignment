"""
rerun: visualize the atlas template itself (atlas.npz).

Marching-cubes each of the 9 atlas bone labels into its own coloured mesh, in the
atlas's native physical frame (idx * pitch + origin). Tibia = labels 1 & 4.
Toggle bones in the rerun tree (atlas/bone_k).
"""
from pathlib import Path
import numpy as np
import rerun as rr
from skimage import measure

HERE = Path(__file__).parent
a = np.load(HERE / "atlas.npz", allow_pickle=True)
lbl = a["bone_labels"]
pitch = float(a["pitch"])
origin = a["origin"].astype(np.float32)
ids = [int(x) for x in a["bone_ids"]]
print(f"atlas anatomy={a['anatomy']}  n_bones={int(a['n_bones'])}  "
      f"pitch={pitch}  grid={lbl.shape}", flush=True)

COLORS = [[230, 25, 75], [60, 180, 75], [255, 215, 20], [0, 130, 200],
          [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
          [170, 110, 40]]

rr.init("atlas", spawn=True)
print("init ok", flush=True)
for i, L in enumerate(ids):
    mask = lbl == L
    if not mask.any():
        continue
    v, f, _, _ = measure.marching_cubes(
        np.pad(mask, 1).astype(np.float32), 0.5)
    v = ((v - 1) * pitch + origin).astype(np.float32)
    col = COLORS[i % len(COLORS)]
    tag = f"bone_{L}_TIBIA" if L in (1, 4) else f"bone_{L}"
    rr.log(f"atlas/{tag}", rr.Mesh3D(
        vertex_positions=v, triangle_indices=f,
        vertex_colors=np.tile(col, (len(v), 1))))
    print(f"  bone {L}: {int(mask.sum()):,} vox  {len(v):,}v  color {col}"
          f"{'   <-- TIBIA' if L in (1, 4) else ''}", flush=True)
print("\ndone. 9 bones coloured; TIBIA = bones 1 & 4. Toggle in the tree.",
      flush=True)
