"""Surface + atlas-segment a contralateral/surgical pair; save tibia parts 1 & 4."""
import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).parent))
from vox_to_stl import vox_to_stl
from pipeline import _segment_via_atlas

import growth_config as cfg

ATLAS = Path(__file__).parent / "atlas.npz"
# Which pair to segment: `python seg_pair.py [PAIR]` (default B256M1).
PAIR = sys.argv[1] if len(sys.argv) > 1 else "B256M1"
_conf = cfg.get(PAIR)
OUT = Path(_conf["occ"])
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = {"cl": Path(_conf["cl_vox"]), "sl": Path(_conf["sl_vox"])}
print(f"pair={PAIR}  out={OUT}", flush=True)

for tag, vox in PAIRS.items():
    print(f"\n===== {tag}: {vox.name} =====", flush=True)
    stl = OUT / f"{tag}.stl"
    vox_to_stl(str(vox), str(stl), gaussian_sigma=3.0, decimate=0.05,
               taubin_iterations=100, step_size=2, n_watershed=1)
    bones = _segment_via_atlas(Path(stl), ATLAS)
    print(f"  {tag} atlas bones: {sorted(bones)}", flush=True)
    for pid in (1, 4):
        if pid in bones:
            bones[pid].export(OUT / f"{tag}_part{pid}.stl")
            m = bones[pid]
            print(f"  {tag} part{pid}: {len(m.vertices)}v vol={abs(m.volume):.0f} "
                  f"centroid={np.round(m.vertices.mean(0),1)}", flush=True)
        else:
            print(f"  {tag} MISSING part {pid}", flush=True)
print("\n===== seg_pair done =====", flush=True)
