"""
Segment B256M1 + B256M7 (cl + sl) with the NEW chamfer registration and save
parts 1 & 4 to {occ}_chamfer (config pairs 'B256M1c' / 'B256M7c'). Reuses the
existing full-bone STLs (skips vox_to_stl). Prints new vs old part centroids.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import _segment_via_atlas
import growth_config as cfg

ATLAS = Path(__file__).parent / "atlas.npz"
for pair in ["B256M1", "B256M7"]:
    base_occ = Path(cfg.get(pair)["occ"])
    out = Path(cfg.get(pair + "c")["occ"]); out.mkdir(parents=True, exist_ok=True)
    for tag in ["cl", "sl"]:
        stl = base_occ / f"{tag}.stl"
        print(f"\n===== {pair} {tag} (chamfer) =====", flush=True)
        bones = _segment_via_atlas(stl, ATLAS, registration="chamfer")
        for pid in (1, 4):
            if pid in bones:
                bones[pid].export(out / f"{tag}_part{pid}.stl")
                m = bones[pid]
                oldp = base_occ / f"{tag}_part{pid}.stl"
                oc = trimesh.load(str(oldp), process=False).vertices.mean(0) if oldp.exists() else None
                print(f"  part{pid}: {len(m.vertices)}v  centroid={np.round(m.vertices.mean(0),1)}"
                      f"  (old {np.round(oc,1) if oc is not None else '?'})", flush=True)
            else:
                print(f"  MISSING part {pid}", flush=True)
print("\n===== seg_chamfer done =====", flush=True)
