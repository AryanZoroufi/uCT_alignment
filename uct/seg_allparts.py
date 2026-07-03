"""
Segment the 4 bones with the working default (ICP) pipeline and export ALL atlas
bones found (not just tibia parts 1 & 4) to {occ}/allsegs/{tag}_bone_{L}.stl, so
we can QC the whole segmentation before focusing on the tibia parts.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import _segment_via_atlas
import growth_config as cfg

ATLAS = Path(__file__).parent / "atlas.npz"
for pair in ["B256M1", "B256M7"]:
    occ = Path(cfg.get(pair)["occ"])
    out = occ / "allsegs"; out.mkdir(parents=True, exist_ok=True)
    for tag in ["cl", "sl"]:
        stl = occ / f"{tag}.stl"
        print(f"\n===== {pair} {tag} =====", flush=True)
        bones = _segment_via_atlas(stl, ATLAS)          # default ICP
        print(f"  {pair} {tag} bones: {sorted(bones)}", flush=True)
        for L, m in sorted(bones.items()):
            m.export(out / f"{tag}_bone_{L}.stl")
            print(f"    bone {L}: {len(m.vertices):,}v  vol={abs(m.volume):.0f}  "
                  f"centroid={np.round(m.vertices.mean(0), 1)}", flush=True)
print("\n===== seg_allparts done =====", flush=True)
