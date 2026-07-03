"""
Regenerate atlas-alignment DEBUG artifacts for QC.

For each (pair, tag) call _segment_via_atlas on the EXISTING full-bone STL with
debug_dir set -> dumps {tag}_atlasdbg.npz (atlas template + aligned scan surface,
both in scan-physical frame). Skips vox_to_stl (reuses the .stl) so it's just the
atlas registration (~1-2 min/scan). Also prints a frame check (full-STL bbox raw
& /STEP) so the viewer knows how the debug meshes overlay the full bone.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import _segment_via_atlas
import growth_config as cfg

ATLAS = Path(__file__).parent / "atlas.npz"
STEP = 2
PAIRS = ["B256M1", "B256M7"]
TAGS = ["cl", "sl"]

for pair in PAIRS:
    occ = Path(cfg.get(pair)["occ"])
    for tag in TAGS:
        stl = occ / f"{tag}.stl"
        print(f"\n===== {pair} {tag}: {stl.name} =====", flush=True)
        _segment_via_atlas(stl, ATLAS, debug_dir=occ)
        m = trimesh.load(str(stl), force="mesh", process=False)
        r = m.vertices
        print(f"    [frame] {pair} {tag} full-STL bbox raw "
              f"{np.round(r.min(0),1)}..{np.round(r.max(0),1)}  /STEP "
              f"{np.round(r.min(0)/STEP,1)}..{np.round(r.max(0)/STEP,1)}", flush=True)
print("\n===== seg_debug done =====", flush=True)
