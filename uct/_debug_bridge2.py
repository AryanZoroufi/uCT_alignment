import trimesh, numpy as np
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.insert(0, '.')
from pipeline import _compute_bridge_profile

sample = trimesh.load('results_CT_20260201_193908/CT_20260310_193908_aligned.stl', force='mesh', process=False)
sample.merge_vertices()
comps = sample.split(only_watertight=False)
comps = sorted(comps, key=lambda m: len(m.faces), reverse=True)
seg = comps[1]
print(f"Segment: {len(seg.faces):,} faces")

p = _compute_bridge_profile(seg)
print(f"Detected={p['detected']}  ratio={p['ratio']:.3f}  cut_pos={p['cut_pos']}  axis={p['axis'].round(3)}")

best_axis    = p['axis']
cut_pos      = p['cut_pos']
plane_origin = best_axis * cut_pos

left_raw  = trimesh.intersections.slice_mesh_plane(seg, -best_axis, plane_origin, cap=True)
right_raw = trimesh.intersections.slice_mesh_plane(seg,  best_axis, plane_origin, cap=True)

print(f"\nleft_raw:  {len(left_raw.faces):,} faces")
print(f"right_raw: {len(right_raw.faces):,} faces")

for label, raw in [("LEFT", left_raw), ("RIGHT", right_raw)]:
    parts = raw.split(only_watertight=False)
    parts = sorted(parts, key=lambda m: len(m.faces), reverse=True)
    print(f"\n{label} splits into {len(parts)} components:")
    for i, part in enumerate(parts[:6]):
        b = part.bounds
        print(f"  [{i}] {len(part.faces):,} faces  X=[{b[0,0]:.2f},{b[1,0]:.2f}]  Y=[{b[0,1]:.2f},{b[1,1]:.2f}]  Z=[{b[0,2]:.2f},{b[1,2]:.2f}]")
    chosen = parts[0]
    print(f"  --> _largest_component keeps [{0}]: {len(chosen.faces):,} faces")

# Compare with actual output files
print("\n--- Actual output files ---")
p2 = trimesh.load('results_CT_20260201_193908/all_segments/sample_part_2.stl', force='mesh', process=False)
p3 = trimesh.load('results_CT_20260201_193908/all_segments/sample_part_3.stl', force='mesh', process=False)
for name, m in [("part_2", p2), ("part_3", p3)]:
    b = m.bounds
    print(f"{name}: {len(m.faces):,} faces  X=[{b[0,0]:.2f},{b[1,0]:.2f}]  Y=[{b[0,1]:.2f},{b[1,1]:.2f}]  Z=[{b[0,2]:.2f},{b[1,2]:.2f}]")
