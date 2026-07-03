"""
Shows exactly which faces from the combined segment ended up in part_2 vs part_3,
and highlights the region near the cut plane to identify misassigned geometry.
"""
import trimesh, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.insert(0, '.')
from pipeline import _compute_bridge_profile

sample = trimesh.load('results_CT_20260201_193908/CT_20260310_193908_aligned.stl', force='mesh', process=False)
sample.merge_vertices()
comps = sample.split(only_watertight=False)
comps = sorted(comps, key=lambda m: len(m.faces), reverse=True)
seg = comps[1]

p = _compute_bridge_profile(seg)
best_axis = p['axis']
cut_pos   = p['cut_pos']
bins      = p['bins']
vi        = p['valley_idx']

face_centroids = seg.vertices[seg.faces].mean(axis=1)
face_proj      = face_centroids @ best_axis

# The valley bin and its neighbours
b_lo = bins[vi]
b_hi = bins[vi + 1]
print(f"Cut axis (PC1): {best_axis.round(4)}")
print(f"Valley bin: [{b_lo:.3f}, {b_hi:.3f}]  (width={b_hi-b_lo:.2f}mm)")
print(f"Faces in valley bin: {((face_proj >= b_lo) & (face_proj <= b_hi)).sum():,}")
print(f"Faces left of valley: {(face_proj < b_lo).sum():,}")
print(f"Faces right of valley: {(face_proj > b_hi).sum():,}")

# Check: after slicing, what dropped?
plane_origin = best_axis * cut_pos
left_raw  = trimesh.intersections.slice_mesh_plane(seg, -best_axis, plane_origin, cap=True)
right_raw = trimesh.intersections.slice_mesh_plane(seg,  best_axis, plane_origin, cap=True)

def largest_and_rest(mesh):
    parts = sorted(mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True)
    return parts[0], parts[1:]

left_main,  left_orphans  = largest_and_rest(left_raw)
right_main, right_orphans = largest_and_rest(right_raw)

print(f"\nAfter slice:")
print(f"  LEFT  → kept {len(left_main.faces):,} faces  +  dropped {sum(len(o.faces) for o in left_orphans):,} faces across {len(left_orphans)} orphan(s)")
print(f"  RIGHT → kept {len(right_main.faces):,} faces  +  dropped {sum(len(o.faces) for o in right_orphans):,} faces across {len(right_orphans)} orphan(s)")

for label, orphans in [("LEFT orphans", left_orphans), ("RIGHT orphans", right_orphans)]:
    for i, o in enumerate(orphans):
        b = o.bounds
        print(f"  {label}[{i}]: {len(o.faces)} faces  X=[{b[0,0]:.2f},{b[1,0]:.2f}]  Y=[{b[0,1]:.2f},{b[1,1]:.2f}]  Z=[{b[0,2]:.2f},{b[1,2]:.2f}]")

# X-overlap: how much do the two kept halves share in X?
lx = (left_main.bounds[0,0],  left_main.bounds[1,0])
rx = (right_main.bounds[0,0], right_main.bounds[1,0])
x_overlap = max(0, min(lx[1], rx[1]) - max(lx[0], rx[0]))
print(f"\nX-overlap between kept halves: {x_overlap:.2f}mm")
print(f"  LEFT  X=[{lx[0]:.2f},{lx[1]:.2f}]")
print(f"  RIGHT X=[{rx[0]:.2f},{rx[1]:.2f}]")

# The cut plane in world coords: axis · x = cut_pos
# At Y=mean, Z=mean, what X does the plane cross?
mean_y = float(seg.vertices[:,1].mean())
mean_z = float(seg.vertices[:,2].mean())
# axis[0]*x + axis[1]*mean_y + axis[2]*mean_z = cut_pos
x_at_mean_yz = (cut_pos - best_axis[1]*mean_y - best_axis[2]*mean_z) / best_axis[0]
print(f"\nCut plane at mean Y={mean_y:.1f}, Z={mean_z:.1f}: X ≈ {x_at_mean_yz:.2f}mm")
print(f"Cut plane normal: {best_axis.round(4)}")
print(f"  X component: {best_axis[0]:.3f}  Y component: {best_axis[1]:.3f}  Z component: {best_axis[2]:.3f}")
print(f"  → cut is {abs(np.degrees(np.arccos(abs(best_axis[0])))):.1f}° away from a pure X cut")

# Smooth profile plot with valley region highlighted
smooth = p['smooth']
raw    = p['counts']
bcs    = (bins[:-1] + bins[1:]) / 2
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(bcs, raw, width=(bins[1]-bins[0])*0.9, alpha=0.35, color='steelblue', label='raw counts')
ax.plot(bcs, smooth, 'k-', lw=2, label='smoothed')
ax.axvline(cut_pos, color='red', lw=2, ls='--', label=f'cut = {cut_pos:.1f}mm')
ax.axvspan(b_lo, b_hi, alpha=0.5, color='black', label='valley bin (bridge)')
ax.set_title(f"Bridge profile along PC1  ratio={p['ratio']:.3f}  valley={smooth[vi]:.0f}  peaks={p['left_peak']:.0f}/{p['right_peak']:.0f}")
ax.set_xlabel('Projection along PC1 (mm)')
ax.set_ylabel('Outer face count')
ax.legend()
plt.tight_layout()
plt.savefig('results_CT_20260201_193908/_debug_bridge_profile_PC1.png', dpi=150)
print("\nSaved: results_CT_20260201_193908/_debug_bridge_profile_PC1.png")
