import trimesh, numpy as np
from scipy.ndimage import gaussian_filter1d

sample = trimesh.load('results_CT_20260201_193908/CT_20260310_193908_aligned.stl', force='mesh', process=False)
sample.merge_vertices()
comps = sample.split(only_watertight=False)
comps = sorted(comps, key=lambda m: len(m.faces), reverse=True)
seg = comps[1]   # the combined part_2+part_3 mesh
print(f"Segment: {len(seg.faces):,} faces  X=[{seg.bounds[0,0]:.2f},{seg.bounds[1,0]:.2f}]")

bin_width_mm      = 0.5
sigma_mm          = 1.0
bridge_ratio      = 0.35
min_peak_fraction = 0.10

centroid       = seg.vertices.mean(0)
v              = seg.vertices - centroid
cov            = (v.T @ v) / max(len(v) - 1, 1)
eigvecs        = np.linalg.eigh(cov)[1]
pca_axes       = eigvecs.T[::-1]   # rows: PC1, PC2, PC3
face_centroids = seg.vertices[seg.faces].mean(axis=1)
outward        = face_centroids - centroid
outer_mask     = (seg.face_normals * outward).sum(axis=1) > 0
print(f"Outer faces: {outer_mask.sum():,} / {len(seg.faces):,}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes_plot = plt.subplots(3, 1, figsize=(12, 10))

overall_best_ratio = float('inf')
overall_best_ax    = None

for ax_i, axis in enumerate(pca_axes):
    face_proj    = face_centroids @ axis
    p_min, p_max = float(face_proj.min()), float(face_proj.max())
    extent       = p_max - p_min
    n_bins       = max(10, int(np.ceil(extent / bin_width_mm)))
    bins         = np.linspace(p_min, p_max, n_bins + 1)
    counts       = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        counts[i] = float(((face_proj >= bins[i]) & (face_proj < bins[i+1]) & outer_mask).sum())
    counts[-1] += float(((face_proj == p_max) & outer_mask).sum())
    smooth = gaussian_filter1d(counts, sigma=sigma_mm / bin_width_mm)

    prefix_max = np.maximum.accumulate(smooth)
    suffix_max = np.maximum.accumulate(smooth[::-1])[::-1]
    global_max = float(smooth.max())

    best_v_ratio = float('inf')
    best_v_idx   = None
    best_lp = best_rp = None
    for vi in range(1, n_bins - 1):
        lp = float(prefix_max[vi - 1])
        rp = float(suffix_max[vi + 1])
        pm = min(lp, rp)
        if pm < 1e-6 or pm < min_peak_fraction * global_max:
            continue
        r = float(smooth[vi]) / pm
        if r < best_v_ratio:
            best_v_ratio = r
            best_v_idx   = vi
            best_lp, best_rp = lp, rp

    detected = best_v_idx is not None and best_v_ratio <= bridge_ratio
    cut_pos  = float(bins[best_v_idx] + bins[best_v_idx+1]) / 2 if best_v_idx is not None else None
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax = axes_plot[ax_i]
    ax.bar(bin_centers, counts, width=bin_width_mm * 0.9, alpha=0.4, color='steelblue', label='raw')
    ax.plot(bin_centers, smooth, 'k-', lw=2, label='smoothed')
    if best_v_idx is not None:
        ax.axvline(cut_pos, color='red', lw=2, linestyle='--', label=f'cut={cut_pos:.1f}mm')
        ax.axvspan(bins[best_v_idx], bins[best_v_idx+1], alpha=0.3, color='black', label='valley bin')
    ax.set_title(f"PC{ax_i+1}  ratio={best_v_ratio:.3f}  detected={detected}")
    ax.set_xlabel('Projection (mm)')
    ax.set_ylabel('Face count')
    ax.legend(fontsize=8)

    print(f"PC{ax_i+1} axis={axis.round(3)}")
    print(f"  extent={extent:.1f}mm  n_bins={n_bins}")
    print(f"  best_ratio={best_v_ratio:.3f}  valley_bin={best_v_idx}  detected={detected}")
    if best_v_idx is not None:
        print(f"  cut_pos={cut_pos:.2f}  bin=[{bins[best_v_idx]:.2f},{bins[best_v_idx+1]:.2f}]")
        print(f"  valley_val={smooth[best_v_idx]:.1f}  left_peak={best_lp:.1f}  right_peak={best_rp:.1f}")
        left_f  = (face_proj < bins[best_v_idx]).sum()
        right_f = (face_proj > bins[best_v_idx+1]).sum()
        print(f"  left_faces={left_f:,}  right_faces={right_f:,}")

    if best_v_ratio < overall_best_ratio:
        overall_best_ratio = best_v_ratio
        overall_best_ax    = ax_i + 1

print(f"\nWinning axis: PC{overall_best_ax}  ratio={overall_best_ratio:.3f}  {'SPLIT' if overall_best_ratio <= bridge_ratio else 'NO SPLIT'}")

plt.tight_layout()
plt.savefig('results_CT_20260201_193908/_debug_bridge_profiles.png', dpi=150)
print("Saved: results_CT_20260201_193908/_debug_bridge_profiles.png")

# Also show where the cut lands relative to part_2 and part_3
print("\n--- Where does the cut plane land? ---")
p2 = trimesh.load('results_CT_20260201_193908/all_segments/sample_part_2.stl', force='mesh', process=False)
p3 = trimesh.load('results_CT_20260201_193908/all_segments/sample_part_3.stl', force='mesh', process=False)
for ax_i, axis in enumerate(pca_axes):
    p2_proj = (p2.vertices @ axis)
    p3_proj = (p3.vertices @ axis)
    print(f"PC{ax_i+1}: part_2 proj=[{p2_proj.min():.2f},{p2_proj.max():.2f}]  part_3 proj=[{p3_proj.min():.2f},{p3_proj.max():.2f}]  overlap={max(0, min(p2_proj.max(),p3_proj.max())-max(p2_proj.min(),p3_proj.min())):.2f}mm")
