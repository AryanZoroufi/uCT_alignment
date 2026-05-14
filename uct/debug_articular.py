"""Debug: diagnose d-distribution in articular patches."""
import sys
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import _facing_side_indices, _articular_mask_protrusion
import trimesh.proximity

OUT = Path("results_CT_20260201_190419")
neighborhood_radius = 0.10
patch_radius        = 0.30
yz_eps              = 0.005

print("Loading meshes ...")
s2 = trimesh.load(str(OUT / "sample_2.stl"), force="mesh", process=False); s2.merge_vertices()
s1 = trimesh.load(str(OUT / "sample_1.stl"), force="mesh", process=False); s1.merge_vertices()
r2 = trimesh.load(str(OUT / "ref_2.stl"),    force="mesh", process=False); r2.merge_vertices()
r1 = trimesh.load(str(OUT / "ref_1.stl"),    force="mesh", process=False); r1.merge_vertices()

# ── Compute distances ──────────────────────────────────────────────────────
_, s_dists, _ = trimesh.proximity.closest_point(s1, s2.vertices)
_, r_dists, _ = trimesh.proximity.closest_point(r1, r2.vertices)
s_dists = s_dists.astype(np.float64)
r_dists = r_dists.astype(np.float64)

# ── Detect spike (original approach) ──────────────────────────────────────
verts = s2.vertices
v = verts - verts.mean(0)
cov = (v.T @ v) / max(len(v)-1, 1)
eigvals = np.linalg.eigvalsh(cov)
bone_length = float(np.sqrt(max(eigvals[-1], 0.0))) * 2
r_abs       = neighborhood_radius * bone_length
r_prime_abs = patch_radius * bone_length
yz_eps_abs  = yz_eps * bone_length

b1_cx = s1.vertices[:,0].mean(); b2_cx = verts[:,0].mean()
facing_idx = _facing_side_indices(verts, b1_cx < b2_cx, yz_eps_abs)
facing_yz = verts[facing_idx, 1:3]; facing_d = s_dists[facing_idx]
tree_facing = KDTree(facing_yz)
nbr_lists = tree_facing.query_ball_point(facing_yz, r=r_abs)
d_bar_facing = np.array([facing_d[n].mean() if n else facing_d[i]
                         for i, n in enumerate(nbr_lists)], dtype=np.float64)
deviation = facing_d - d_bar_facing
lowest_in_facing = int(np.argmin(deviation))
lowest_idx = int(facing_idx[lowest_in_facing])
lowest_pt = verts[lowest_idx]
d_bar_spike = float(d_bar_facing[lowest_in_facing])

print(f"\nSpike: d={s_dists[lowest_idx]:.4f}  d_bar={d_bar_spike:.4f}  deviation={deviation[lowest_in_facing]:.4f}")
print(f"Spike tip: y={lowest_pt[1]:.3f}  z={lowest_pt[2]:.3f}")

# ── y-z circle (all vertices, no dedup) ───────────────────────────────────
yz_dist_s = np.sqrt((verts[:,1]-lowest_pt[1])**2 + (verts[:,2]-lowest_pt[2])**2)
in_circle  = yz_dist_s < r_prime_abs
print(f"\n--- sample_2: all vertices in y-z circle ---")
print(f"  count: {in_circle.sum()}")
d_circ = s_dists[in_circle]
for pct in [10,25,50,75,90,95,99,100]:
    print(f"  d p{pct:3d}: {np.percentile(d_circ, pct):.3f}")

# ── min-d per cell (current approach) ─────────────────────────────────────
cell_size = r_prime_abs / 10.0
yz_cells  = (verts[:,1:3] / cell_size).round().astype(np.int64)
cell_best: dict = {}
for i in range(len(verts)):
    if yz_dist_s[i] >= r_prime_abs: continue
    cell = (int(yz_cells[i,0]), int(yz_cells[i,1]))
    if cell not in cell_best or s_dists[i] < cell_best[cell][0]:
        cell_best[cell] = (s_dists[i], i)
s_patch_idx = np.array([idx for _, idx in cell_best.values()], dtype=np.int64)

print(f"\n--- sample_2: min-d per cell ---")
print(f"  count: {len(s_patch_idx)}")
dp = s_dists[s_patch_idx]
for pct in [10,25,50,75,90,95,99,100]:
    print(f"  d p{pct:3d}: {np.percentile(dp, pct):.3f}")
print(f"  vertices with d > d_bar ({d_bar_spike:.2f}): {(dp > d_bar_spike).sum()}")
print(f"  vertices with d > 1.5*d_bar ({1.5*d_bar_spike:.2f}): {(dp > 1.5*d_bar_spike).sum()}")

# ── ref_2 circle ───────────────────────────────────────────────────────────
r2_verts   = r2.vertices
r2v        = r2_verts - r2_verts.mean(0)
r2_bl      = float(np.sqrt(max(np.linalg.eigvalsh(
    (r2v.T@r2v)/max(len(r2v)-1,1))[-1],0.0)))*2
r2_prime   = patch_radius * r2_bl
yz_dist_r2 = np.sqrt((r2_verts[:,1]-lowest_pt[1])**2 + (r2_verts[:,2]-lowest_pt[2])**2)
r2_in_circle = yz_dist_r2 < r2_prime

print(f"\n--- ref_2: all vertices in y-z circle ---")
print(f"  count: {r2_in_circle.sum()}")
dr_circ = r_dists[r2_in_circle]
for pct in [10,25,50,75,90,95,99,100]:
    print(f"  d p{pct:3d}: {np.percentile(dr_circ, pct):.3f}")

cell_size_r2 = r2_prime / 10.0
yz_cells_r2  = (r2_verts[:,1:3] / cell_size_r2).round().astype(np.int64)
cell_best_r2: dict = {}
for i in range(len(r2_verts)):
    if yz_dist_r2[i] >= r2_prime: continue
    cell = (int(yz_cells_r2[i,0]), int(yz_cells_r2[i,1]))
    if cell not in cell_best_r2 or r_dists[i] < cell_best_r2[cell][0]:
        cell_best_r2[cell] = (r_dists[i], i)
r2_patch_idx = np.array([idx for _, idx in cell_best_r2.values()], dtype=np.int64)

print(f"\n--- ref_2: min-d per cell ---")
print(f"  count: {len(r2_patch_idx)}")
drp = r_dists[r2_patch_idx]
for pct in [10,25,50,75,90,95,99,100]:
    print(f"  d p{pct:3d}: {np.percentile(drp, pct):.3f}")

# ── Area-weighted volumes ──────────────────────────────────────────────────
def vert_areas(mesh):
    a = np.zeros(len(mesh.vertices), dtype=np.float64)
    np.add.at(a, mesh.faces[:,0], mesh.area_faces/3.)
    np.add.at(a, mesh.faces[:,1], mesh.area_faces/3.)
    np.add.at(a, mesh.faces[:,2], mesh.area_faces/3.)
    return a

sa = vert_areas(s2); ra = vert_areas(r2)

print(f"\n{'='*50}")
print("Volume comparison")
print('='*50)

# all-in-circle
s_vol_all = float(np.sum(s_dists[in_circle] * sa[in_circle]))
r_vol_all = float(np.sum(r_dists[r2_in_circle] * ra[r2_in_circle]))
print(f"\nAll-in-circle (backup):  sample={s_vol_all:.1f}  ref={r_vol_all:.1f}  "
      f"inj={(r_vol_all-s_vol_all)*1.25e-4:+.4f}")

# min-d per cell
s_mask = np.zeros(len(verts), dtype=bool); s_mask[s_patch_idx] = True
r2_mask = np.zeros(len(r2_verts), dtype=bool); r2_mask[r2_patch_idx] = True
s_vol_cell = float(np.sum(s_dists[s_mask] * sa[s_mask]))
r_vol_cell = float(np.sum(r_dists[r2_mask] * ra[r2_mask]))
print(f"Min-d per cell (current):  sample={s_vol_cell:.1f}  ref={r_vol_cell:.1f}  "
      f"inj={(r_vol_cell-s_vol_cell)*1.25e-4:+.4f}")

# min-d per cell + d-threshold at 1.5*d_bar
thresh = 1.5 * d_bar_spike
s_mask2 = s_mask & (s_dists < thresh)
r2_mask2 = r2_mask & (r_dists < thresh)
s_vol2 = float(np.sum(s_dists[s_mask2] * sa[s_mask2]))
r_vol2 = float(np.sum(r_dists[r2_mask2] * ra[r2_mask2]))
print(f"Min-d + d<1.5*d_bar ({thresh:.1f}):  sample={s_vol2:.1f}  ref={r_vol2:.1f}  "
      f"inj={(r_vol2-s_vol2)*1.25e-4:+.4f}  "
      f"  s_verts={s_mask2.sum()}  r_verts={r2_mask2.sum()}")
