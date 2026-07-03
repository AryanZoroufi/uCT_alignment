"""
Split a 2-bone mesh connected by a thin bridge.
Bridge -> top bone (here: top = higher centroid x).
Output: two clean watertight meshes via marching cubes on labeled voxels.
"""
import numpy as np
import trimesh, pymeshfix
from scipy.ndimage import (
    binary_erosion, binary_dilation, binary_fill_holes, label,
)
from skimage import measure

PITCH = 0.4
TOP_AXIS = 0          # 0=x, 1=y, 2=z
TOP_HIGHER = True     # True: top = higher coord; False: top = lower coord
MIN_BONE_VOX = 5000   # a "real bone core" must be at least this many voxels

# ---------- 1. repair + voxelize + fill cavities ----------
raw = trimesh.load('/mnt/user-data/uploads/sample_1.stl')
print(f'raw: {len(raw.vertices)} v, {len(raw.faces)} f, watertight={raw.is_watertight}')

mf = pymeshfix.MeshFix(raw.vertices, raw.faces)
mf.repair(joincomp=True, remove_smallest_components=True)
mesh = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=True)
print(f'repaired: {len(mesh.vertices)} v, {len(mesh.faces)} f, '
      f'watertight={mesh.is_watertight}, vol={mesh.volume:.1f}')

vox = mesh.voxelized(pitch=PITCH).fill()
origin = vox.translation
solid = binary_fill_holes(vox.matrix.astype(bool))
print(f'grid {solid.shape}, solid voxels {solid.sum():,}')

# ---------- 2. erode until two substantial cores exist ----------
struct = np.ones((3, 3, 3), dtype=bool)
eroded = solid.copy()
N = 0
trace = []
while eroded.any():
    new = binary_erosion(eroded, structure=struct)
    N += 1
    lbl, n = label(new, structure=struct)
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    big_ids = [i for i in range(len(sizes)) if sizes[i] >= MIN_BONE_VOX]
    trace.append((N, int(new.sum()), len(big_ids), [int(sizes[i]) for i in big_ids]))
    if len(big_ids) >= 2:
        order = sorted(big_ids, key=lambda i: -sizes[i])
        a_id, b_id = order[0], order[1]
        seed_a = (lbl == a_id); seed_b = (lbl == b_id)
        eroded = new
        break
    eroded = new
else:
    raise RuntimeError("never split into two substantial cores")

for t in trace: print(f'  erode {t[0]:2d}: {t[1]:>7,} vox, {t[2]} big cores, sizes {t[3]}')
print(f'-> bridge breaks at N={N}; cores {sizes[a_id]:,} and {sizes[b_id]:,}')

# ---------- 3. choose top vs bottom ----------
def cw(mask):
    idx = np.argwhere(mask)
    return idx.mean(axis=0) * PITCH + origin
ca, cb = cw(seed_a), cw(seed_b)
a_is_top = (ca[TOP_AXIS] > cb[TOP_AXIS]) if TOP_HIGHER else (ca[TOP_AXIS] < cb[TOP_AXIS])
top_seed, bot_seed = (seed_a, seed_b) if a_is_top else (seed_b, seed_a)
print(f'TOP centroid    {cw(top_seed)} (seed size {top_seed.sum():,})')
print(f'BOTTOM centroid {cw(bot_seed)} (seed size {bot_seed.sum():,})')

# ---------- 4. re-grow bottom only -> bridge stays with top ----------
bot_region = bot_seed.copy()
for _ in range(N):
    bot_region = binary_dilation(bot_region, structure=struct) & solid
top_region = solid & ~bot_region
print(f'before cleanup: TOP {top_region.sum():,}, BOT {bot_region.sum():,} voxels')

# ---------- 4b. reassign disconnected fragments to nearest main region ----------
# Find main (largest) component of each region
def main_cc(mask):
    lbl, n = label(mask, structure=struct)
    if n == 0: return mask
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    main_id = int(np.argmax(sizes))
    return lbl == main_id, lbl, sizes

top_main, top_lbl, top_sizes = main_cc(top_region)
bot_main, bot_lbl, bot_sizes = main_cc(bot_region)
print(f'TOP has {(top_sizes > 0).sum()} components (main: {top_sizes.max():,})')
print(f'BOT has {(bot_sizes > 0).sum()} components (main: {bot_sizes.max():,})')

# Distance fields from each main region (in voxel units)
from scipy.ndimage import distance_transform_edt
d_to_top = distance_transform_edt(~top_main)
d_to_bot = distance_transform_edt(~bot_main)

# Reassign every solid voxel by nearest main region — but force any voxel
# already in the bottom MAIN component to stay bottom (preserves bridge-to-top rule)
new_bot = (d_to_bot <= d_to_top) & solid
new_bot |= bot_main      # protect main bottom
new_top = solid & ~new_bot
# Make sure the previously-defined bridge area (top_region minus top_main fragments)
# stays with top: anything in top_main definitely stays top
new_top |= top_main
new_bot &= ~top_main

top_region, bot_region = new_top, new_bot
print(f'after cleanup:  TOP {top_region.sum():,}, BOT {bot_region.sum():,} voxels')

# Final sanity: report component counts
_, _, ts = main_cc(top_region)
_, _, bs = main_cc(bot_region)
print(f'TOP components after cleanup: {(ts > 0).sum()}, sizes (top 5): {sorted(ts[ts>0], reverse=True)[:5]}')
print(f'BOT components after cleanup: {(bs > 0).sum()}, sizes (top 5): {sorted(bs[bs>0], reverse=True)[:5]}')

# ---------- 5. marching cubes per region ----------
def label_to_mesh(mask, name):
    padded = np.pad(mask, 1, constant_values=False).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = (verts - 1) * PITCH + origin
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if m.volume < 0:
        m.invert()
    print(f'{name}: {len(m.vertices):>6,} v, {len(m.faces):>6,} f, '
          f'watertight={m.is_watertight}, vol={m.volume:.1f}')
    return m

top_mesh = label_to_mesh(top_region, 'TOP   ')
bot_mesh = label_to_mesh(bot_region, 'BOTTOM')

top_mesh.export('/home/claude/bone_top.stl')
bot_mesh.export('/home/claude/bone_bottom.stl')
