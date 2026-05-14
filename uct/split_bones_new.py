"""
Split a 2-bone mesh, with bridge/contact zone assigned to 'top' bone.

Strategy:
  Method A (thin bridge, e.g. sample_1):
    - Iteratively erode until 2 substantial cores appear
    - Re-grow only the bottom seed; top gets the rest
  Method B (thick contact, e.g. sample_part_1):
    - Find long axis via PCA on solid voxels
    - Compute area along axis; find deepest valley between two peaks
    - Use largest CC on each side of the valley as seeds
    - Assign every voxel to nearest seed by distance transform
    - Erode bottom by k mm to push contact zone toward top

Pipeline tries A first; falls back to B if A doesn't find a clean split.
"""
import numpy as np
import trimesh, pymeshfix
from scipy.ndimage import (
    binary_erosion, binary_dilation, binary_fill_holes,
    label, distance_transform_edt,
)
from skimage import measure

# ============== USER PARAMS ==============
INPUT_PATH = '/mnt/user-data/uploads/sample_part_1.stl'
OUT_TOP    = '/home/claude/s2_top.stl'
OUT_BOT    = '/home/claude/s2_bot.stl'
PITCH      = 0.4         # voxel size, mm
TOP_AXIS   = 0           # 0=x, 1=y, 2=z (axis used to choose top vs bottom)
TOP_HIGHER = True        # True: 'top' is the bone whose centroid is higher on TOP_AXIS
MIN_BONE_VOX = 5000      # required size of a 'real' bone core (Method A)
BRIDGE_BIAS_MM = 1.0     # extra dilation of top-region after labeling (mm)
NECK_AREA_FRAC = 0.75    # fraction of shaft peak area where the cut goes
NECK_TUBE_RADIUS_MM = 8.0  # Method B: only reassign neck voxels within this 3D
                           # distance of the watershed boundary surface; this
                           # prevents the plane-cut from catching far-away chunks
                           # of shaft body when the bone is curved/bent.
APPENDAGE_PRUNE_MM = 1.0   # Method B: remove TOP-region "fingers" thinner than
                           # ~2x this value. Done by morphological opening +
                           # main-component-only filter.
                         # this fraction of its peak. Lower = more conservative
                         # (less material reassigned to top); higher = more.
# =========================================

# ---------- 0. load + repair ----------
raw = trimesh.load(INPUT_PATH)
print(f'raw: {len(raw.vertices)} v, {len(raw.faces)} f, watertight={raw.is_watertight}')
mf = pymeshfix.MeshFix(raw.vertices, raw.faces)
mf.repair(joincomp=True, remove_smallest_components=True)
mesh = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=True)
if mesh.volume < 0:
    mesh.invert()
print(f'repaired: {len(mesh.vertices)} v, {len(mesh.faces)} f, vol={mesh.volume:.0f}')

# ---------- 1. voxelize ----------
vox = mesh.voxelized(pitch=PITCH).fill()
origin = vox.translation
solid = binary_fill_holes(vox.matrix.astype(bool))
print(f'grid {solid.shape}, solid {solid.sum():,} voxels')

struct = np.ones((3, 3, 3), dtype=bool)

# ---------- 2. METHOD A: erosion-thinning ----------
def method_A(solid):
    eroded = solid.copy()
    N = 0
    while eroded.any():
        new = binary_erosion(eroded, structure=struct)
        N += 1
        lbl, n = label(new, structure=struct)
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        big_ids = [i for i in range(len(sizes)) if sizes[i] >= MIN_BONE_VOX]
        if len(big_ids) >= 2:
            order = sorted(big_ids, key=lambda i: -sizes[i])
            seed_a = (lbl == order[0])
            seed_b = (lbl == order[1])
            return ('A', N, seed_a, seed_b)
        eroded = new
    return None

# ---------- 3. METHOD B: long-axis cross-section cut ----------
def method_B(solid):
    # 3a. long axis via PCA on solid voxel positions
    pts = np.argwhere(solid).astype(np.float64)   # in voxel units
    pts -= pts.mean(0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, np.argmax(eigvals)]    # principal direction
    print(f'  long axis (voxel): {long_axis.round(3)}')

    # 3b. project all solid voxels onto long axis -> 1D coordinate
    proj = (np.argwhere(solid).astype(np.float64) @ long_axis)
    proj_min, proj_max = proj.min(), proj.max()
    n_bins = max(int((proj_max - proj_min)) , 30)
    bins = np.linspace(proj_min, proj_max, n_bins + 1)
    area = np.histogram(proj, bins=bins)[0].astype(float) * PITCH**2
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # smooth
    k = 5
    kernel = np.ones(k) / k
    area_s = np.convolve(area, kernel, mode='same')

    # 3c. find two highest peaks; deepest valley between them
    # simple peak-finding: local max in smoothed area, exclude tails
    deriv = np.diff(area_s)
    # candidates: indices where deriv changes from + to -
    peaks = np.where((deriv[:-1] > 0) & (deriv[1:] <= 0))[0] + 1
    # require minimum prominence (relative to global max)
    peaks = peaks[area_s[peaks] > area_s.max() * 0.4]
    if len(peaks) < 2:
        # fallback: argmin in the middle 60% of the bin range
        ks = int(n_bins * 0.2)
        v = ks + int(np.argmin(area_s[ks:n_bins-ks]))
        print(f'  no two clear peaks, cutting at middle minimum (bin {v})')
    else:
        order = peaks[np.argsort(-area_s[peaks])]
        p1, p2 = sorted([order[0], order[1]])
        v = p1 + int(np.argmin(area_s[p1:p2+1]))
        print(f'  peaks at bins {p1},{p2} (proj {bin_centers[p1]:.1f}, {bin_centers[p2]:.1f}); valley at bin {v} (proj {bin_centers[v]:.1f}), area {area_s[v]:.0f}mm²')

    cut_proj = bin_centers[v]

    # 3d. binary mask: which voxels are on each side of the cut?
    grid_idx = np.indices(solid.shape).astype(np.float64).reshape(3, -1).T
    proj_full = (grid_idx @ long_axis).reshape(solid.shape)
    side_left = solid & (proj_full < cut_proj)
    side_right = solid & (proj_full >= cut_proj)

    # 3e. take largest CC on each side as seed
    def largest_cc(mask):
        lbl, _ = label(mask, structure=struct)
        if lbl.max() == 0:
            return mask
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        return lbl == int(np.argmax(sizes))

    seed_a = largest_cc(side_left)
    seed_b = largest_cc(side_right)
    print(f'  seed sizes: left {seed_a.sum():,}, right {seed_b.sum():,}')
    return ('B', cut_proj, seed_a, seed_b, long_axis, area_s, bin_centers, p1, p2)

# ---------- 4. run A, fallback to B ----------
result = method_A(solid)
if result is None:
    print('Method A failed -> falling back to Method B (cross-section cut)')
    result = method_B(solid)

method = result[0]
seed_a = result[2]
seed_b = result[3]

# ---------- 5. choose top vs bottom ----------
def cw(mask):
    return np.argwhere(mask).mean(axis=0) * PITCH + origin
ca, cb = cw(seed_a), cw(seed_b)
a_is_top = (ca[TOP_AXIS] > cb[TOP_AXIS]) if TOP_HIGHER else (ca[TOP_AXIS] < cb[TOP_AXIS])
top_seed, bot_seed = (seed_a, seed_b) if a_is_top else (seed_b, seed_a)
print(f'TOP centroid    {cw(top_seed)}, seed size {top_seed.sum():,}')
print(f'BOTTOM centroid {cw(bot_seed)}, seed size {bot_seed.sum():,}')

# ---------- 6. assign all solid voxels to top or bottom ----------
if method == 'A':
    # Re-grow bottom seed by N steps inside solid; rest is top.
    N = result[1]
    bot_region = bot_seed.copy()
    for _ in range(N):
        bot_region = binary_dilation(bot_region, structure=struct) & solid
    top_region = solid & ~bot_region
else:
    # Watershed on the inverted distance transform of the solid mask.
    from skimage.segmentation import watershed
    distance_inside = distance_transform_edt(solid)
    markers = np.zeros(solid.shape, dtype=np.int32)
    markers[top_seed] = 1
    markers[bot_seed] = 2
    ws_labels = watershed(-distance_inside, markers=markers, mask=solid)
    top_region = (ws_labels == 1)
    bot_region = (ws_labels == 2)
    print(f'watershed: TOP {top_region.sum():,}, BOT {bot_region.sum():,}')

    # --- Neck-claim: cut at where the shaft cross-section drops to a fraction
    # --- of its peak (the START of the narrow neck region, not the shaft peak)
    long_axis_v = result[4]
    area_s = result[5]
    bin_centers = result[6]
    p1, p2 = result[7], result[8]
    # Identify which peak is the shaft (the larger one in cross-section)
    shaft_peak_idx = p1 if area_s[p1] > area_s[p2] else p2
    compact_peak_idx = p2 if shaft_peak_idx == p1 else p1
    shaft_peak_area = area_s[shaft_peak_idx]
    threshold = shaft_peak_area * NECK_AREA_FRAC
    # Walk from shaft peak toward compact peak; cut at first bin where area < threshold
    step = 1 if compact_peak_idx > shaft_peak_idx else -1
    cut_idx = shaft_peak_idx
    for i in range(shaft_peak_idx, compact_peak_idx, step):
        if area_s[i] < threshold:
            cut_idx = i
            break
    cut_proj_neck = bin_centers[cut_idx]
    print(f'neck cut: shaft peak {shaft_peak_area:.0f}mm² at bin {shaft_peak_idx}; '
          f'cut at bin {cut_idx} (area {area_s[cut_idx]:.0f}mm² = '
          f'{area_s[cut_idx]/shaft_peak_area*100:.0f}% of peak)')

    # Reassign BOT voxels past the neck cut (toward compact) to TOP,
    # BUT only those that are within NECK_TUBE_RADIUS_MM of the watershed
    # boundary surface. This prevents the plane cut from catching distant
    # chunks of shaft body when the bone is curved.
    grid_idx = np.indices(solid.shape).astype(np.float64).reshape(3, -1).T
    proj_full = (grid_idx @ long_axis_v).reshape(solid.shape)
    bot_centroid_proj = proj_full[bot_region].mean()
    top_centroid_proj = proj_full[top_region].mean()

    # Build the watershed boundary surface (BOT voxels touching TOP)
    ws_boundary = bot_region & binary_dilation(top_region, structure=struct)
    dist_from_boundary_mm = distance_transform_edt(~ws_boundary) * PITCH
    in_tube = dist_from_boundary_mm < NECK_TUBE_RADIUS_MM
    print(f'tube radius {NECK_TUBE_RADIUS_MM}mm: {in_tube.sum():,} voxels in tube')

    if top_centroid_proj < bot_centroid_proj:
        plane_mask = solid & (proj_full < cut_proj_neck) & bot_region
    else:
        plane_mask = solid & (proj_full > cut_proj_neck) & bot_region
    neck_mask = plane_mask & in_tube
    rejected = plane_mask & ~in_tube
    print(f'plane cut would have grabbed {plane_mask.sum():,} voxels; '
          f'tube limits to {neck_mask.sum():,} (rejected {rejected.sum():,} far chunks)')
    top_region = top_region | neck_mask
    bot_region = bot_region & ~neck_mask
    print(f'after neck-claim: TOP {top_region.sum():,}, BOT {bot_region.sum():,}')

    k_voxels = max(0, int(round(BRIDGE_BIAS_MM / PITCH)))
    if k_voxels > 0:
        top_region = binary_dilation(top_region, structure=struct, iterations=k_voxels) & solid
        bot_region = solid & ~top_region
        print(f'after {BRIDGE_BIAS_MM}mm bias: TOP {top_region.sum():,}, BOT {bot_region.sum():,}')

    # --- Appendage pruning: remove thin "fingers" of TOP that stick out ---
    # Erode TOP, find connected components, keep only the largest, then dilate
    # back. Anything disconnected by the erosion was a thin appendage.
    prune_iters = max(1, int(round(APPENDAGE_PRUNE_MM / PITCH)))
    if prune_iters > 0:
        top_eroded = binary_erosion(top_region, structure=struct, iterations=prune_iters)
        lbl, n = label(top_eroded, structure=struct)
        if n > 1:
            sizes = np.bincount(lbl.ravel()); sizes[0] = 0
            main_id = int(np.argmax(sizes))
            main_eroded = (lbl == main_id)
            cleaned = binary_dilation(main_eroded, structure=struct,
                                      iterations=prune_iters) & top_region
            removed = top_region & ~cleaned
            top_region = cleaned
            bot_region = bot_region | (removed & solid)
            print(f'pruned {removed.sum():,} appendage voxels '
                  f'({n-1} disconnected fragment(s) at {prune_iters}-vox erosion)')
        else:
            print(f'no appendages found (TOP stayed connected after {prune_iters}-vox erosion)')

print(f'TOP region: {top_region.sum():,}, BOT region: {bot_region.sum():,}')

# ---------- 7. cleanup: keep largest CC of each region ----------
def main_cc(mask):
    lbl, _ = label(mask, structure=struct)
    if lbl.max() == 0: return mask
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    return lbl == int(np.argmax(sizes))
top_main = main_cc(top_region)
bot_main = main_cc(bot_region)
# reassign loose fragments to nearest main
d2t = distance_transform_edt(~top_main)
d2b = distance_transform_edt(~bot_main)
final_bot = solid & (d2b <= d2t)
final_bot |= bot_main
final_top = solid & ~final_bot
final_top |= top_main
final_bot &= ~top_main
print(f'after cleanup: TOP {final_top.sum():,}, BOT {final_bot.sum():,}')

# ---------- 8. marching cubes per region ----------
def label_to_mesh(mask, name):
    padded = np.pad(mask, 1, constant_values=False).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = (verts - 1) * PITCH + origin
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if m.volume < 0: m.invert()
    print(f'{name}: {len(m.vertices):,} v, {len(m.faces):,} f, '
          f'watertight={m.is_watertight}, vol={m.volume:.0f}')
    return m

top_mesh = label_to_mesh(final_top, 'TOP   ')
bot_mesh = label_to_mesh(final_bot, 'BOTTOM')
top_mesh.export(OUT_TOP)
bot_mesh.export(OUT_BOT)
print(f'wrote {OUT_TOP}, {OUT_BOT}')
