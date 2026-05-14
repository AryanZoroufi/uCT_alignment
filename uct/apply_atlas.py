"""
Apply a pre-built atlas to a new scan using a 3-stage registration:
  1. Coarse: PCA-based principal-axes alignment (8 sign configs incl. mirror)
  2. Mid:    similarity ICP refinement (rigid + uniform scale)
  3. Fine:   SimpleITK affine registration (12 DOF: handles non-uniform scale,
             shear, individual shape variation between subjects)

Then transfer atlas labels onto the new scan via the composed transform.
"""
import os
import numpy as np
import trimesh, pymeshfix
import SimpleITK as sitk
from scipy.spatial import cKDTree
from scipy.ndimage import (
    binary_dilation, binary_erosion, binary_fill_holes, label,
    distance_transform_edt,
)
from skimage import measure

# =============== USER PARAMS ===============
INPUT_PATH = '/mnt/user-data/uploads/CT_20251229_163017_aligned.stl'
ATLAS_PATH = '/home/claude/atlas.npz'
OUT_DIR    = '/home/claude/atlas_segmented'
ICP_MAX_ITERS = 50
ICP_SUBSAMPLE = 8000
AFFINE_ITERS = 100      # SimpleITK affine iterations
USE_BSPLINE = False     # add a B-spline non-rigid refinement after affine
                        # (slow: ~5-10x affine time; only enable if affine isn't enough)
BSPLINE_GRID_SPACING_MM = 40.0  # control point spacing
BSPLINE_ITERS = 30
MIN_BONE_VOX = 100      # don't drop tiny bones
MIN_COMPONENT_VERTS = 200

# Neck rule: for each pair of adjacent bones, the entire neck region
# (where the larger bone narrows toward the smaller one) gets reassigned
# to the smaller bone.
NECK_AREA_FRAC = 0.75      # cut where bigger bone's cross-section drops to this
                           # fraction of its peak. Higher = more goes to small bone.
NECK_TUBE_RADIUS_MM = 8.0  # only reassign voxels within this distance of the
                           # actual contact surface (prevents far-away chunks
                           # from being claimed when bones are curved/bent)
# ===========================================

os.makedirs(OUT_DIR, exist_ok=True)
struct = np.ones((3, 3, 3), dtype=bool)

# ---------- Helpers: numpy <-> SimpleITK conversion ----------
def np_to_sitk(arr_xyz, origin_xyz, spacing):
    """Convert numpy array indexed [x,y,z] to SimpleITK image."""
    arr_zyx = np.ascontiguousarray(np.transpose(arr_xyz, (2, 1, 0)))
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetOrigin(tuple(map(float, origin_xyz)))
    img.SetSpacing((float(spacing), float(spacing), float(spacing)))
    return img

def sitk_to_np(img):
    """Convert SimpleITK image to numpy array indexed [x,y,z]."""
    arr_zyx = sitk.GetArrayFromImage(img)
    return np.ascontiguousarray(np.transpose(arr_zyx, (2, 1, 0)))

# ---------- Load atlas ----------
atlas = np.load(ATLAS_PATH, allow_pickle=True)
A_lbl = atlas['bone_labels']
A_solid = atlas['solid']
A_origin = atlas['origin']
A_pitch = float(atlas['pitch'])
A_surf = atlas['surface_points']
A_nbones = int(atlas['n_bones'])
print(f'atlas: shape {A_lbl.shape}, pitch {A_pitch}, max bone id {A_nbones}, '
      f'{len(A_surf):,} surface pts')

# ---------- Voxelize new scan (per-component pymeshfix repair) ----------
mesh_raw = trimesh.load(INPUT_PATH)
print(f'new scan: {len(mesh_raw.vertices):,}v, '
      f'watertight={mesh_raw.is_watertight}, repairing...')
comps = mesh_raw.split(only_watertight=False)
big_comps = [c for c in comps if len(c.vertices) > MIN_COMPONENT_VERTS]
print(f'  {len(comps)} mesh components, {len(big_comps)} substantial')
fixed = []
for c in big_comps:
    try:
        mf = pymeshfix.MeshFix(c.vertices, c.faces)
        mf.repair(joincomp=True, remove_smallest_components=True)
        cm = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=True)
        if cm.volume < 0: cm.invert()
        fixed.append(cm)
    except Exception:
        pass
print(f'  repaired {len(fixed)} components')

all_b = np.array([fm.bounds for fm in fixed])
b_lo = all_b[:, 0, :].min(axis=0) - 4 * A_pitch
b_hi = all_b[:, 1, :].max(axis=0) + 4 * A_pitch
N_shape = np.ceil((b_hi - b_lo) / A_pitch).astype(int)
N_origin = b_lo.astype(np.float32)
N_solid = np.zeros(N_shape, dtype=bool)
for fm in fixed:
    vox = fm.voxelized(pitch=A_pitch).fill()
    bm = vox.matrix.astype(bool)
    bone_origin = vox.translation
    offset = np.round((bone_origin - N_origin) / A_pitch).astype(int)
    sx, sy, sz = bm.shape
    ox, oy, oz = offset
    x0, x1 = max(ox, 0), min(ox + sx, N_shape[0])
    y0, y1 = max(oy, 0), min(oy + sy, N_shape[1])
    z0, z1 = max(oz, 0), min(oz + sz, N_shape[2])
    sub = bm[x0-ox:x1-ox, y0-oy:y1-oy, z0-oz:z1-oz]
    N_solid[x0:x1, y0:y1, z0:z1] |= sub
print(f'  new scan grid: {N_shape}, solid voxels: {N_solid.sum():,}')

N_boundary = N_solid & ~binary_erosion(N_solid, structure=struct)
N_surf = np.argwhere(N_boundary).astype(np.float32) * A_pitch + N_origin

# ============================================================
# STAGE 1: PCA-based coarse alignment
# ============================================================
print('\n=== STAGE 1: PCA coarse alignment ===')
def pca_basis(pts):
    c = pts.mean(0)
    z = pts - c
    cov = (z.T @ z) / len(z)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return c, eigvecs[:, np.argsort(eigvals)[::-1]]

A_c, A_R = pca_basis(A_surf)
N_c, N_R = pca_basis(N_surf)
rng = np.random.default_rng(0)
A_sub_idx = rng.choice(len(A_surf), min(len(A_surf), ICP_SUBSAMPLE), replace=False)
N_sub_idx = rng.choice(len(N_surf), min(len(N_surf), ICP_SUBSAMPLE), replace=False)
A_sub = A_surf[A_sub_idx]
N_sub = N_surf[N_sub_idx]
N_tree = cKDTree(N_sub)

best = (np.inf, None, None, False)
for mirror_flag in [False, True]:
    A_pts_try = A_sub.copy()
    if mirror_flag:
        A_pts_try[:, 0] = 2 * A_c[0] - A_pts_try[:, 0]
    A_c_m, A_R_m = pca_basis(A_pts_try)
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            R_try = A_R_m.copy()
            R_try[:, 1] *= s1; R_try[:, 2] *= s2
            if np.linalg.det(R_try) < 0:
                R_try[:, 0] *= -1
            R = N_R @ R_try.T
            t = N_c - R @ A_c_m
            warped = (R @ A_pts_try.T).T + t
            d, _ = N_tree.query(warped)
            score = float(d.mean())
            if score < best[0]:
                best = (score, R, t, mirror_flag)
score0, R0, t0, mirror_used = best
print(f'  best: {score0:.2f}mm ({"MIRROR" if mirror_used else "normal"} atlas)')

# Apply mirror to atlas data if needed
if mirror_used:
    A_sub_for_icp = A_sub.copy()
    A_sub_for_icp[:, 0] = 2 * A_c[0] - A_sub_for_icp[:, 0]
else:
    A_sub_for_icp = A_sub

# ============================================================
# STAGE 2: similarity ICP (rigid + uniform scale)
# ============================================================
print('\n=== STAGE 2: similarity ICP (rigid + scale) ===')
def icp_refine(src, tgt, R, t, s=1.0, max_iters=50, tol=1e-3):
    tgt_tree = cKDTree(tgt)
    prev_err = np.inf
    for it in range(max_iters):
        warped = s * (R @ src.T).T + t
        d, idx = tgt_tree.query(warped)
        nn = tgt[idx]
        src_c = warped.mean(0); nn_c = nn.mean(0)
        Xc = warped - src_c; Yc = nn - nn_c
        H = Xc.T @ Yc
        U, S_, Vt = np.linalg.svd(H)
        D = np.eye(3); D[2,2] = np.sign(np.linalg.det(Vt.T @ U.T))
        dR = Vt.T @ D @ U.T
        var_x = (Xc ** 2).sum() / len(Xc)
        ds = (S_ * np.diag(D)).sum() / (var_x * len(Xc)) if var_x > 0 else 1.0
        dt = nn_c - ds * (dR @ src_c)
        R = dR @ R; t = ds * (dR @ t) + dt; s = ds * s
        err = float(d.mean())
        if abs(prev_err - err) < tol: break
        prev_err = err
    return R, t, s, err

R1, t1, s1, icp_err = icp_refine(A_sub_for_icp, N_sub, R0, t0, max_iters=ICP_MAX_ITERS)
print(f'  similarity ICP: {icp_err:.2f}mm, scale={s1:.3f}')

# ============================================================
# STAGE 3: SimpleITK affine refinement
# ============================================================
print('\n=== STAGE 3: SimpleITK affine refinement ===')

# Convert solids to SITK images. If mirror was used, mirror the atlas data too.
A_solid_for_sitk = A_solid
A_lbl_for_sitk = A_lbl
A_origin_for_sitk = A_origin.copy()
if mirror_used:
    # Flip atlas along x-axis (in voxel space + adjust origin)
    A_solid_for_sitk = A_solid[::-1, :, :].copy()
    A_lbl_for_sitk = A_lbl[::-1, :, :].copy()
    # New origin x: old extent in x, then shift to keep center fixed
    # Original x range: [origin_x, origin_x + shape_x*pitch]
    # After flip, x_voxel=0 corresponds to old x_voxel=shape_x-1
    # In world coords, that's origin_x + (shape_x-1)*pitch
    # So we want new origin to be such that new world coords match the mirror
    # Mirror axis is x = A_c[0] (old centroid). After flip:
    #   new_world_x = 2*A_c[0] - old_world_x
    # Voxel i in mirrored array corresponds to (shape-1-i) in original
    # original world x = origin_x + (shape-1-i)*pitch
    # mirrored world x = 2*A_c[0] - (origin_x + (shape-1-i)*pitch)
    # We want mirrored array indexing: voxel i corresponds to that mirrored world x
    # i.e., world(i) = new_origin_x + i*pitch
    # so new_origin_x = 2*A_c[0] - origin_x - (shape-1)*pitch
    A_origin_for_sitk[0] = (2 * A_c[0] - A_origin[0]
                            - (A_solid.shape[0] - 1) * A_pitch)

# Build signed distance maps for stable optimization
def signed_dt(solid_xyz, origin, pitch):
    img = np_to_sitk(solid_xyz.astype(np.uint8), origin, pitch)
    dt_filter = sitk.SignedMaurerDistanceMapImageFilter()
    dt_filter.SetSquaredDistance(False)
    dt_filter.SetUseImageSpacing(True)
    return sitk.Cast(dt_filter.Execute(img), sitk.sitkFloat32)

print('  computing signed distance maps...')
A_dt = signed_dt(A_solid_for_sitk, A_origin_for_sitk, A_pitch)
N_dt = signed_dt(N_solid, N_origin, A_pitch)

# Initial affine = inverse of (s1*R1, t1) because SITK transforms map fixed->moving
# i.e. new (fixed) -> atlas (moving)
M_fwd = s1 * R1                  # atlas -> new
M_inv = np.linalg.inv(M_fwd)     # new -> atlas
t_inv = -M_inv @ t1

initial = sitk.AffineTransform(3)
initial.SetMatrix(M_inv.flatten().tolist())
initial.SetTranslation(t_inv.tolist())
# Don't set Center; default (0,0,0) makes T(p) = M_inv*p + t_inv exactly

reg = sitk.ImageRegistrationMethod()
reg.SetMetricAsMeanSquares()
reg.SetMetricSamplingStrategy(reg.RANDOM)
reg.SetMetricSamplingPercentage(0.05, seed=42)  # smaller sample = faster
reg.SetInterpolator(sitk.sitkLinear)
reg.SetOptimizerAsRegularStepGradientDescent(
    learningRate=4.0, minStep=0.01, numberOfIterations=AFFINE_ITERS,
    gradientMagnitudeTolerance=1e-5,
)
reg.SetOptimizerScalesFromPhysicalShift()
reg.SetInitialTransform(initial, inPlace=False)
# Multi-resolution: start coarse, refine
reg.SetShrinkFactorsPerLevel([8, 4, 2])
reg.SetSmoothingSigmasPerLevel([4, 2, 1])
reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

print('  running affine registration...')
final_affine = reg.Execute(N_dt, A_dt)
print(f'  affine final metric: {reg.GetMetricValue():.4f}')
print(f'  iterations: {reg.GetOptimizerIteration()}')

# ============================================================
# STAGE 4 (optional): B-spline non-rigid refinement
# ============================================================
if USE_BSPLINE:
    print('\n=== STAGE 4: B-spline non-rigid refinement ===')
    # Build a B-spline transform with control point grid based on physical extent.
    img_size = N_dt.GetSize()
    img_spacing = N_dt.GetSpacing()
    physical_dims = [s * sp for s, sp in zip(img_size, img_spacing)]
    grid_size = [int(round(d / BSPLINE_GRID_SPACING_MM)) for d in physical_dims]
    grid_size = [max(g, 4) for g in grid_size]  # at least 4 control points per axis
    print(f'  B-spline grid: {grid_size} control points')

    bspline = sitk.BSplineTransformInitializer(
        image1=N_dt, transformDomainMeshSize=grid_size, order=3
    )
    # Compose with the affine result so the B-spline only adds local deformation
    composite = sitk.CompositeTransform([final_affine, bspline])

    reg2 = sitk.ImageRegistrationMethod()
    reg2.SetMetricAsMeanSquares()
    reg2.SetMetricSamplingStrategy(reg2.RANDOM)
    reg2.SetMetricSamplingPercentage(0.05, seed=42)
    reg2.SetInterpolator(sitk.sitkLinear)
    reg2.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=BSPLINE_ITERS,
        maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000,
    )
    reg2.SetInitialTransform(bspline, inPlace=True)
    reg2.SetMovingInitialTransform(final_affine)
    reg2.SetShrinkFactorsPerLevel([4, 2, 1])
    reg2.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    print('  running B-spline registration...')
    bspline_result = reg2.Execute(N_dt, A_dt)
    print(f'  B-spline final metric: {reg2.GetMetricValue():.4f}')
    final_transform = sitk.CompositeTransform([final_affine, bspline_result])
else:
    final_transform = final_affine

# ============================================================
# Transfer labels: resample atlas labels onto new-scan grid
# ============================================================
print('\n=== Label transfer ===')
A_lbl_sitk = np_to_sitk(A_lbl_for_sitk.astype(np.uint8), A_origin_for_sitk, A_pitch)
N_solid_sitk = np_to_sitk(N_solid.astype(np.uint8), N_origin, A_pitch)
resampled = sitk.Resample(
    A_lbl_sitk, N_solid_sitk,
    transform=final_transform,
    interpolator=sitk.sitkNearestNeighbor,
    defaultPixelValue=0,
)
N_lbl = sitk_to_np(resampled).astype(np.uint8)
N_lbl[~N_solid] = 0
unlabeled = N_solid & (N_lbl == 0)
print(f'  unlabeled voxels: {unlabeled.sum():,} ({unlabeled.sum()/N_solid.sum()*100:.1f}%)')

# Refine: fill unlabeled by nearest labeled neighbor
if unlabeled.any():
    labeled_mask = N_solid & (N_lbl > 0)
    _, indices = distance_transform_edt(~labeled_mask, return_indices=True)
    N_lbl[unlabeled] = N_lbl[indices[0][unlabeled],
                              indices[1][unlabeled],
                              indices[2][unlabeled]]

# Keep largest CC per bone
for lid in range(1, A_nbones + 1):
    mask = (N_lbl == lid)
    if not mask.any(): continue
    cc, n = label(mask, structure=struct)
    if n <= 1: continue
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    main = int(np.argmax(sizes))
    drop = mask & (cc != main)
    N_lbl[drop] = 0

# Re-fill orphans
unlabeled2 = N_solid & (N_lbl == 0)
if unlabeled2.any():
    labeled_mask = N_solid & (N_lbl > 0)
    _, indices = distance_transform_edt(~labeled_mask, return_indices=True)
    N_lbl[unlabeled2] = N_lbl[indices[0][unlabeled2],
                               indices[1][unlabeled2],
                               indices[2][unlabeled2]]

# ============================================================
# Neck rule: for each pair of adjacent bones, give the entire neck
# (anything past where the larger bone narrows) to the smaller bone.
# ============================================================
print('\n=== Neck rule: full neck -> smaller bone ===')
def is_pair_connected_in_solid(mask_a, mask_b):
    """Are these two bone regions actually touching (sharing a voxel boundary)?"""
    return (mask_a & binary_dilation(mask_b, structure=struct)).any()

# Build adjacency list
present = sorted([int(l) for l in np.unique(N_lbl) if l > 0])
sizes_now = {l: int((N_lbl == l).sum()) for l in present}
pairs_seen = set()
adjacency = []
for la in present:
    ma = (N_lbl == la)
    nbrs = N_lbl[binary_dilation(ma, structure=struct) & ~ma & N_solid]
    for lb in np.unique(nbrs):
        if lb == 0 or lb == la: continue
        key = tuple(sorted([int(la), int(lb)]))
        if key in pairs_seen: continue
        pairs_seen.add(key)
        adjacency.append(key)
print(f'  {len(adjacency)} adjacent bone pairs')

for la, lb in adjacency:
    sa, sb = sizes_now[la], sizes_now[lb]
    small, big = (la, lb) if sa < sb else (lb, la)
    s_small, s_big = (sa, sb) if sa < sb else (sb, sa)
    mask_small = (N_lbl == small)
    mask_big = (N_lbl == big)
    if not is_pair_connected_in_solid(mask_small, mask_big):
        continue

    # Find the long axis of the bigger bone
    big_pts = np.argwhere(mask_big).astype(np.float64)
    if len(big_pts) < 100:
        continue
    big_centered = big_pts - big_pts.mean(0)
    cov = np.cov(big_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, np.argmax(eigvals)]

    # Project both bones onto this axis
    grid_idx = np.indices(N_lbl.shape).astype(np.float64).reshape(3, -1).T
    proj_full = (grid_idx @ long_axis).reshape(N_lbl.shape)
    proj_big = proj_full[mask_big]
    proj_small = proj_full[mask_small]

    # Cross-section of BIG bone along its long axis
    bin_count = max(int(proj_big.max() - proj_big.min()), 30)
    bins = np.linspace(proj_big.min(), proj_big.max(), bin_count + 1)
    area, _ = np.histogram(proj_big, bins=bins)
    area = area.astype(float) * A_pitch ** 2
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # Smooth
    k = max(3, bin_count // 15)
    area_s = np.convolve(area, np.ones(k)/k, mode='same')

    # Find the BIG bone's peak cross-section
    peak_idx = int(np.argmax(area_s))
    peak_area = area_s[peak_idx]
    threshold = peak_area * NECK_AREA_FRAC

    # Direction from big to small along the axis
    dir_to_small = 1 if proj_small.mean() > proj_big.mean() else -1

    # Walk from peak toward the small bone; cut at first bin where area < threshold
    cut_idx = peak_idx
    if dir_to_small > 0:
        rng = range(peak_idx, len(area_s))
    else:
        rng = range(peak_idx, -1, -1)
    for i in rng:
        if area_s[i] < threshold:
            cut_idx = i
            break
    cut_proj = bin_centers[cut_idx]

    # Plane mask: voxels of BIG bone past the cut, in direction toward small bone
    if dir_to_small > 0:
        plane_mask = mask_big & (proj_full > cut_proj)
    else:
        plane_mask = mask_big & (proj_full < cut_proj)

    # Tube constraint: only voxels within NECK_TUBE_RADIUS_MM of the
    # boundary surface between the two bones. This prevents the plane from
    # catching far-away chunks when the bone is bent.
    boundary = mask_big & binary_dilation(mask_small, structure=struct)
    if not boundary.any():
        continue
    dist_from_boundary = distance_transform_edt(~boundary) * A_pitch
    in_tube = dist_from_boundary < NECK_TUBE_RADIUS_MM
    neck_mask = plane_mask & in_tube

    n_moved = int(neck_mask.sum())
    if n_moved == 0:
        continue
    N_lbl[neck_mask] = small
    sizes_now[small] = int((N_lbl == small).sum())
    sizes_now[big]   = int((N_lbl == big).sum())
    print(f'  bones {small}<-{big}: peak area {peak_area:.0f}mm², '
          f'cut at {area_s[cut_idx]/peak_area*100:.0f}% of peak; '
          f'moved {n_moved:,} neck vox (tube {NECK_TUBE_RADIUS_MM}mm)')

# ============================================================
# Stranded-fragment cleanup: orphan pieces of each bone that ended up
# inside another bone's interior get reassigned by majority-neighbor.
# This fixes the case where porous trabecular bone of bone A leaks
# through holes and gets relabeled as bone B during nearest-neighbor refill.
# ============================================================
print('\n=== Stranded-fragment cleanup ===')
present = sorted([int(l) for l in np.unique(N_lbl) if l > 0])
total_reassigned = 0
for lid in present:
    mask = (N_lbl == lid)
    cc, n_cc = label(mask, structure=struct)
    if n_cc <= 1:
        continue
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    main_id = int(np.argmax(sizes))
    main_size = sizes[main_id]
    # For each NON-main fragment, find which OTHER bone label is its
    # majority neighbour (counted over the fragment's voxel-face neighbours
    # of any other label). Reassign the fragment to that label.
    for frag_id in range(1, n_cc + 1):
        if frag_id == main_id: continue
        frag_mask = (cc == frag_id)
        frag_size = int(frag_mask.sum())
        # Don't move very large fragments — they might be a real disconnected
        # bone region (e.g. if registration genuinely split a bone in two).
        if frag_size > main_size * 0.30:
            continue
        # Get neighbour labels around the fragment
        nbrs = N_lbl[binary_dilation(frag_mask, structure=struct) & ~frag_mask]
        nbrs = nbrs[(nbrs != 0) & (nbrs != lid)]
        if len(nbrs) == 0:
            continue
        # Majority neighbour label
        vals, counts = np.unique(nbrs, return_counts=True)
        new_lid = int(vals[np.argmax(counts)])
        N_lbl[frag_mask] = new_lid
        total_reassigned += frag_size
        print(f'  bone {lid}: {frag_size:,}-vox fragment -> bone {new_lid} '
              f'(majority of {len(nbrs):,} face neighbours)')
print(f'  total reassigned: {total_reassigned:,} voxels')

# ============================================================
# Intrusion cleanup: thin protrusions of one bone that stick into
# another bone's interior (e.g. through trabecular cavities) get
# reassigned to whichever bone surrounds them most.
# Method: morphological opening on each bone removes thin appendages;
# the removed voxels go to their majority neighbour.
# ============================================================
print('\n=== Intrusion cleanup ===')
INTRUSION_OPEN_VOX = 2  # erosion radius — anything thinner than ~2x this in any
                        # direction is treated as an intrusion. 2 voxels = 1mm at
                        # 0.5mm pitch — small enough to keep real bone features.
INTRUSION_MAX_FRAG_VOX = 5000  # don't reassign fragments bigger than this
                                # (they're more likely real bone parts)

intrusion_total = 0
for lid in present:
    mask = (N_lbl == lid)
    if mask.sum() < 200: continue
    eroded = binary_erosion(mask, structure=struct, iterations=INTRUSION_OPEN_VOX)
    if not eroded.any():
        continue
    kept = binary_dilation(eroded, structure=struct,
                           iterations=INTRUSION_OPEN_VOX) & mask
    intrusion = mask & ~kept
    if not intrusion.any(): continue

    # Connected components
    cc, n_cc = label(intrusion, structure=struct)
    sizes = np.bincount(cc.ravel()); sizes[0] = 0

    # For each fragment, look at its own voxels and their face-neighbours
    # (precompute one shifted-label array to find neighbour labels efficiently)
    moved = 0
    # Precompute: for each intrusion voxel, the labels of its 6 face neighbours
    # (we use binary_dilation once at the cc level, not per fragment)
    cc_dilated = binary_dilation(intrusion, structure=struct) & ~intrusion
    # Voxels in cc_dilated have label N_lbl[that voxel]; we need to associate each
    # with its closest fragment id. Use distance_transform with indices.
    # Simpler: iterate fragments but only over the small bbox of each.
    for fid in range(1, n_cc + 1):
        frag_size = int(sizes[fid])
        if frag_size > INTRUSION_MAX_FRAG_VOX: continue
        # bbox of this fragment to limit work
        idx = np.argwhere(cc == fid)
        lo = idx.min(0); hi = idx.max(0) + 1
        # expand by 1 for face neighbours
        lo = np.maximum(lo - 1, 0)
        hi = np.minimum(hi + 1, np.array(cc.shape))
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        frag_local = (cc[sl] == fid)
        nbr_local = binary_dilation(frag_local, structure=struct) & ~frag_local
        nbr_labels = N_lbl[sl][nbr_local]
        n_other = int(((nbr_labels != 0) & (nbr_labels != lid)).sum())
        n_self  = int((nbr_labels == lid).sum())
        if n_other < 2 * n_self: continue
        non_self = nbr_labels[(nbr_labels != 0) & (nbr_labels != lid)]
        if len(non_self) == 0: continue
        vals, counts = np.unique(non_self, return_counts=True)
        new_lid = int(vals[np.argmax(counts)])
        N_lbl[sl][frag_local] = new_lid
        moved += frag_size
    if moved > 0:
        print(f'  bone {lid}: removed {moved:,} intrusion voxels')
        intrusion_total += moved
print(f'  total intrusion voxels reassigned: {intrusion_total:,}')

# ============================================================
# Output STLs
# ============================================================
print('\n=== Writing STLs ===')
out_files = []
for lid in range(1, A_nbones + 1):
    mask = (N_lbl == lid)
    if mask.sum() < MIN_BONE_VOX:
        if mask.any():
            print(f'  bone {lid}: {mask.sum()} voxels - SKIP (below threshold)')
        continue
    # Smooth: opening then closing to remove protrusions and fill cavities
    mask = binary_erosion(mask, structure=struct, iterations=1)
    mask = binary_dilation(mask, structure=struct, iterations=1)
    if mask.sum() == 0: continue
    # Keep largest connected component (in case opening disconnected things)
    cc, n = label(mask, structure=struct)
    if n > 1:
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        mask = (cc == int(np.argmax(sizes)))
    mask = binary_fill_holes(mask)
    padded = np.pad(mask, 1, constant_values=False).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = (verts - 1) * A_pitch + N_origin
    bm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if bm.volume < 0: bm.invert()
    fn = f'{OUT_DIR}/bone_{lid:02d}.stl'
    bm.export(fn)
    out_files.append(fn)
    print(f'  bone {lid:>2}: {len(bm.vertices):>6,}v {len(bm.faces):>6,}f '
          f'vol={bm.volume:>7.0f}mm³ watertight={bm.is_watertight}')
print(f'\nwrote {len(out_files)} bones to {OUT_DIR}/')
