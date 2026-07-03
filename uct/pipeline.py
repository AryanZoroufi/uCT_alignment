"""
End-to-end pipeline: VOX → mesh → align → atlas-based segmentation → SMC refine.

Steps:
  1. Convert ref.vox and sample.vox to STL  (sigma=3, decimate=0.05, laplacian=100)
  2. Align sample onto ref  (PCA + ICP)
  3. Atlas-based segmentation: registers atlas.npz onto each scan via
     PCA + similarity ICP + SimpleITK affine, then transfers atlas labels
     onto the scan voxel grid.  Produces N bones per scan (atlas.n_bones).
  4. Pick bones of interest via --bones (default: 1 2).
  5. Save:
       <out>/ref_<i>.stl, sample_<i>.stl     ← selected bones (renumbered 1..K)
       <out>/all_segments/ref_part_<atlas_id>.stl
       <out>/all_segments/sample_part_<atlas_id>.stl
  6. Aggregate PCA+ICP pre-alignment (adaptive).
  7. SMC aggregate + per-bone alignment (skipped with --no-smc).
  8. Injection score (protrusion-based articular patch).

Usage:
    python pipeline.py ref.vox sample.vox
    python pipeline.py ref.vox sample.vox -o results/
    python pipeline.py ref.vox sample.vox --bones 1 4
    python pipeline.py ref.vox sample.vox --no-smc
    python pipeline.py ref.vox sample.vox --atlas-npz /path/to/atlas.npz
"""

import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
sys.path.insert(0, str(Path(__file__).parent))
from vox_to_stl    import vox_to_stl
from align_mesh    import align_mesh, align_longest_axis_to_x, aggregate_align_transform
from smc_align     import smc_align_aggregate, smc_align, smc_align_per_bone
from segment_mesh  import split_bone_xslice, split_thin_bridge



# ---------------------------------------------------------------------------
# Segmentation — atlas-based registration & label transfer
# ---------------------------------------------------------------------------

def _segment_via_atlas(
    stl_path: Path,
    atlas_path: Path,
    return_transform: bool = False,
    debug_dir=None,
    registration: str = "icp",
) -> dict[int, trimesh.Trimesh]:
    """
    Atlas-based segmentation (ported from apply_atlas.py).

    Registers a precomputed voxel atlas onto the new scan via:
      1. PCA principal-axes alignment (8 sign configs incl. mirror)
      2. Similarity ICP (rigid + uniform scale, 7 DOF)
      3. SMC importance-sampling refinement (9 DOF, anisotropic similarity)
      4. SimpleITK affine on signed-distance maps (12 DOF — adds shear)
    Then resamples atlas labels onto the new-scan grid and cleans up
    (neck-rule, stranded fragments, intrusion pruning).

    Returns
    -------
    dict mapping atlas bone id (1..N) to surviving Trimesh.
    """
    import SimpleITK as sitk
    import pymeshfix
    import fast_simplification
    from scipy.spatial import cKDTree
    from scipy.ndimage import (
        binary_dilation, binary_erosion, binary_fill_holes,
        label as _label, distance_transform_edt,
    )
    from skimage import measure

    # ---- constants (from apply_atlas.py) ----
    ICP_MAX_ITERS          = 50
    ICP_SUBSAMPLE          = 8000
    AFFINE_ITERS           = 100
    MIN_BONE_VOX           = 100
    MIN_COMPONENT_VERTS    = 200
    NECK_AREA_FRAC         = 0.75
    NECK_TUBE_RADIUS_MM    = 8.0
    INTRUSION_OPEN_VOX     = 2
    INTRUSION_MAX_FRAG_VOX = 5000

    struct = np.ones((3, 3, 3), dtype=bool)

    # ---- numpy <-> SimpleITK helpers ----
    def np_to_sitk(arr_xyz, origin_xyz, spacing):
        arr_zyx = np.ascontiguousarray(np.transpose(arr_xyz, (2, 1, 0)))
        img = sitk.GetImageFromArray(arr_zyx)
        img.SetOrigin(tuple(map(float, origin_xyz)))
        img.SetSpacing((float(spacing), float(spacing), float(spacing)))
        return img

    def sitk_to_np(img):
        arr_zyx = sitk.GetArrayFromImage(img)
        return np.ascontiguousarray(np.transpose(arr_zyx, (2, 1, 0)))

    # ---------- load atlas ----------
    atlas    = np.load(str(atlas_path), allow_pickle=True)
    A_lbl    = atlas['bone_labels']
    A_solid  = atlas['solid']
    A_origin = atlas['origin']
    A_pitch  = float(atlas['pitch'])
    A_surf   = atlas['surface_points']
    A_nbones = int(atlas['n_bones'])
    print(f"  atlas: shape {A_lbl.shape}, pitch {A_pitch}, "
          f"{A_nbones} bones, {len(A_surf):,} surface pts")

    # ---------- voxelize new scan (per-component pymeshfix repair) ----------
    mesh_raw = trimesh.load(str(stl_path))
    comps = mesh_raw.split(only_watertight=False)
    big_comps = [c for c in comps if len(c.vertices) > MIN_COMPONENT_VERTS]
    print(f"  scan: {len(mesh_raw.vertices):,}v, "
          f"{len(comps)} components, {len(big_comps)} substantial")
    fixed = []
    for c in big_comps:
        try:
            mf = pymeshfix.MeshFix(c.vertices, c.faces)
            mf.repair(joincomp=True, remove_smallest_components=True)
            cm = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=True)
            if cm.volume < 0:
                cm.invert()
            fixed.append(cm)
        except Exception:
            pass
    print(f"  repaired {len(fixed)} components")

    all_b = np.array([fm.bounds for fm in fixed])
    b_lo = all_b[:, 0, :].min(axis=0) - 4 * A_pitch
    b_hi = all_b[:, 1, :].max(axis=0) + 4 * A_pitch
    N_shape  = np.ceil((b_hi - b_lo) / A_pitch).astype(int)
    N_origin = b_lo.astype(np.float32)
    N_solid  = np.zeros(N_shape, dtype=bool)
    for fm in fixed:
        vox = fm.voxelized(pitch=A_pitch).fill()
        bm = vox.matrix.astype(bool)
        offset = np.round((vox.translation - N_origin) / A_pitch).astype(int)
        sx, sy, sz = bm.shape
        ox, oy, oz = offset
        x0, x1 = max(ox, 0), min(ox + sx, N_shape[0])
        y0, y1 = max(oy, 0), min(oy + sy, N_shape[1])
        z0, z1 = max(oz, 0), min(oz + sz, N_shape[2])
        sub = bm[x0-ox:x1-ox, y0-oy:y1-oy, z0-oz:z1-oz]
        N_solid[x0:x1, y0:y1, z0:z1] |= sub
    print(f"  scan grid: {tuple(N_shape)}, solid: {N_solid.sum():,}")

    N_boundary = N_solid & ~binary_erosion(N_solid, structure=struct)
    N_surf = np.argwhere(N_boundary).astype(np.float32) * A_pitch + N_origin

    # ---------- Optional: chamfer-selected 7-DOF registration ----------
    # Replaces the PCA + ICP-residual selection + IoU-SMC below with a
    # chamfer-selected, mirror-correct 7-DOF fit (see atlas_register.py). The
    # PCA/ICP still run (cheap, unused) and the IoU-SMC is skipped; A_total is
    # overridden at the composition step. Arrays stay un-mirrored (mirror is in
    # the transform).
    _chamfer_A_total = None
    if registration == "chamfer":
        from atlas_register import register_atlas_7dof_chamfer
        _chamfer_A_total, _cinfo = register_atlas_7dof_chamfer(A_surf, N_surf)
        print(f"  chamfer 7-DOF: chamfer={_cinfo['chamfer']:.3f}mm  "
              f"scale={_cinfo['scale']:.4f}  mirror={_cinfo['mirror']}")

    # ---------- STAGE 1: PCA coarse alignment ----------
    print("  stage 1: PCA coarse alignment")
    def pca_basis(pts):
        c = pts.mean(0)
        z = pts - c
        cov = (z.T @ z) / len(z)
        eigvals, eigvecs = np.linalg.eigh(cov)
        return c, eigvecs[:, np.argsort(eigvals)[::-1]]

    A_c, _ = pca_basis(A_surf)
    N_c, N_R = pca_basis(N_surf)
    rng = np.random.default_rng(0)
    A_sub = A_surf[rng.choice(len(A_surf), min(len(A_surf), ICP_SUBSAMPLE), replace=False)]
    N_sub = N_surf[rng.choice(len(N_surf), min(len(N_surf), ICP_SUBSAMPLE), replace=False)]
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
    print(f"    best: {score0:.2f}mm ({'MIRROR' if mirror_used else 'normal'} atlas)")
    if registration == "chamfer":
        mirror_used = False   # mirror lives in the chamfer transform; keep arrays original

    if mirror_used:
        A_sub_for_icp = A_sub.copy()
        A_sub_for_icp[:, 0] = 2 * A_c[0] - A_sub_for_icp[:, 0]
    else:
        A_sub_for_icp = A_sub

    # ---------- STAGE 2: similarity ICP ----------
    print("  stage 2: similarity ICP")
    def icp_refine(src, tgt, R, t, s=1.0, max_iters=50, tol=1e-3):
        tgt_tree = cKDTree(tgt)
        prev_err = np.inf
        for _it in range(max_iters):
            warped = s * (R @ src.T).T + t
            d, idx = tgt_tree.query(warped)
            nn = tgt[idx]
            src_c = warped.mean(0); nn_c = nn.mean(0)
            Xc = warped - src_c; Yc = nn - nn_c
            H = Xc.T @ Yc
            U, S_, Vt = np.linalg.svd(H)
            D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
            dR = Vt.T @ D @ U.T
            var_x = (Xc ** 2).sum() / len(Xc)
            ds = (S_ * np.diag(D)).sum() / (var_x * len(Xc)) if var_x > 0 else 1.0
            dt = nn_c - ds * (dR @ src_c)
            R = dR @ R; t = ds * (dR @ t) + dt; s = ds * s
            err = float(d.mean())
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        return R, t, s, err

    R1, t1, s1, icp_err = icp_refine(A_sub_for_icp, N_sub, R0, t0, max_iters=ICP_MAX_ITERS)
    print(f"    ICP: {icp_err:.2f}mm, scale={s1:.3f}")

    # Apply mirror to atlas voxel data (needed by both SMC mesh and SITK).
    A_solid_for_sitk  = A_solid
    A_lbl_for_sitk    = A_lbl
    A_origin_for_sitk = A_origin.copy()
    if mirror_used:
        A_solid_for_sitk  = A_solid[::-1, :, :].copy()
        A_lbl_for_sitk    = A_lbl[::-1, :, :].copy()
        A_origin_for_sitk[0] = (2 * A_c[0] - A_origin[0]
                                - (A_solid.shape[0] - 1) * A_pitch)

    # ---------- STAGE 2.5: SMC importance-sampling refinement ----------
    # Builds a mesh from the post-mirror atlas voxels, pre-aligns it with the
    # ICP result, and runs the same SMC alignment we use in step 7.  This is
    # globally more robust than gradient descent (won't get stuck in bad
    # poses), but constrained to 9-DOF (anisotropic similarity).  The SITK
    # affine step below then adds the missing shear DOFs.
    #
    # Memory hygiene: each SMC restart compiles fresh XLA programs and holds
    # GPU memory until the cache is flushed.  Without explicit cleanup the
    # pipeline OOMs after the second _segment_via_atlas call + step 7.  We
    # also decimate both meshes since SMC only samples 8000 surface points
    # anyway — keeping a million marching-cubes faces around is wasteful.
    print("  stage 2.5: SMC importance-sampling refinement")
    import gc as _gc
    try:
        import jax as _jax
        _gc.collect(); _jax.clear_caches()
    except ImportError:
        _jax = None

    av_padded = np.pad(A_solid_for_sitk, 1, constant_values=False).astype(np.float32)
    av, af, _, _ = measure.marching_cubes(av_padded, level=0.5)
    av = (av - 1) * A_pitch + A_origin_for_sitk
    # Pre-apply ICP so SMC starts from a good initial pose
    av_icp = (av @ R1.T) * s1 + t1

    # Decimate both meshes — SMC samples 8000 surface points regardless of
    # face count, so a ~10k-face mesh is sufficient and far cheaper in RAM.
    def _decimate_for_smc(verts_np, faces_np, target_faces: int = 10_000):
        if len(faces_np) <= target_faces * 1.2:
            return verts_np, faces_np
        tr = float(np.clip(1.0 - target_faces / len(faces_np), 0.0, 0.99))
        v2, f2 = fast_simplification.simplify(
            verts_np.astype(np.float64), faces_np.astype(np.int32),
            target_reduction=tr,
        )
        return v2, f2

    av_dec, af_dec = _decimate_for_smc(av_icp, af)
    atlas_mesh_smc = trimesh.Trimesh(vertices=av_dec, faces=af_dec, process=True)

    scan_full = trimesh.util.concatenate(fixed)
    scan_full.merge_vertices()
    sv_dec, sf_dec = _decimate_for_smc(scan_full.vertices, scan_full.faces)
    scan_mesh_smc = trimesh.Trimesh(vertices=sv_dec, faces=sf_dec, process=True)
    del scan_full
    print(f"    atlas mesh: {len(atlas_mesh_smc.faces):,}f  |  "
          f"scan mesh: {len(scan_mesh_smc.faces):,}f")

    smc_params = smc_mirror_axis = None
    try:
        if registration != "chamfer":
            from smc_align import _run_is
            smc_params, smc_iou, smc_baseline, smc_mirror_axis = _run_is(
                ref_mesh=scan_mesh_smc, sample_mesh=atlas_mesh_smc,
                seed=0, n_restarts=3,
            )
            print(f"    SMC IoU: {smc_iou:.4f}  (baseline {smc_baseline:.4f})")
    except Exception as e:
        print(f"    SMC failed: {type(e).__name__}: {e} — continuing with ICP only")

    smc_atlas_centroid = atlas_mesh_smc.vertices.mean(0).astype(np.float64)

    # Free GPU/XLA memory before the next stage / next _segment_via_atlas call
    del atlas_mesh_smc, scan_mesh_smc, av, af, av_icp, av_dec, af_dec, sv_dec, sf_dec
    if _jax is not None:
        _gc.collect(); _jax.clear_caches()

    # ---------- STAGE 3: SimpleITK affine refinement ----------
    print("  stage 3: SimpleITK affine")

    def signed_dt(solid_xyz, origin, pitch):
        img = np_to_sitk(solid_xyz.astype(np.uint8), origin, pitch)
        dt_filter = sitk.SignedMaurerDistanceMapImageFilter()
        dt_filter.SetSquaredDistance(False)
        dt_filter.SetUseImageSpacing(True)
        return sitk.Cast(dt_filter.Execute(img), sitk.sitkFloat32)

    A_dt = signed_dt(A_solid_for_sitk, A_origin_for_sitk, A_pitch)
    N_dt = signed_dt(N_solid, N_origin, A_pitch)

    # ---- Build composed atlas→scan affine: ICP, then SMC (mirror + transform).
    def _to_4x4(M, t_vec):
        T = np.eye(4); T[:3, :3] = M; T[:3, 3] = t_vec; return T

    def _smc_mirror_4x4(centroid, axis):
        D = np.eye(3); D[axis, axis] = -1
        v = np.zeros(3); v[axis] = 2.0 * centroid[axis]
        return _to_4x4(D, v)

    def _smc_transform_4x4(centroid, p):
        crx, srx = np.cos(p['rx']), np.sin(p['rx'])
        cry, sry = np.cos(p['ry']), np.sin(p['ry'])
        crz, srz = np.cos(p['rz']), np.sin(p['rz'])
        Rx = np.array([[1, 0, 0], [0, crx, -srx], [0, srx, crx]])
        Ry = np.array([[cry, 0, sry], [0, 1, 0], [-sry, 0, cry]])
        Rz = np.array([[crz, -srz, 0], [srz, crz, 0], [0, 0, 1]])
        M  = Rz @ Ry @ Rx @ np.diag([p['sx'], p['sy'], p['sz']])
        t_smc = np.array([p['tx'], p['ty'], p['tz']])
        # transform_pts: p' = M(p - c) + c + t = Mp + (c - Mc + t)
        return _to_4x4(M, centroid - M @ centroid + t_smc)

    A_total = _to_4x4(s1 * R1, t1)               # ICP atlas→scan
    if smc_params is not None:
        if smc_mirror_axis is not None:
            A_total = _smc_mirror_4x4(smc_atlas_centroid, smc_mirror_axis) @ A_total
        A_total = _smc_transform_4x4(smc_atlas_centroid, smc_params) @ A_total
    if _chamfer_A_total is not None:
        A_total = _chamfer_A_total               # chamfer-selected 7-DOF overrides

    A_inv = np.linalg.inv(A_total)               # scan→atlas (SITK convention)
    M_inv = A_inv[:3, :3]
    t_inv = A_inv[:3, 3]

    initial = sitk.AffineTransform(3)
    initial.SetMatrix(M_inv.flatten().tolist())
    initial.SetTranslation(t_inv.tolist())

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=4.0, minStep=0.01, numberOfIterations=AFFINE_ITERS,
        gradientMagnitudeTolerance=1e-5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)
    reg.SetShrinkFactorsPerLevel([8, 4, 2])
    reg.SetSmoothingSigmasPerLevel([4, 2, 1])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_affine = reg.Execute(N_dt, A_dt)
    print(f"    affine metric: {reg.GetMetricValue():.4f}, "
          f"iters {reg.GetOptimizerIteration()}")

    # ---------- label transfer ----------
    A_lbl_sitk   = np_to_sitk(A_lbl_for_sitk.astype(np.uint8), A_origin_for_sitk, A_pitch)
    N_solid_sitk = np_to_sitk(N_solid.astype(np.uint8), N_origin, A_pitch)
    resampled = sitk.Resample(
        A_lbl_sitk, N_solid_sitk,
        transform=final_affine,
        interpolator=sitk.sitkNearestNeighbor,
        defaultPixelValue=0,
    )
    N_lbl = sitk_to_np(resampled).astype(np.uint8)
    N_lbl[~N_solid] = 0

    # ---- DEBUG: dump the atlas template + aligned scan surface (scan-physical
    #      frame, same as the returned per-bone meshes) for QC visualisation. ----
    if debug_dir is not None:
        from pathlib import Path as _P
        _dd = _P(debug_dir); _dd.mkdir(parents=True, exist_ok=True)
        _inv = final_affine.GetInverse()               # atlas-phys -> scan-phys
        _av, _af, _, _ = measure.marching_cubes(
            np.pad(A_solid_for_sitk, 1, constant_values=False).astype(np.float32), 0.5)
        _av = (_av - 1) * A_pitch + A_origin_for_sitk                  # atlas-phys
        _avs = np.array([_inv.TransformPoint([float(c) for c in p]) for p in _av],
                        dtype=np.float32)                              # -> scan-phys
        _ai = np.clip(np.round((_av - A_origin_for_sitk) / A_pitch).astype(int),
                      0, np.array(A_lbl_for_sitk.shape) - 1)
        _vl = A_lbl_for_sitk[_ai[:, 0], _ai[:, 1], _ai[:, 2]].astype(np.int16)
        _sv, _sf, _, _ = measure.marching_cubes(
            np.pad(N_solid, 1, constant_values=False).astype(np.float32), 0.5)
        _sv = ((_sv - 1) * A_pitch + N_origin).astype(np.float32)
        np.savez(_dd / f"{stl_path.stem}_atlasdbg.npz",
                 atlas_verts=_avs, atlas_faces=_af.astype(np.int32), atlas_vlabels=_vl,
                 scan_verts=_sv, scan_faces=_sf.astype(np.int32))
        print(f"    [debug] {stl_path.stem}_atlasdbg.npz  atlas {len(_avs):,}v  "
              f"scan {len(_sv):,}v | atlas-scan bbox "
              f"{np.round(_avs.min(0),1)}..{np.round(_avs.max(0),1)}  vs scan "
              f"{np.round(_sv.min(0),1)}..{np.round(_sv.max(0),1)}", flush=True)

    # Refill unlabeled by nearest labeled neighbour
    unlabeled = N_solid & (N_lbl == 0)
    if unlabeled.any():
        labeled_mask = N_solid & (N_lbl > 0)
        _, indices = distance_transform_edt(~labeled_mask, return_indices=True)
        N_lbl[unlabeled] = N_lbl[indices[0][unlabeled],
                                  indices[1][unlabeled],
                                  indices[2][unlabeled]]

    # Keep largest CC per bone
    for lid in range(1, A_nbones + 1):
        mask = (N_lbl == lid)
        if not mask.any():
            continue
        cc, n = _label(mask, structure=struct)
        if n <= 1:
            continue
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        N_lbl[mask & (cc != int(np.argmax(sizes)))] = 0

    # Re-fill orphans after CC pruning
    unlabeled2 = N_solid & (N_lbl == 0)
    if unlabeled2.any():
        labeled_mask = N_solid & (N_lbl > 0)
        _, indices = distance_transform_edt(~labeled_mask, return_indices=True)
        N_lbl[unlabeled2] = N_lbl[indices[0][unlabeled2],
                                   indices[1][unlabeled2],
                                   indices[2][unlabeled2]]

    # ---------- neck rule: neck of bigger bone → smaller bone ----------
    print("  neck rule")
    present = sorted([int(l) for l in np.unique(N_lbl) if l > 0])
    sizes_now = {l: int((N_lbl == l).sum()) for l in present}
    pairs_seen = set()
    adjacency = []
    for la in present:
        ma = (N_lbl == la)
        nbrs = N_lbl[binary_dilation(ma, structure=struct) & ~ma & N_solid]
        for lb in np.unique(nbrs):
            if lb == 0 or lb == la:
                continue
            key = tuple(sorted([int(la), int(lb)]))
            if key in pairs_seen:
                continue
            pairs_seen.add(key)
            adjacency.append(key)

    for la, lb in adjacency:
        sa, sb = sizes_now[la], sizes_now[lb]
        small, big = (la, lb) if sa < sb else (lb, la)
        mask_small = (N_lbl == small)
        mask_big   = (N_lbl == big)
        if not (mask_small & binary_dilation(mask_big, structure=struct)).any():
            continue

        big_pts = np.argwhere(mask_big).astype(np.float64)
        if len(big_pts) < 100:
            continue
        cov = np.cov((big_pts - big_pts.mean(0)).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        long_axis = eigvecs[:, np.argmax(eigvals)]

        grid_idx  = np.indices(N_lbl.shape).astype(np.float64).reshape(3, -1).T
        proj_full = (grid_idx @ long_axis).reshape(N_lbl.shape)
        proj_big   = proj_full[mask_big]
        proj_small = proj_full[mask_small]

        bin_count = max(int(proj_big.max() - proj_big.min()), 30)
        bins = np.linspace(proj_big.min(), proj_big.max(), bin_count + 1)
        area, _ = np.histogram(proj_big, bins=bins)
        area = area.astype(float) * A_pitch ** 2
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        k = max(3, bin_count // 15)
        area_s = np.convolve(area, np.ones(k) / k, mode='same')

        peak_idx  = int(np.argmax(area_s))
        threshold = area_s[peak_idx] * NECK_AREA_FRAC
        dir_to_small = 1 if proj_small.mean() > proj_big.mean() else -1
        cut_idx = peak_idx
        if dir_to_small > 0:
            for i in range(peak_idx, len(area_s)):
                if area_s[i] < threshold:
                    cut_idx = i; break
        else:
            for i in range(peak_idx, -1, -1):
                if area_s[i] < threshold:
                    cut_idx = i; break
        cut_proj = bin_centers[cut_idx]

        if dir_to_small > 0:
            plane_mask = mask_big & (proj_full > cut_proj)
        else:
            plane_mask = mask_big & (proj_full < cut_proj)
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
        print(f"    bones {small}<-{big}: cut at "
              f"{area_s[cut_idx] / area_s[peak_idx] * 100:.0f}% of peak; "
              f"moved {n_moved:,} neck vox")

    # ---------- stranded-fragment cleanup ----------
    print("  stranded-fragment cleanup")
    present = sorted([int(l) for l in np.unique(N_lbl) if l > 0])
    total_reassigned = 0
    for lid in present:
        mask = (N_lbl == lid)
        cc, n_cc = _label(mask, structure=struct)
        if n_cc <= 1:
            continue
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        main_id   = int(np.argmax(sizes))
        main_size = sizes[main_id]
        for frag_id in range(1, n_cc + 1):
            if frag_id == main_id:
                continue
            frag_mask = (cc == frag_id)
            frag_size = int(frag_mask.sum())
            if frag_size > main_size * 0.30:
                continue
            nbrs = N_lbl[binary_dilation(frag_mask, structure=struct) & ~frag_mask]
            nbrs = nbrs[(nbrs != 0) & (nbrs != lid)]
            if len(nbrs) == 0:
                continue
            vals, counts = np.unique(nbrs, return_counts=True)
            N_lbl[frag_mask] = int(vals[np.argmax(counts)])
            total_reassigned += frag_size
    if total_reassigned:
        print(f"    reassigned {total_reassigned:,} stranded voxels")

    # ---------- intrusion cleanup ----------
    print("  intrusion cleanup")
    intrusion_total = 0
    for lid in present:
        mask = (N_lbl == lid)
        if mask.sum() < 200:
            continue
        eroded = binary_erosion(mask, structure=struct, iterations=INTRUSION_OPEN_VOX)
        if not eroded.any():
            continue
        kept = binary_dilation(eroded, structure=struct,
                               iterations=INTRUSION_OPEN_VOX) & mask
        intrusion = mask & ~kept
        if not intrusion.any():
            continue
        cc, n_cc = _label(intrusion, structure=struct)
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        moved = 0
        for fid in range(1, n_cc + 1):
            if int(sizes[fid]) > INTRUSION_MAX_FRAG_VOX:
                continue
            idx = np.argwhere(cc == fid)
            lo  = np.maximum(idx.min(0) - 1, 0)
            hi  = np.minimum(idx.max(0) + 2, np.array(cc.shape))
            sl  = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
            frag_local = (cc[sl] == fid)
            nbr_local  = binary_dilation(frag_local, structure=struct) & ~frag_local
            nbr_labels = N_lbl[sl][nbr_local]
            n_other = int(((nbr_labels != 0) & (nbr_labels != lid)).sum())
            n_self  = int((nbr_labels == lid).sum())
            if n_other < 2 * n_self:
                continue
            non_self = nbr_labels[(nbr_labels != 0) & (nbr_labels != lid)]
            if len(non_self) == 0:
                continue
            vals, counts = np.unique(non_self, return_counts=True)
            N_lbl[sl][frag_local] = int(vals[np.argmax(counts)])
            moved += int(sizes[fid])
        intrusion_total += moved
    if intrusion_total:
        print(f"    pruned {intrusion_total:,} intrusion voxels")

    # ---------- marching cubes per surviving bone (with decimation) ----------
    # Decimate proportionally so total output face count matches the input STL.
    # Without this, downstream proximity queries (step 8) OOM on ~10× denser meshes.
    print("  building per-bone meshes")
    input_face_count = len(mesh_raw.faces)

    # First pass: collect masks and voxel counts
    bone_masks: dict[int, np.ndarray] = {}
    for lid in range(1, A_nbones + 1):
        mask = (N_lbl == lid)
        if mask.sum() < MIN_BONE_VOX:
            continue
        mask = binary_erosion(mask, structure=struct, iterations=1)
        mask = binary_dilation(mask, structure=struct, iterations=1)
        if mask.sum() == 0:
            continue
        cc, n = _label(mask, structure=struct)
        if n > 1:
            sizes = np.bincount(cc.ravel()); sizes[0] = 0
            mask = (cc == int(np.argmax(sizes)))
        mask = binary_fill_holes(mask)
        bone_masks[lid] = mask
    total_kept_vox = sum(int(m.sum()) for m in bone_masks.values()) or 1

    # Cap per-bone face count so high-res input scans don't produce dense
    # meshes that OOM step-8 proximity queries.  Proportional to voxel volume
    # within the cap so smaller bones still get fewer faces.
    MAX_FACES_PER_BONE = 80_000

    bone_meshes: dict[int, trimesh.Trimesh] = {}
    for lid, mask in bone_masks.items():
        target = max(int(input_face_count * int(mask.sum()) / total_kept_vox), 500)
        target = min(target, MAX_FACES_PER_BONE)
        padded = np.pad(mask, 1, constant_values=False).astype(np.float32)
        verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
        verts = (verts - 1) * A_pitch + N_origin
        if len(faces) > target * 1.2:
            tr = float(np.clip(1.0 - target / len(faces), 0.0, 0.99))
            verts, faces = fast_simplification.simplify(
                verts.astype(np.float64), faces.astype(np.int32),
                target_reduction=tr,
            )
        bm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        if bm.volume < 0:
            bm.invert()
        bone_meshes[lid] = bm
        print(f"    bone {lid:>2}: {len(bm.vertices):>6,}v {len(bm.faces):>6,}f "
              f"vol={bm.volume:>7.0f}mm³")
    if return_transform:
        # final_affine maps SCAN physical coords -> ATLAS physical coords
        return bone_meshes, final_affine
    return bone_meshes


# ---------------------------------------------------------------------------
# Articular surface: protrusion-based patch selection
# ---------------------------------------------------------------------------

def _facing_side_indices(
    verts: np.ndarray,
    want_min_x: bool,
    yz_eps_abs: float,
) -> np.ndarray:
    """
    For each y-z cell of size yz_eps_abs, return the index of the vertex
    closest to the opposing bone in x (min-x when want_min_x, else max-x).
    Discards back-face duplicates so only the articular surface is considered.
    """
    yz_cells = (verts[:, 1:3] / max(yz_eps_abs, 1e-12)).round().astype(np.int64)
    x = verts[:, 0]
    cell_best: dict = {}
    for i, cell in enumerate(map(tuple, yz_cells)):
        xi = float(x[i])
        if cell not in cell_best:
            cell_best[cell] = (xi, i)
        else:
            bx, _ = cell_best[cell]
            if (want_min_x and xi < bx) or (not want_min_x and xi > bx):
                cell_best[cell] = (xi, i)
    return np.array([idx for _, idx in cell_best.values()], dtype=np.int64)


def _flood_fill_faces(
    mesh: trimesh.Trimesh,
    seed_face: int,
    center_pt: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    BFS flood fill over mesh faces starting from seed_face.
    Expands to adjacent faces whose centroid is within radius of center_pt.
    Returns bool array (n_faces,).
    The connectivity constraint keeps the fill on one surface of the mesh —
    it cannot jump from articular to back face without traversing the bone edge.
    """
    from collections import deque

    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)

    adj: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for f1, f2 in mesh.face_adjacency:
        adj[int(f1)].append(int(f2))
        adj[int(f2)].append(int(f1))

    face_mask = np.zeros(len(mesh.faces), dtype=bool)
    face_mask[seed_face] = True
    queue: deque[int] = deque([seed_face])
    while queue:
        fi = queue.popleft()
        for fj in adj[fi]:
            if face_mask[fj]:
                continue
            if np.linalg.norm(face_centroids[fj] - center_pt) < radius:
                face_mask[fj] = True
                queue.append(fj)
    return face_mask


def _closed_patch_volume(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> float:
    """
    Volume of the closed mesh formed by the patch faces + a fan-triangulated cap.

    The patch is an open surface (a curved patch with a boundary loop).
    We close it by adding triangles from each boundary edge to the centroid of
    the boundary vertices, then compute the absolute signed volume.
    """
    patch_faces = mesh.faces[face_mask]

    # Dense re-indexing so trimesh.volume works on the sub-mesh
    vert_idx = np.unique(patch_faces.ravel())
    remap = np.zeros(len(mesh.vertices), dtype=np.int64)
    remap[vert_idx] = np.arange(len(vert_idx))
    dense_faces = remap[patch_faces]
    dense_verts = mesh.vertices[vert_idx].copy()

    # Find directed boundary edges: half-edges that appear in exactly one face
    all_half = np.vstack([dense_faces[:, [0, 1]],
                          dense_faces[:, [1, 2]],
                          dense_faces[:, [2, 0]]])
    canon = np.sort(all_half, axis=1)
    _, inv, counts = np.unique(canon, axis=0, return_inverse=True, return_counts=True)
    boundary = all_half[counts[inv] == 1]   # directed (v1, v2) in face winding order

    # Cap: fan from centroid of boundary vertices
    bverts = np.unique(boundary.ravel())
    cap_pt  = dense_verts[bverts].mean(axis=0)
    cap_idx = len(dense_verts)

    # Cap triangles: reverse each boundary edge so the surface closes consistently
    cap_faces = np.column_stack([boundary[:, 1], boundary[:, 0],
                                 np.full(len(boundary), cap_idx, dtype=np.int64)])

    verts_c = np.vstack([dense_verts, cap_pt])
    faces_c = np.vstack([dense_faces, cap_faces])

    closed = trimesh.Trimesh(vertices=verts_c, faces=faces_c, process=False)
    return abs(float(closed.volume))


def _combined_patch_volume(
    sample_mesh: trimesh.Trimesh,
    sample_face_mask: np.ndarray,
    ref_mesh: trimesh.Trimesh,
    ref_face_mask: np.ndarray,
) -> float:
    """
    Volume of the closed mesh formed by the sample patch + the flipped ref patch
    + a fan-triangulated cap stitching their perimeters.

    The sample patch and the ref patch face roughly the same direction (toward
    bone_1). Flipping the ref winding makes it face the opposite way, so the
    two surfaces together form a 'blister' that encloses the osteophyte volume.
    Any remaining open boundary is closed with a fan cap.
    """
    def _dense(mesh, fmask, flip):
        faces = mesh.faces[fmask]
        if flip:
            faces = faces[:, ::-1]
        vi = np.unique(faces.ravel())
        remap = np.zeros(len(mesh.vertices), dtype=np.int64)
        remap[vi] = np.arange(len(vi))
        return mesh.vertices[vi].copy(), remap[faces]

    s_verts, s_faces = _dense(sample_mesh, sample_face_mask, flip=False)
    r_verts, r_faces = _dense(ref_mesh,    ref_face_mask,    flip=True)

    # Combine, offset ref face indices
    combined_verts = np.vstack([s_verts, r_verts])
    combined_faces = np.vstack([s_faces, r_faces + len(s_verts)])

    # Fan-cap any remaining boundary edges
    all_half = np.vstack([combined_faces[:, [0, 1]],
                          combined_faces[:, [1, 2]],
                          combined_faces[:, [2, 0]]])
    canon = np.sort(all_half, axis=1)
    _, inv, counts = np.unique(canon, axis=0, return_inverse=True, return_counts=True)
    boundary = all_half[counts[inv] == 1]

    if len(boundary) > 0:
        bverts = np.unique(boundary.ravel())
        cap_pt  = combined_verts[bverts].mean(axis=0)
        cap_idx = len(combined_verts)
        cap_faces = np.column_stack([boundary[:, 1], boundary[:, 0],
                                     np.full(len(boundary), cap_idx, dtype=np.int64)])
        combined_verts = np.vstack([combined_verts, cap_pt])
        combined_faces = np.vstack([combined_faces, cap_faces])

    closed = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces, process=False)
    return abs(float(closed.volume)), closed


def _heightmap_volume(
    sample_mesh: trimesh.Trimesh,
    sample_face_mask: np.ndarray,
    ref_mesh: trimesh.Trimesh,
    ref_face_mask: np.ndarray,
    protrusion_threshold_vox: float = 1.0,
) -> float:
    """
    Volume of the osteophyte protrusion using a height-map integral.

    For each sample patch face:
      - find the nearest ref patch face in y-z (ignores x, which differs due to protrusion)
      - depth = ref_x - sample_x   (positive = sample protrudes toward bone_1)
      - contribution = max(0, depth - threshold) × face_area

    The threshold (default 1 vox = 1 voxel edge) discards subvoxel noise and
    alignment error, bringing the result in line with Dragonfly's voxel count.

    Returns raw volume in mesh units³ (multiply by volume_scale for mm³).
    """
    s_cents = sample_mesh.vertices[sample_mesh.faces[sample_face_mask]].mean(axis=1)
    r_cents = ref_mesh.vertices[ref_mesh.faces[ref_face_mask]].mean(axis=1)

    tree = KDTree(r_cents[:, 1:3])   # match in y-z only
    _, idx = tree.query(s_cents[:, 1:3])

    x_diff = r_cents[idx, 0] - s_cents[:, 0]   # + = osteophyte protrudes toward bone_1
    depth  = np.maximum(x_diff - protrusion_threshold_vox, 0.0)
    return float(np.sum(depth * sample_mesh.area_faces[sample_face_mask]))


def _articular_mask_protrusion(
    b2: trimesh.Trimesh,
    b1: trimesh.Trimesh,
    neighborhood_radius: float,
    patch_radius: float,
    yz_eps: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Detect the articular spike on bone_2 (target) and return the patch of
    faces around it.

    Spike detection: orientation-invariant.  The tip is simply the vertex of
    bone_2 closest in 3D Euclidean distance to bone_1's surface.  The
    anomaly is the protrusion of bone_2 that extends toward bone_1, so its
    tip is by definition the bone_2 point nearest bone_1.

    Patch selection: BFS flood fill from the triangle containing the spike
    tip, expanding to adjacent triangles within patch_radius of the spike
    tip.  Connectivity ensures the patch stays on the articular surface — the
    fill cannot cross to the back face without going around the bone edge.

    Parameters
    ----------
    b2 : target bone (the one with the protrusion, e.g. bone 4 / sample_2)
    b1 : reference bone the protrusion points toward (e.g. bone 1 / sample_1)
    neighborhood_radius, yz_eps : unused (kept for backward-compatible signature)

    Returns
    -------
    face_mask  : bool array (n_faces_b2,) — True for patch faces
    d          : float array (n_verts_b2,) — nearest distance to bone_1
    spike_pt   : (3,) — spike tip position in 3D
    spike_d    : float — distance to bone_1 at the spike
    """
    verts = b2.vertices

    v = verts - verts.mean(0)
    cov = (v.T @ v) / max(len(v) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    bone_length = float(np.sqrt(max(eigvals[-1], 0.0))) * 2
    r_prime_abs = patch_radius * bone_length

    # Distance from every b2 vertex to b1's surface — used both for spike
    # detection and for the Σ d·a volume integral downstream.
    _, d, _ = trimesh.proximity.closest_point(b1, verts)
    d = d.astype(np.float64)

    # Spike = the bone_2 vertex closest to bone_1.
    spike_idx = int(np.argmin(d))
    spike_pt  = verts[spike_idx]
    spike_d   = float(d[spike_idx])

    # Seed face: the face containing spike_idx whose centroid is closest to spike_pt
    seed_faces     = np.where(np.any(b2.faces == spike_idx, axis=1))[0]
    face_centroids = b2.vertices[b2.faces].mean(axis=1)
    seed_face      = int(seed_faces[np.argmin(
        np.linalg.norm(face_centroids[seed_faces] - spike_pt, axis=1))])

    face_mask = _flood_fill_faces(b2, seed_face, spike_pt, r_prime_abs)

    print(f"  Bone length: {bone_length:.2f}  r_prime_abs: {r_prime_abs:.2f}")
    print(f"  Spike (closest b2 vertex to b1): "
          f"({spike_pt[0]:.3f}, {spike_pt[1]:.3f}, {spike_pt[2]:.3f})  "
          f"d_to_b1={spike_d:.4f}")
    print(f"  Articular patch: {face_mask.sum()} faces")

    return face_mask, d, spike_pt, spike_d


def _injection_volume_protrusion(
    sample2_path:        str,
    sample1_path:        str,
    ref2_path:           str,
    ref1_path:           str,
    neighborhood_radius:        float,
    patch_radius:               float,
    yz_eps:                     float = 0.005,
    protrusion_threshold_vox:   float = 1.0,
    ply_sample2:                str | None = None,
    ply_ref2:                   str | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Compute injection volume using the protrusion-based articular patch method.

    The articular patch on sample_2 is found via distance-to-bone_1 deviation.
    The matching region on ref_2 is found by projecting the spike tip's y-z
    coordinates onto the facing side of ref_2, then expanding a ball of
    patch_radius around that center — x is ignored because the spike protrudes
    in x and 3D NN would land at the wrong anatomical location.

    PLY exports color the articular patch red and the rest grey.

    Returns
    -------
    ref_art_vol    : Σ d·a over matched ref_2 patch (distances to ref_1)
    sample_art_vol : Σ d·a over sample_2 patch (distances to sample_1)
    injection_vol  : ref_art_vol - sample_art_vol
    """
    RED  = np.array([255,   0,   0], dtype=np.uint8)
    GREY = np.array([180, 180, 180], dtype=np.uint8)

    s2 = trimesh.load(sample2_path, force="mesh", process=False)
    s1 = trimesh.load(sample1_path, force="mesh", process=False)
    r2 = trimesh.load(ref2_path,    force="mesh", process=False)
    r1 = trimesh.load(ref1_path,    force="mesh", process=False)
    for m in (s2, s1, r2, r1):
        m.merge_vertices()

    # --- sample_2 articular patch (face-based flood fill) ---
    print(f"\n  [sample_2] Finding articular patch ...")
    sample_face_mask, s_dists, spike_tip, d_bar_spike = _articular_mask_protrusion(
        s2, s1, neighborhood_radius, patch_radius, yz_eps)

    # Volume: Σ face_d × face_area  (face_d = mean of 3 vertex distances)
    s_face_d = s_dists[s2.faces].mean(axis=1)
    sample_art_vol = float(np.sum(s_face_d[sample_face_mask] * s2.area_faces[sample_face_mask]))
    print(f"  sample articular volume (Σd·a): {sample_art_vol:.4f}")

    # --- ref_2 matching patch: flood fill from closest point to spike_tip ---
    print(f"\n  [ref_2] Matching patch by flood fill from closest point to spike_tip ...")
    r2_verts = r2.vertices
    r2v = r2_verts - r2_verts.mean(0)
    r2_bone_length = float(np.sqrt(max(np.linalg.eigvalsh(
        (r2v.T @ r2v) / max(len(r2v) - 1, 1))[-1], 0.0))) * 2
    r2_prime_abs = patch_radius * r2_bone_length

    _, r_dists, _ = trimesh.proximity.closest_point(r1, r2_verts)
    r_dists = r_dists.astype(np.float64)

    # Seed face on ref_2 = face closest to spike_tip on the ref_2 surface
    seed_r2_pts, _, seed_r2_faces = trimesh.proximity.closest_point(r2, spike_tip.reshape(1, 3))
    seed_r2    = int(seed_r2_faces[0])
    seed_r2_pt = seed_r2_pts[0]

    r2_face_d = r_dists[r2.faces].mean(axis=1)
    ref_face_mask = _flood_fill_faces(r2, seed_r2, seed_r2_pt, r2_prime_abs)
    ref_art_vol = float(np.sum(r2_face_d[ref_face_mask] * r2.area_faces[ref_face_mask]))
    print(f"  Spike tip:      ({spike_tip[0]:.3f}, {spike_tip[1]:.3f}, {spike_tip[2]:.3f})")
    print(f"  Closest on ref_2: ({seed_r2_pt[0]:.3f}, {seed_r2_pt[1]:.3f}, {seed_r2_pt[2]:.3f})")
    print(f"  ref_2 patch: {ref_face_mask.sum()} faces")
    print(f"  ref articular volume (Σd·a):    {ref_art_vol:.4f}")

    # --- PLY export: patch faces → vertices colored red, rest grey ---
    if ply_sample2 is not None:
        s_vert_in_patch = np.zeros(len(s2.vertices), dtype=bool)
        s_vert_in_patch[s2.faces[sample_face_mask].ravel()] = True
        colors = np.where(s_vert_in_patch[:, None], RED, GREY)
        s2.visual = trimesh.visual.ColorVisuals(mesh=s2, vertex_colors=colors)
        s2.export(ply_sample2)
        print(f"  Articular patch (sample_2) → {ply_sample2}")
        # Save spike tip for visualization
        np.save(str(Path(ply_sample2).parent / "spike_tip.npy"), spike_tip)

    if ply_ref2 is not None:
        r_vert_in_patch = np.zeros(len(r2.vertices), dtype=bool)
        r_vert_in_patch[r2.faces[ref_face_mask].ravel()] = True
        colors = np.where(r_vert_in_patch[:, None], RED, GREY)
        r2.visual = trimesh.visual.ColorVisuals(mesh=r2, vertex_colors=colors)
        r2.export(ply_ref2)
        print(f"  Articular patch (ref_2)    → {ply_ref2}")

    # --- Alt-1: height-map integral (x-protrusion × face area) ---
    # Closest to Dragonfly: for each sample face, depth = ref_x - sample_x,
    # volume = Σ max(0, depth - threshold) × face_area.
    # threshold=1 vox strips subvoxel noise / alignment error.
    print(f"\n  [alt-1] Computing height-map protrusion volume ...")
    alt_vol = _heightmap_volume(
        s2, sample_face_mask, r2, ref_face_mask, protrusion_threshold_vox)
    print(f"  Alt-1 (height-map, threshold={protrusion_threshold_vox} vox):  {alt_vol:.4f}")

    # Build and save the blister mesh for visualization (separate from volume)
    _, blister_mesh = _combined_patch_volume(s2, sample_face_mask, r2, ref_face_mask)
    if ply_sample2 is not None:
        GREEN = np.array([0, 200, 0], dtype=np.uint8)
        green = np.tile(GREEN, (len(blister_mesh.vertices), 1))
        blister_mesh.visual = trimesh.visual.ColorVisuals(
            mesh=blister_mesh, vertex_colors=green)
        blister_path = str(Path(ply_sample2).parent / "blister_mesh.ply")
        blister_mesh.export(blister_path)
        print(f"  Blister mesh (viz only) → {blister_path}")

    # --- Alt-2: Dragonfly-style — N labeled points × voxel_size³ ---
    # Each patch triangle center is treated as one labeled voxel.
    # Multiplying by volume_scale (= voxel_size³) in run_pipeline gives mm³.
    alt_vol2 = float(sample_face_mask.sum())
    print(f"  Alt-2 (Dragonfly N×vox³): {alt_vol2:.0f} triangle centers")

    return ref_art_vol, sample_art_vol, ref_art_vol - sample_art_vol, alt_vol, alt_vol2

    # ------------------------------------------------------------------
    # DISABLED: sample_1 spike cross-validation.
    # Was: also find a spike on sample_1 (toward sample_2); if its tip is
    # within SPIKE_MATCH_TOL_MM (0.01 mm) of the sample_2 spike tip, run
    # the full step-8 pipeline on the sample_1/ref_1 pair too and sum the
    # volumes.  PLYs were always written so the detection could be inspected.
    # Uncomment the block below (and remove the `return` above) to re-enable.
    # ------------------------------------------------------------------
    # SPIKE_MATCH_TOL_MM = 0.01
    #
    # # spike search on both bones (cross-validation)
    # print(f"\n  [sample_2] Finding articular patch ...")
    # s2_face_mask, s2_dists_to_s1, spike_tip_s2, _ = _articular_mask_protrusion(
    #     s2, s1, neighborhood_radius, patch_radius, yz_eps)
    #
    # print(f"\n  [sample_1] Finding articular patch (cross-validation) ...")
    # s1_face_mask, s1_dists_to_s2, spike_tip_s1, _ = _articular_mask_protrusion(
    #     s1, s2, neighborhood_radius, patch_radius, yz_eps)
    #
    # spike_dist = float(np.linalg.norm(spike_tip_s2 - spike_tip_s1))
    # spikes_match = spike_dist < SPIKE_MATCH_TOL_MM
    # print(f"\n  Spike-tip cross-check:")
    # print(f"    sample_2 spike: {spike_tip_s2}")
    # print(f"    sample_1 spike: {spike_tip_s1}")
    # print(f"    distance:       {spike_dist:.4f} mm   (threshold {SPIKE_MATCH_TOL_MM} mm)")
    # if spikes_match:
    #     print(f"  Spikes match — running full step 8 on both bones and summing volumes.")
    # else:
    #     print(f"  Spikes do not match — running both bones for VISUALIZATION ONLY; "
    #           f"volumes will be set to 0.")
    #
    # def _compute_pair(
    #     sample_b, sample_b_face_mask, sample_dists_to_other, spike_tip,
    #     ref_b, ref_other,
    #     ply_sample, ply_ref, label_b,
    # ):
    #     """Run the ref-side flood fill + Σ d·a + alt volumes for one bone pair."""
    #     s_face_d = sample_dists_to_other[sample_b.faces].mean(axis=1)
    #     sample_vol = float(np.sum(
    #         s_face_d[sample_b_face_mask] * sample_b.area_faces[sample_b_face_mask]))
    #     print(f"  sample_{label_b} articular volume (Σd·a): {sample_vol:.4f}")
    #
    #     rb_verts = ref_b.vertices
    #     rbv = rb_verts - rb_verts.mean(0)
    #     rb_bone_length = float(np.sqrt(max(np.linalg.eigvalsh(
    #         (rbv.T @ rbv) / max(len(rbv) - 1, 1))[-1], 0.0))) * 2
    #     rb_prime_abs = patch_radius * rb_bone_length
    #
    #     _, r_dists, _ = trimesh.proximity.closest_point(ref_other, rb_verts)
    #     r_dists = r_dists.astype(np.float64)
    #
    #     seed_pts, _, seed_faces = trimesh.proximity.closest_point(ref_b, spike_tip.reshape(1, 3))
    #     seed_face = int(seed_faces[0])
    #     seed_pt   = seed_pts[0]
    #
    #     rb_face_d = r_dists[ref_b.faces].mean(axis=1)
    #     ref_face_mask = _flood_fill_faces(ref_b, seed_face, seed_pt, rb_prime_abs)
    #     ref_vol = float(np.sum(rb_face_d[ref_face_mask] * ref_b.area_faces[ref_face_mask]))
    #     print(f"  ref_{label_b} patch: {ref_face_mask.sum()} faces  Σd·a: {ref_vol:.4f}")
    #
    #     if ply_sample is not None:
    #         sv = np.zeros(len(sample_b.vertices), dtype=bool)
    #         sv[sample_b.faces[sample_b_face_mask].ravel()] = True
    #         colors = np.where(sv[:, None], RED, GREY)
    #         sample_b.visual = trimesh.visual.ColorVisuals(mesh=sample_b, vertex_colors=colors)
    #         sample_b.export(ply_sample)
    #         np.save(str(Path(ply_sample).parent / f"spike_tip_{label_b}.npy"), spike_tip)
    #
    #     if ply_ref is not None:
    #         rv = np.zeros(len(ref_b.vertices), dtype=bool)
    #         rv[ref_b.faces[ref_face_mask].ravel()] = True
    #         colors = np.where(rv[:, None], RED, GREY)
    #         ref_b.visual = trimesh.visual.ColorVisuals(mesh=ref_b, vertex_colors=colors)
    #         ref_b.export(ply_ref)
    #
    #     alt = _heightmap_volume(
    #         sample_b, sample_b_face_mask, ref_b, ref_face_mask, protrusion_threshold_vox)
    #
    #     if ply_sample is not None:
    #         _, blister_mesh = _combined_patch_volume(
    #             sample_b, sample_b_face_mask, ref_b, ref_face_mask)
    #         GREEN = np.array([0, 200, 0], dtype=np.uint8)
    #         green = np.tile(GREEN, (len(blister_mesh.vertices), 1))
    #         blister_mesh.visual = trimesh.visual.ColorVisuals(
    #             mesh=blister_mesh, vertex_colors=green)
    #         blister_mesh.export(str(Path(ply_sample).parent / f"blister_mesh_{label_b}.ply"))
    #
    #     alt2 = float(sample_b_face_mask.sum())
    #     return sample_vol, ref_vol, alt, alt2
    #
    # s2_vol, r2_vol, alt_b2, alt2_b2 = _compute_pair(
    #     sample_b=s2, sample_b_face_mask=s2_face_mask,
    #     sample_dists_to_other=s2_dists_to_s1, spike_tip=spike_tip_s2,
    #     ref_b=r2, ref_other=r1,
    #     ply_sample=ply_sample2, ply_ref=ply_ref2, label_b=2,
    # )
    #
    # ply_sample1 = str(Path(ply_sample2).parent / "sample_1_articular.ply") \
    #               if ply_sample2 is not None else None
    # ply_ref1    = str(Path(ply_ref2).parent / "ref_1_articular.ply") \
    #               if ply_ref2 is not None else None
    # s1_vol, r1_vol, alt_b1, alt2_b1 = _compute_pair(
    #     sample_b=s1, sample_b_face_mask=s1_face_mask,
    #     sample_dists_to_other=s1_dists_to_s2, spike_tip=spike_tip_s1,
    #     ref_b=r1, ref_other=r2,
    #     ply_sample=ply_sample1, ply_ref=ply_ref1, label_b=1,
    # )
    #
    # if spikes_match:
    #     sample_art_vol = s2_vol + s1_vol
    #     ref_art_vol    = r2_vol + r1_vol
    #     alt_vol        = alt_b2 + alt_b1
    #     alt_vol2       = alt2_b2 + alt2_b1
    #     return ref_art_vol, sample_art_vol, ref_art_vol - sample_art_vol, alt_vol, alt_vol2
    # else:
    #     return 0.0, 0.0, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Rerun visualization
# ---------------------------------------------------------------------------

def _plot_segment_profiles(all_seg_dir: Path, out_dir: Path) -> None:
    """No-op: bridge-profile plots are obsolete with atlas-based segmentation."""
    return


_SEG_PALETTE = np.array([
    [230, 100, 100],  # red
    [100, 195, 100],  # green
    [100, 130, 230],  # blue
    [220, 180,  50],  # yellow
    [170,  80, 190],  # purple
    [ 50, 200, 200],  # cyan
    [230, 140,  50],  # orange
    [150, 210,  90],  # lime
    [210,  90, 165],  # pink
], dtype=np.uint8)


def _visualize_segmentation_rerun(all_seg_dir: Path, rr) -> None:
    """
    For each atlas-segmented bone in all_seg_dir, log a distinct-color mesh.
    With atlas-based segmentation there are no bridges or cut planes to show.
    """
    for label in ("ref", "sample"):
        parts = sorted(all_seg_dir.glob(f"{label}_part_*.stl"))
        for i, part_path in enumerate(parts, start=1):
            seg = trimesh.load(str(part_path), force="mesh", process=False)
            seg.merge_vertices()

            seg_color = _SEG_PALETTE[(i - 1) % len(_SEG_PALETTE)]
            mesh_colors = np.tile(seg_color, (len(seg.vertices), 1))

            entity = f"segments/{label}/part_{i:02d}"
            rr.log(f"{entity}/mesh", rr.Mesh3D(
                vertex_positions=seg.vertices,
                triangle_indices=seg.faces,
                vertex_colors=mesh_colors,
            ))


def _visualize_rerun(out_dir: Path) -> None:
    import rerun as rr

    rr.init("uCT_pipeline", spawn=True)

    GREY = np.array([180, 180, 180], dtype=np.uint8)

    ref_1    = trimesh.load(str(out_dir / "ref_1.stl"),    force="mesh", process=False)
    sample_1 = trimesh.load(str(out_dir / "sample_1.stl"), force="mesh", process=False)
    ref_1.merge_vertices(); sample_1.merge_vertices()

    # Load PLY files saved by step 8 — colors already correctly computed there
    ref_2    = trimesh.load(str(out_dir / "ref_2_articular.ply"),    process=False)
    sample_2 = trimesh.load(str(out_dir / "sample_2_articular.ply"), process=False)

    grey_ref    = np.full((len(ref_1.vertices),    3), 180, dtype=np.uint8)
    grey_sample = np.full((len(sample_1.vertices), 3), 180, dtype=np.uint8)

    rr.log("ref/bone_1",    rr.Mesh3D(vertex_positions=ref_1.vertices,    triangle_indices=ref_1.faces,    vertex_colors=grey_ref))
    rr.log("ref/bone_2",    rr.Mesh3D(vertex_positions=ref_2.vertices,    triangle_indices=ref_2.faces,    vertex_colors=ref_2.visual.vertex_colors[:, :3]))
    rr.log("sample/bone_1", rr.Mesh3D(vertex_positions=sample_1.vertices, triangle_indices=sample_1.faces, vertex_colors=grey_sample))
    rr.log("sample/bone_2", rr.Mesh3D(vertex_positions=sample_2.vertices, triangle_indices=sample_2.faces, vertex_colors=sample_2.visual.vertex_colors[:, :3]))

    spike_tip_path = out_dir / "spike_tip.npy"
    if spike_tip_path.exists():
        spike_tip = np.load(str(spike_tip_path))
        rr.log("sample/spike_tip", rr.Points3D(
            positions=spike_tip.reshape(1, 3),
            colors=np.array([[0, 0, 255]], dtype=np.uint8),
            radii=np.array([0.5]),
        ))

    blister_path = out_dir / "blister_mesh.ply"
    if blister_path.exists():
        blister = trimesh.load(str(blister_path), process=False)
        green = np.tile(np.array([0, 200, 0], dtype=np.uint8), (len(blister.vertices), 1))
        rr.log("sample/blister", rr.Mesh3D(
            vertex_positions=blister.vertices,
            triangle_indices=blister.faces,
            vertex_colors=green,
        ))

    all_seg_dir = out_dir / "all_segments"
    if all_seg_dir.exists():
        print("  Loading segmentation debug overlay...")
        _visualize_segmentation_rerun(all_seg_dir, rr)

    print("  Rerun viewer launched — close it to continue.")


# ---------------------------------------------------------------------------
# Alignment quality check (used to gate Step 6)
# ---------------------------------------------------------------------------

def _best_mirror_iou(
    ref_meshes: list[trimesh.Trimesh],
    sample_meshes: list[trimesh.Trimesh],
    n_pts: int = 8_000,
    grid_size: int = 32,
) -> float:
    """
    Best-case IoU across 4 mirror configurations, measured the same way SMC does:
      - uniform surface samples (not all vertices)
      - ref-only bounding box with 15% padding
      - 32^3 voxel grid

    This is the correct gate metric for Step 6: it answers "does Step 6 improve
    the best orientation SMC could start from (including simple axis flips)?"
    Using raw vertex counts or a combined bounding box inflates IoU and gives
    false signals that Step 6 helped when it didn't.
    """
    ref_agg = trimesh.util.concatenate(ref_meshes); ref_agg.merge_vertices()
    smp_agg = trimesh.util.concatenate(sample_meshes); smp_agg.merge_vertices()

    rv, _ = trimesh.sample.sample_surface(ref_agg, n_pts, seed=0)
    rv = rv.astype(np.float32)
    pad = (rv.max(0) - rv.min(0)) * 0.15 + 1e-6
    vmin = rv.min(0) - pad
    vmax = rv.max(0) + pad + 1e-6

    def _vox(pts: np.ndarray) -> np.ndarray:
        crd = ((pts - vmin) / (vmax - vmin) * (grid_size - 1)).clip(0, grid_size - 1).astype(np.int32)
        g = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        g[crd[:, 0], crd[:, 1], crd[:, 2]] = True
        return g

    ref_grid = _vox(rv)
    agg_c = smp_agg.vertices.mean(0)
    best_iou = -1.0
    for axis in [None, 0, 1, 2]:
        verts = smp_agg.vertices.copy()
        if axis is not None:
            verts[:, axis] = 2.0 * agg_c[axis] - verts[:, axis]
        sv, _ = trimesh.sample.sample_surface(
            trimesh.Trimesh(vertices=verts, faces=smp_agg.faces, process=False),
            n_pts, seed=0)
        sv = sv.astype(np.float32)
        smp_grid = _vox(sv)
        iou = float((ref_grid & smp_grid).sum()) / float((ref_grid | smp_grid).sum() + 1)
        best_iou = max(best_iou, iou)
    return best_iou


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    ref_vox:          str,
    sample_vox:       str,
    atlas_npz:        str | None = None,
    bone_ids:         list[int] | None = None,
    output_dir:       str | None = None,
    run_smc:             bool = True,
    smc_restarts:        int = 5,
    alpha:               float = 0.0,
    volume_scale:        float = 1.0,
    sigma:               float = 3.0,
    decimate:            float = 0.05,
    taubin_iterations:   int = 100,
    neighborhood_radius:       float = 0.10,
    patch_radius:              float = 0.20,
    yz_eps:                    float = 0.005,
    protrusion_threshold_vox:  float = 1.0,
    step6_min_gain:            float = 0.05,
    visualize:                 bool = False,
) -> None:

    ref_vox    = Path(ref_vox)
    sample_vox = Path(sample_vox)
    atlas_npz  = Path(atlas_npz) if atlas_npz else Path(__file__).parent / "atlas.npz"
    bone_ids   = list(bone_ids) if bone_ids else [1, 2]


    out_dir = Path(output_dir) if output_dir \
              else Path(f"{ref_vox.stem}_vs_{sample_vox.stem}")
    all_seg_dir = out_dir / "all_segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_seg_dir.mkdir(exist_ok=True)
    print(f"Output directory: {out_dir}/\n")

    # ------------------------------------------------------------------ 1/5
    print("=" * 60)
    print("STEP 1/5  Convert VOX → STL")
    print("=" * 60)

    ref_stl    = out_dir / f"{ref_vox.stem}.stl"
    sample_stl = out_dir / f"{sample_vox.stem}.stl"

    mesh_kwargs = dict(
        gaussian_sigma    = sigma,
        decimate          = decimate,
        taubin_iterations = taubin_iterations,
        step_size         = 2,
    )

    print(f"\n[ref] {ref_vox.name}")
    ref_voxel_size = vox_to_stl(str(ref_vox), str(ref_stl), **mesh_kwargs)

    print(f"\n[sample] {sample_vox.name}")
    sample_voxel_size = vox_to_stl(str(sample_vox), str(sample_stl), **mesh_kwargs)

    # ------------------------------------------------------------------ 1b
    print("\n" + "=" * 60)
    print("STEP 1b   Orient meshes: longest PCA axis → X")
    print("=" * 60)

    for label, path in [("ref", ref_stl), ("sample", sample_stl)]:
        print(f"\n[{label}] {path.name}")
        m = trimesh.load(str(path), force="mesh", process=False)
        m.merge_vertices()
        m = align_longest_axis_to_x(m)
        m.export(str(path))
        print(f"  Longest axis → X  ({path.name} updated)")

    # ------------------------------------------------------------------ 2/5
    print("\n" + "=" * 60)
    print("STEP 2/5  Align sample → ref  (PCA + ICP)")
    print("=" * 60)

    sample_aligned_stl = out_dir / f"{sample_vox.stem}_aligned.stl"
    align_mesh(str(ref_stl), str(sample_stl), str(sample_aligned_stl))

    # ------------------------------------------------------------------ 3/5
    print("\n" + "=" * 60)
    print("STEP 3/5  Atlas-based segmentation (PCA + ICP + SITK affine)")
    print("=" * 60)
    print(f"\n  Atlas: {atlas_npz}")

    print(f"\n[ref]")
    ref_bones = _segment_via_atlas(ref_stl, atlas_npz)

    print(f"\n[sample]")
    sample_bones = _segment_via_atlas(sample_aligned_stl, atlas_npz)

    # ------------------------------------------------------------------ 4/5
    print("\n" + "=" * 60)
    print("STEP 4/5  Pick bones of interest")
    print("=" * 60)
    print(f"\n  Selected atlas bone IDs: {bone_ids}")

    missing_ref    = [b for b in bone_ids if b not in ref_bones]
    missing_sample = [b for b in bone_ids if b not in sample_bones]
    if missing_ref or missing_sample:
        print(f"  WARNING: missing bones — ref:{missing_ref}  sample:{missing_sample}")
        bone_ids = [b for b in bone_ids
                    if b in ref_bones and b in sample_bones]
        print(f"  Proceeding with: {bone_ids}")
    if len(bone_ids) < 2:
        raise RuntimeError(
            f"Fewer than 2 bones available in both ref and sample after "
            f"atlas segmentation. Available bones — ref: {sorted(ref_bones)}, "
            f"sample: {sorted(sample_bones)}"
        )

    # ------------------------------------------------------------------ 5/5
    print("\n" + "=" * 60)
    print("STEP 5/5  Save results")
    print("=" * 60)

    print(f"\n  Selected bones → {out_dir}/")
    # Rename selected bones to sequential 1..K (matching downstream expectations)
    for out_idx, atlas_i in enumerate(bone_ids, start=1):
        r = ref_bones[atlas_i]
        s = sample_bones[atlas_i]
        r_path = out_dir / f"ref_{out_idx}.stl"
        s_path = out_dir / f"sample_{out_idx}.stl"
        r.export(str(r_path))
        s.export(str(s_path))
        print(f"    atlas bone {atlas_i} → {r_path.name} ({len(r.faces):,} faces)  |  "
              f"{s_path.name} ({len(s.faces):,} faces)")

    # --- all atlas bones in all_segments/ -----------------------------------
    print(f"\n  All atlas bones → {all_seg_dir}/")
    for atlas_i in sorted(ref_bones):
        p = all_seg_dir / f"ref_part_{atlas_i}.stl"
        ref_bones[atlas_i].export(str(p))
        print(f"    {p.name}  ({len(ref_bones[atlas_i].faces):,} faces)")
    for atlas_i in sorted(sample_bones):
        p = all_seg_dir / f"sample_part_{atlas_i}.stl"
        sample_bones[atlas_i].export(str(p))
        print(f"    {p.name}  ({len(sample_bones[atlas_i].faces):,} faces)")

    # ------------------------------------------------------------------ 6/8
    sorted_atlas     = list(range(1, len(bone_ids) + 1))
    ref_seg_paths    = [out_dir / f"ref_{i}.stl"    for i in sorted_atlas]
    sample_seg_paths = [out_dir / f"sample_{i}.stl" for i in sorted_atlas]

    print("\n" + "=" * 60)
    print("STEP 6/8  Aggregate PCA+ICP pre-alignment (adaptive)")
    print("=" * 60)
    print("\n  One PCA+ICP transform over the concatenated ref/sample aggregates.")
    print(f"  Applied only if it improves aggregate IoU by >= {step6_min_gain:.2f}")
    print("  (set --step6-min-gain -1.0 to always apply, 1.0 to always skip).\n")

    ref_pre   = [trimesh.load(str(p), force="mesh", process=False) for p in ref_seg_paths]
    smp_pre   = [trimesh.load(str(p), force="mesh", process=False) for p in sample_seg_paths]
    for m in ref_pre + smp_pre:
        m.merge_vertices()

    # Use mirror-aware IoU (same method as SMC baseline screening) so the gain
    # reflects the best orientation SMC could reach, not just the no-flip case.
    # Using raw vertex counts or a combined bounding box (old _quick_iou) inflates
    # IoU and produces false gains that cause Step 6 to be applied when it shouldn't.
    baseline_iou = _best_mirror_iou(ref_pre, smp_pre)
    print(f"  Baseline best-mirror IoU (after step-2 PCA+ICP): {baseline_iou:.4f}")

    T_pre = aggregate_align_transform(ref_pre, smp_pre)

    smp_pre_t = [m.copy() for m in smp_pre]
    for m in smp_pre_t:
        m.apply_transform(T_pre)
    step6_iou = _best_mirror_iou(ref_pre, smp_pre_t)
    gain = step6_iou - baseline_iou
    print(f"  Post-step6 best-mirror IoU:                       {step6_iou:.4f}  "
          f"(gain: {gain:+.4f})")

    if gain >= step6_min_gain:
        print(f"  Step 6 applied  (gain {gain:+.4f} >= threshold {step6_min_gain:.2f})")
        for smp_p, smp_mesh in zip(sample_seg_paths, smp_pre_t):
            smp_mesh.export(str(smp_p))
            print(f"  Saved pre-aligned: {smp_p.name}")
    else:
        print(f"  Step 6 skipped  (gain {gain:+.4f} < threshold {step6_min_gain:.2f})")
        print("  Keeping step-2 alignment for SMC.")

    # ------------------------------------------------------------------ 7/8
    if run_smc:
        import jax
        gc.collect(); jax.clear_caches()
        print("\n" + "=" * 60)
        print("STEP 7/8  SMC aggregate alignment")
        print("=" * 60)
        print("\n  Treating all atlas-matched sample segments as one aggregate.")
        print("  One shared transform maximises aggregate IoU with ref aggregate.")
        print("  The same transform (with aggregate centroid as pivot) is applied")
        print("  to each sample segment individually → inter-bone structure preserved.\n")

        smc_align_aggregate(
            ref_paths    = [str(p) for p in ref_seg_paths],
            sample_paths = [str(p) for p in sample_seg_paths],
            output_dir   = out_dir,
            seed         = 0,
            n_restarts   = smc_restarts,
        )

        gc.collect(); jax.clear_caches()
        print("\n" + "=" * 60)
        print("STEP 7b/8  SMC per-bone refinement  (full 9-DOF)")
        print("=" * 60)
        print("\n  Refines each bone pair individually after aggregate alignment.")
        print("  Full translation + rotation + scale search per bone.")
        print("  Each bone pivots on its own centroid.\n")

        smc_align_per_bone(
            ref_paths    = [str(p) for p in ref_seg_paths],
            sample_paths = [str(p) for p in sample_seg_paths],
            output_dir   = out_dir,
            seed         = 0,
            n_restarts   = smc_restarts,
        )

    else:
        print("\n  (SMC step skipped — use without --no-smc to enable)")

    # ------------------------------------------------------------------ 8/8
    print("\n" + "=" * 60)
    print("STEP 8/8  Injection score  (protrusion-based articular patch)")
    print("=" * 60)
    print(f"\n  Method: find the most protruding vertex on sample_2 toward sample_1")
    print(f"  via local x-deviation (neighborhood_radius={neighborhood_radius:.2f}),")
    print(f"  select the patch within patch_radius={patch_radius:.2f} of that vertex,")
    print(f"  then compute Σ d·a (distance to opposing bone × vertex area).")
    print(f"  ref_2 patch matched by y-z nearest neighbor to sample_2 patch.")
    print(f"  Positive = sample gap narrower than ref = bone grew in.\n")

    ref_art_vol, sample_art_vol, injection_vol, alt_vol, alt_vol2 = _injection_volume_protrusion(
        sample2_path              = str(out_dir / "sample_2.stl"),
        sample1_path              = str(out_dir / "sample_1.stl"),
        ref2_path                 = str(out_dir / "ref_2.stl"),
        ref1_path                 = str(out_dir / "ref_1.stl"),
        neighborhood_radius       = neighborhood_radius,
        patch_radius              = patch_radius,
        yz_eps                    = yz_eps,
        protrusion_threshold_vox  = protrusion_threshold_vox,
        ply_sample2               = str(out_dir / "sample_2_articular.ply"),
        ply_ref2                  = str(out_dir / "ref_2_articular.ply"),
    )

    injection_volume_raw      = injection_vol  * volume_scale
    alt_injection_volume_raw  = alt_vol        * volume_scale
    alt_injection_volume2_raw = alt_vol2       * volume_scale

    print(f"\n  ref    articular volume (Σd·a): {ref_art_vol:+.4f}")
    print(f"  sample articular volume (Σd·a): {sample_art_vol:+.4f}")
    print(f"\n  Primary  (Σd·a delta):                     {injection_volume_raw:+.6f}")
    print(f"  Alt-1    (height-map, >{protrusion_threshold_vox:.1f} vox threshold): {alt_injection_volume_raw:+.6f}  ← closest to Dragonfly")
    print(f"  Alt-2    (N_triangles × voxel_size³):       {alt_injection_volume2_raw:+.6f}")

    inj_txt = out_dir / "injection_volume.txt"
    inj_txt.write_text(
        f"# protrusion-based articular patch method\n"
        f"# positive = bone grew into joint in sample\n"
        f"\n"
        f"neighborhood_radius:            {neighborhood_radius}\n"
        f"patch_radius:                   {patch_radius}\n"
        f"yz_eps:                         {yz_eps}\n"
        f"protrusion_threshold_vox:       {protrusion_threshold_vox}\n"
        f"volume_scale:                   {volume_scale}\n"
        f"\n"
        f"sample_articular_volume:        {sample_art_vol:.6f}\n"
        f"ref_articular_volume:           {ref_art_vol:.6f}\n"
        f"\n"
        f"injection_volume_raw:           {injection_volume_raw:+.6f}  # primary: (ref-sample)Σd·a × scale\n"
        f"alt_injection_volume_raw:       {alt_injection_volume_raw:+.6f}  # alt-1: height-map × scale  (closest to Dragonfly)\n"
        f"alt_injection_volume2_raw:      {alt_injection_volume2_raw:+.6f}  # alt-2: N_triangles × voxel_size³\n"
    )
    print(f"\n  Saved {inj_txt.name}")

    if visualize:
        print("\n" + "=" * 60)
        print("Visualization  (rerun-sdk + matplotlib)")
        print("=" * 60)
        _visualize_rerun(out_dir)
        _plot_segment_profiles(out_dir / "all_segments", out_dir)

    print(f"\nDone.")
    print(f"  {len(bone_ids)} selected bones → {out_dir}/")
    print(f"  {len(ref_bones) + len(sample_bones)} total atlas bones → {all_seg_dir}/")
    print(f"  Primary  (Σd·a delta):          {injection_volume_raw:+.6f}  → {inj_txt}")
    print(f"  Alt-1    (height-map):           {alt_injection_volume_raw:+.6f}  → {inj_txt}")
    print(f"  Alt-2    (N×vox³):               {alt_injection_volume2_raw:+.6f}  → {inj_txt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VOX → mesh → align → segment → atlas-guided labelling."
    )
    parser.add_argument("ref",    nargs="?", help="Reference VOX file")
    parser.add_argument("sample", nargs="?", help="Sample VOX file")
    parser.add_argument(
        "--atlas-npz", default=None, metavar="PATH",
        help="Path to atlas.npz (default: atlas.npz next to pipeline.py). "
             "The atlas defines which voxel labels correspond to which bone "
             "(1..N) for registration-based segmentation.",
    )
    parser.add_argument(
        "--bones", type=int, nargs="+", default=[1, 2], metavar="ID",
        help="Atlas bone IDs of interest, in order. The first becomes "
             "ref_1.stl/sample_1.stl, the second ref_2.stl/sample_2.stl, etc. "
             "(default: 1 2)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: <ref>_vs_<sample>/)",
    )
    parser.add_argument(
        "--no-smc", action="store_true",
        help="Skip SMC aggregate alignment step (step 7/8)",
    )
    parser.add_argument(
        "--smc-restarts", type=int, default=5, metavar="N",
        help="Number of IS restarts for SMC alignment (default: 5)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, metavar="A",
        help="Alpha value for alpha-shape cartilage volume (default: 0 = convex hull). "
             "Increase for tighter fit around bone surfaces.",
    )
    parser.add_argument(
        "--volume-scale", type=float, default=1.0, metavar="S",
        help="Multiply all cartilage/injection volumes by this factor. "
             "Use 1e-3 for cm³, 1e-9 for m³ (default: 1.0 = raw mm³). "
             "Check the voxel size printed in step 8 to verify STL units.",
    )
    parser.add_argument(
        "--sigma", type=float, default=3.0, metavar="S",
        help="Gaussian pre-smooth σ in voxels for VOX→STL (default: 3.0; 0 to disable)",
    )
    parser.add_argument(
        "--decimate", type=float, default=0.05, metavar="R",
        help="Fraction of faces to keep after decimation (default: 0.05 = keep 5%%)",
    )
    parser.add_argument(
        "--laplacian", type=int, default=100, metavar="N",
        help="Taubin smoothing iterations for VOX→STL (default: 100; 0 to disable)",
    )
    parser.add_argument(
        "--neighborhood-radius", type=float, default=0.10, metavar="R",
        help="Fraction of bone_2's longest axis used as y-z radius when computing "
             "local x mean for protrusion detection (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--patch-radius", type=float, default=0.20, metavar="R",
        help="Fraction of bone_2's longest axis used as 3D radius to select the "
             "articular patch around the most-protruding vertex (default: 0.20 = 20%%)",
    )
    parser.add_argument(
        "--yz-eps", type=float, default=0.005, metavar="E",
        help="Fraction of bone_2's longest axis used as y-z cell size for "
             "deduplication: within each cell only the facing-side (lowest-x-toward-"
             "bone_1) vertex is used, discarding back-face duplicates (default: 0.005)",
    )
    parser.add_argument(
        "--protrusion-threshold", type=float, default=1.0, metavar="V",
        help="Minimum protrusion depth in voxels to count toward Alt-1 height-map "
             "volume (strips subvoxel noise; default: 1.0 vox ≈ 50 µm)",
    )
    parser.add_argument(
        "--step6-min-gain", type=float, default=0.05, metavar="G",
        help="Minimum best-mirror IoU gain required for step 6 aggregate PCA+ICP to "
             "be applied. Compared against all 4 axis-flip baselines (same as SMC "
             "mirror screening). Set to -1.0 to always apply, 1.0 to always skip. "
             "(default: 0.05)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch rerun-sdk viewer showing ref_1, ref_2, sample_1, sample_2 "
             "with bone-2 articular patch colored red.",
    )
    parser.add_argument(
        "--visualize-only", action="store_true",
        help="Skip all processing; only re-render rerun + matplotlib for an existing "
             "results directory (requires -o OUTPUT_DIR).",
    )
    args = parser.parse_args()

    if args.visualize_only:
        if not args.output_dir:
            print("ERROR: --visualize-only requires -o OUTPUT_DIR")
            sys.exit(1)
        out = Path(args.output_dir)
        print(f"Visualizing existing results: {out}/")
        _visualize_rerun(out)
        _plot_segment_profiles(out / "all_segments", out)
        return

    if not args.ref or not args.sample:
        print("ERROR: ref and sample VOX paths are required (unless --visualize-only)")
        sys.exit(1)

    run_pipeline(args.ref, args.sample,
                 atlas_npz=args.atlas_npz, bone_ids=args.bones,
                 output_dir=args.output_dir,
                 run_smc=not args.no_smc, smc_restarts=args.smc_restarts,
                 alpha=args.alpha, volume_scale=args.volume_scale,
                 sigma=args.sigma, decimate=args.decimate,
                 taubin_iterations=args.laplacian,
                 neighborhood_radius=args.neighborhood_radius,
                 patch_radius=args.patch_radius,
                 yz_eps=args.yz_eps,
                 protrusion_threshold_vox=args.protrusion_threshold,
                 step6_min_gain=args.step6_min_gain,
                 visualize=args.visualize)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        print("Usage: python pipeline.py ref.vox sample.vox "
              "[--bones 1 2] [-o results/]")
        sys.exit(0)
    main()
