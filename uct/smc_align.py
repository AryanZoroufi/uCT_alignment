"""
GenJAX importance sampling to refine mesh alignment via IoU maximisation.

Algorithm — 4-stage factorised SMC (full, used in aggregate mode):
  Stage 1: search translation only         (3-D, rotation=0, scale=1)
  Stage 2: search rotation only            (3-D, fix t from stage 1)
  Stage 3: search per-axis scale only      (3-D, fix t+r from stages 1-2)
  Stage 4: joint 9-D fine-tuning           (tight priors around stages 1-3 MAP)

Algorithm — 3-stage restricted SMC (used in per-bone refinement mode):
  Stage 1: search x-translation only       (ty=tz=0, all rotations=0, scale=1)
  Stage 2: search per-axis scale only      (fix tx from stage 1)
  Stage 3: joint fine-tune tx+scale        (tight priors, rotations still locked)

Factorisation reduces effective dimensionality from 9 → 3 per stage, making
importance sampling tractable with ~3000 particles per stage.

Two-stage pipeline (recommended workflow):
  1. Aggregate SMC (full 9-DOF): all ref/sample bones concatenated into aggregate
     meshes. One shared transform found; applied to each sample bone using the
     AGGREGATE centroid as pivot — preserves relative inter-bone structure.
  2. Per-bone SMC (tx + scale only): each matched bone pair refined individually.
     Only x-translation and per-axis scale are searched; y/z translation and
     all rotations are locked at 0 so the aggregate alignment is not disturbed.
     Each bone uses its OWN centroid as pivot.

IoU is computed over voxelised uniform surface point clouds (N_SURFACE_PTS points
sampled with seed=0 → deterministic).  Multiple restarts (--restarts N) run the
stage sequence with different random keys and take the best result.

Rotation convention (ZYX Euler):
  rx  — around X axis (YZ-plane rotation) — LARGE prior ±90°
  ry  — around Y axis (XZ-plane rotation) — moderate ±45°
  rz  — around Z axis (XY-plane rotation) — moderate ±45°

Usage (single pair):
    python smc_align.py ref_1.stl sample_1.stl -o sample_1_smc.stl

Usage (aggregate — preserves inter-bone structure):
    python smc_align.py --ref ref_1.stl ref_2.stl \\
                        --sample sample_1.stl sample_2.stl \\
                        -o results/
    # writes results/sample_1.stl and results/sample_2.stl
"""

import os
import sys
import gc
import argparse
from pathlib import Path

# Use the platform (CUDA) allocator instead of XLA's BFC allocator.
# BFC fragments between stages; the CUDA allocator handles this gracefully.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import trimesh
import jax
import jax.numpy as jnp
import genjax
from genjax.inference.smc import ImportanceK
from genjax import ChoiceMapBuilder as C

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
RESOLUTION      = 32      # voxel grid side (R^3 = 32768 cells)
N_PARTICLES     = 3_000   # particles per stage
N_SURFACE_PTS   = 5_000   # surface samples per mesh (uniform, seed=0 → deterministic)
SIGMA_OBS       = 0.05    # IoU likelihood bandwidth: obs ~ N(iou, σ) at 1.0
EPS             = 0.01    # half-width for "fixed" parameters in non-searched dims

# Stage-1 translation search bounds (mm, centred at 0)
T_RANGE = 10.0

# Stage-2 rotation bounds (radians)
RX_RANGE = np.pi / 2   # YZ plane — largest range
RY_RANGE = np.pi / 4   # XZ plane
RZ_RANGE = np.pi / 4   # XY plane

# Stage-3 scale bounds (around 1.0)
S_RANGE = 0.20

# Stage-4 joint fine-tuning half-widths
S4_HALF = np.array([2.0, 2.0, 2.0,               # translation (mm)
                    np.pi/10, np.pi/16, np.pi/16, # rotation (rad)
                    0.06, 0.06, 0.06],             # scale
                   dtype=np.float32)

# ---------------------------------------------------------------------------
# Rotation matrices (JAX)
# ---------------------------------------------------------------------------

def _Rx(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _Ry(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _Rz(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_matrix(rx, ry, rz):
    """ZYX convention: Rz @ Ry @ Rx (Rx applied first)."""
    return _Rz(rz) @ _Ry(ry) @ _Rx(rx)

# ---------------------------------------------------------------------------
# Point transformation (JAX)
# ---------------------------------------------------------------------------

def transform_pts(pts, centroid, t, rx, ry, rz, sx, sy, sz):
    """Centre → scale → rotate → translate back → offset."""
    c = (pts - centroid) * jnp.array([sx, sy, sz])
    c = (rotation_matrix(rx, ry, rz) @ c.T).T
    return c + centroid + t

# ---------------------------------------------------------------------------
# Voxelisation & IoU (JAX, RESOLUTION is a compile-time constant)
# ---------------------------------------------------------------------------

def voxelize(pts, vmin, vmax):
    R   = RESOLUTION
    crd = (pts - vmin) / (vmax - vmin + 1e-8) * (R - 1)
    crd = jnp.clip(crd, 0, R - 1).astype(jnp.int32)
    flat = crd[:, 0] * R * R + crd[:, 1] * R + crd[:, 2]
    return jnp.zeros(R**3, dtype=jnp.float32).at[flat].set(1.0).reshape(R, R, R)

def iou_grids(a, b):
    return jnp.sum(a * b) / (jnp.sum(jnp.clip(a + b, 0.0, 1.0)) + 1e-8)

# ---------------------------------------------------------------------------
# Generic 9-param GenJAX model
# ---------------------------------------------------------------------------

@genjax.gen
def alignment_model(sample_pts, centroid, ref_grid, vmin, vmax, lo, hi):
    """
    Sample 9 params uniformly from [lo, hi].
    Param layout: [tx, ty, tz, rx, ry, rz, sx, sy, sz]
    Condition on iou_obs = 1.0 to up-weight high-IoU configurations.
    """
    tx = genjax.uniform(lo[0], hi[0]) @ "tx"
    ty = genjax.uniform(lo[1], hi[1]) @ "ty"
    tz = genjax.uniform(lo[2], hi[2]) @ "tz"
    rx = genjax.uniform(lo[3], hi[3]) @ "rx"
    ry = genjax.uniform(lo[4], hi[4]) @ "ry"
    rz = genjax.uniform(lo[5], hi[5]) @ "rz"
    sx = genjax.uniform(lo[6], hi[6]) @ "sx"
    sy = genjax.uniform(lo[7], hi[7]) @ "sy"
    sz = genjax.uniform(lo[8], hi[8]) @ "sz"

    t_vec       = jnp.array([tx, ty, tz])
    transformed = transform_pts(sample_pts, centroid, t_vec, rx, ry, rz, sx, sy, sz)
    sample_grid = voxelize(transformed, vmin, vmax)
    iou         = iou_grids(sample_grid, ref_grid)

    _ = genjax.normal(iou, SIGMA_OBS) @ "iou_obs"
    return iou

# ---------------------------------------------------------------------------
# Run one importance sampling stage
# ---------------------------------------------------------------------------

def _run_stage(key, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax,
               lo, hi, n_particles=N_PARTICLES):
    lo_j = jnp.array(lo, dtype=jnp.float32)
    hi_j = jnp.array(hi, dtype=jnp.float32)
    target  = genjax.Target(alignment_model,
                            (sample_pts_j, centroid_j, ref_grid_j, vmin, vmax, lo_j, hi_j),
                            C["iou_obs"].set(1.0))
    result = ImportanceK(target=target, k_particles=n_particles).run_smc(key)
    jax.effects_barrier()   # wait for GPU ops to complete
    return result

def _map_params(particles) -> dict:
    """Extract best-particle params and immediately discard the particle object."""
    best_idx = int(jnp.argmax(particles.log_weights))
    ch = particles.particles.get_choices()
    params = {k: float(ch[k][best_idx])
              for k in ("tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz")}
    del particles          # release GPU arrays held in the trace
    gc.collect()
    jax.clear_caches()     # free XLA compilation caches
    return params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou_for(params, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax) -> float:
    t  = jnp.array([params["tx"], params["ty"], params["tz"]])
    tr = transform_pts(sample_pts_j, centroid_j, t,
                       params["rx"], params["ry"], params["rz"],
                       params["sx"], params["sy"], params["sz"])
    return float(iou_grids(voxelize(tr, vmin, vmax), ref_grid_j))

def _print_params(tag, params, iou_val):
    rd = np.degrees
    print(f"  [{tag}] IoU={iou_val:.4f} | "
          f"t=({params['tx']:.2f},{params['ty']:.2f},{params['tz']:.2f})mm | "
          f"r=({rd(params['rx']):.1f}°,{rd(params['ry']):.1f}°,{rd(params['rz']):.1f}°) | "
          f"s=({params['sx']:.3f},{params['sy']:.3f},{params['sz']:.3f})")

def _mirror_verts(verts_np: np.ndarray, centroid_np: np.ndarray,
                  axis: int) -> np.ndarray:
    """Reflect vertices around centroid along the given axis (0=x,1=y,2=z)."""
    v = verts_np.copy()
    v[:, axis] = 2.0 * centroid_np[axis] - v[:, axis]
    return v

def _apply_params(verts_np: np.ndarray, centroid_np: np.ndarray,
                  params: dict) -> np.ndarray:
    """Apply transform to a vertex array using the given centroid as pivot."""
    verts_j    = jnp.array(verts_np,    dtype=jnp.float32)
    centroid_j = jnp.array(centroid_np, dtype=jnp.float32)
    t = jnp.array([params["tx"], params["ty"], params["tz"]])
    new_v = transform_pts(verts_j, centroid_j, t,
                          params["rx"], params["ry"], params["rz"],
                          params["sx"], params["sy"], params["sz"])
    return np.array(new_v)

# ---------------------------------------------------------------------------
# 4-stage IS run (one restart)
# ---------------------------------------------------------------------------

def _run_4stage(key, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax,
                run_label: str = "") -> tuple[dict, float]:
    """Run the 4-stage factorised IS; return (best_params, best_iou)."""
    tag = f" [{run_label}]" if run_label else ""

    print(f"\nStage 1/4{tag} — translation ({N_PARTICLES} particles) ...")
    lo1 = np.array([-T_RANGE, -T_RANGE, -T_RANGE,
                    -EPS, -EPS, -EPS,
                     1-EPS, 1-EPS, 1-EPS], dtype=np.float32)
    hi1 = np.array([ T_RANGE,  T_RANGE,  T_RANGE,
                     EPS,  EPS,  EPS,
                     1+EPS, 1+EPS, 1+EPS], dtype=np.float32)
    key, sk = jax.random.split(key)
    p1 = _run_stage(sk, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax, lo1, hi1)
    m1 = _map_params(p1);  del p1
    iou1 = _iou_for(m1, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax)
    _print_params("stage 1", m1, iou1)

    print(f"Stage 2/4{tag} — rotation ({N_PARTICLES} particles) ...")
    lo2 = np.array([m1["tx"]-EPS, m1["ty"]-EPS, m1["tz"]-EPS,
                    -RX_RANGE, -RY_RANGE, -RZ_RANGE,
                     1-EPS, 1-EPS, 1-EPS], dtype=np.float32)
    hi2 = np.array([m1["tx"]+EPS, m1["ty"]+EPS, m1["tz"]+EPS,
                     RX_RANGE,  RY_RANGE,  RZ_RANGE,
                     1+EPS, 1+EPS, 1+EPS], dtype=np.float32)
    key, sk = jax.random.split(key)
    p2 = _run_stage(sk, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax, lo2, hi2)
    m2 = _map_params(p2);  del p2
    iou2 = _iou_for(m2, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax)
    _print_params("stage 2", m2, iou2)

    print(f"Stage 3/4{tag} — scale ({N_PARTICLES} particles) ...")
    lo3 = np.array([m2["tx"]-EPS, m2["ty"]-EPS, m2["tz"]-EPS,
                    m2["rx"]-EPS, m2["ry"]-EPS, m2["rz"]-EPS,
                    1-S_RANGE, 1-S_RANGE, 1-S_RANGE], dtype=np.float32)
    hi3 = np.array([m2["tx"]+EPS, m2["ty"]+EPS, m2["tz"]+EPS,
                    m2["rx"]+EPS, m2["ry"]+EPS, m2["rz"]+EPS,
                    1+S_RANGE, 1+S_RANGE, 1+S_RANGE], dtype=np.float32)
    key, sk = jax.random.split(key)
    p3 = _run_stage(sk, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax, lo3, hi3)
    m3 = _map_params(p3);  del p3
    iou3 = _iou_for(m3, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax)
    _print_params("stage 3", m3, iou3)

    print(f"Stage 4/4{tag} — joint fine-tune ({N_PARTICLES} particles) ...")
    center4 = np.array([m3["tx"], m3["ty"], m3["tz"],
                        m3["rx"], m3["ry"], m3["rz"],
                        m3["sx"], m3["sy"], m3["sz"]], dtype=np.float32)
    lo4 = center4 - S4_HALF
    hi4 = center4 + S4_HALF
    key, sk = jax.random.split(key)
    p4 = _run_stage(sk, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax, lo4, hi4)
    m4 = _map_params(p4);  del p4
    iou4 = _iou_for(m4, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax)
    _print_params("stage 4", m4, iou4)

    best_iou, best_params = iou1, m1
    for iou_s, m_s in [(iou2, m2), (iou3, m3), (iou4, m4)]:
        if iou_s > best_iou:
            best_iou, best_params = iou_s, m_s
    return best_params, best_iou


# ---------------------------------------------------------------------------
# 3-stage restricted IS: x-translation + scale only (no rotation, no ty/tz)
# ---------------------------------------------------------------------------

def _run_3stage_tx_scale(key, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax,
                         run_label: str = "") -> tuple[dict, float]:
    """
    Search only tx (x-translation) and sx/sy/sz (per-axis scale).
    ty, tz, rx, ry, rz are all locked to 0 / 1.
    """
    tag = f" [{run_label}]" if run_label else ""

    print(f"\nStage 1/3{tag} — x-translation ({N_PARTICLES} particles) ...")
    lo1 = np.array([-T_RANGE, -EPS, -EPS,  -EPS, -EPS, -EPS,
                     1-EPS, 1-EPS, 1-EPS], dtype=np.float32)
    hi1 = np.array([ T_RANGE,  EPS,  EPS,   EPS,  EPS,  EPS,
                     1+EPS, 1+EPS, 1+EPS], dtype=np.float32)
    key, sk = jax.random.split(key)
    p1 = _run_stage(sk, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax, lo1, hi1)
    m1 = _map_params(p1); del p1
    iou1 = _iou_for(m1, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax)
    _print_params("stage 1", m1, iou1)

    print(f"Stage 2/3{tag} — scale ({N_PARTICLES} particles) ...")
    lo2 = np.array([m1["tx"]-EPS, -EPS, -EPS,  -EPS, -EPS, -EPS,
                    1-S_RANGE, 1-S_RANGE, 1-S_RANGE], dtype=np.float32)
    hi2 = np.array([m1["tx"]+EPS,  EPS,  EPS,   EPS,  EPS,  EPS,
                    1+S_RANGE, 1+S_RANGE, 1+S_RANGE], dtype=np.float32)
    key, sk = jax.random.split(key)
    p2 = _run_stage(sk, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax, lo2, hi2)
    m2 = _map_params(p2); del p2
    iou2 = _iou_for(m2, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax)
    _print_params("stage 2", m2, iou2)

    print(f"Stage 3/3{tag} — joint fine-tune tx+scale ({N_PARTICLES} particles) ...")
    S3_HALF = np.array([2.0, EPS, EPS,  EPS, EPS, EPS,  0.06, 0.06, 0.06],
                       dtype=np.float32)
    center3 = np.array([m2["tx"], 0., 0.,  0., 0., 0.,
                        m2["sx"], m2["sy"], m2["sz"]], dtype=np.float32)
    lo3 = center3 - S3_HALF
    hi3 = center3 + S3_HALF
    key, sk = jax.random.split(key)
    p3 = _run_stage(sk, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax, lo3, hi3)
    m3 = _map_params(p3); del p3
    iou3 = _iou_for(m3, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax)
    _print_params("stage 3", m3, iou3)

    best_iou, best_params = iou1, m1
    for iou_s, m_s in [(iou2, m2), (iou3, m3)]:
        if iou_s > best_iou:
            best_iou, best_params = iou_s, m_s
    return best_params, best_iou


# ---------------------------------------------------------------------------
# Core IS loop (shared by all modes)
# ---------------------------------------------------------------------------

def _run_is(ref_mesh: trimesh.Trimesh,
            sample_mesh: trimesh.Trimesh,
            seed: int = 0,
            n_restarts: int = 3,
            stage_fn=None) -> tuple[dict, float, float]:
    """
    Run multi-restart IS on a (ref, sample) mesh pair.

    Returns
    -------
    best_params   : dict with tx/ty/tz/rx/ry/rz/sx/sy/sz
    best_iou      : IoU achieved by best_params
    baseline_iou  : IoU with no transform applied
    best_mirror   : None, or 0/1/2 (axis that was mirrored before IS)
    """
    if stage_fn is None:
        stage_fn = _run_4stage

    rv_sub, _ = trimesh.sample.sample_surface(ref_mesh, N_SURFACE_PTS, seed=0)
    rv_sub = rv_sub.astype(np.float32)
    ref_pts_j = jnp.array(rv_sub, dtype=jnp.float32)

    rpts = np.array(ref_mesh.vertices, dtype=np.float32)
    pad  = (rpts.max(0) - rpts.min(0)) * 0.15
    vmin = jnp.array(rpts.min(0) - pad)
    vmax = jnp.array(rpts.max(0) + pad)
    ref_grid_j = voxelize(ref_pts_j, vmin, vmax)

    # ── Mirror screening: try identity + 3 axis flips, pick best baseline IoU
    sample_verts_np  = np.array(sample_mesh.vertices, dtype=np.float32)
    sample_centroid_np = sample_verts_np.mean(0)

    print(f"  Screening mirror configurations ...")
    best_mirror_iou  = -1.0
    best_mirror_axis = None           # None = no flip
    mirror_label     = "none"

    for axis, label in [(None, "none"), (0, "flip-x"), (1, "flip-y"), (2, "flip-z")]:
        v = (_mirror_verts(sample_verts_np, sample_centroid_np, axis)
             if axis is not None else sample_verts_np)
        sv, _ = trimesh.sample.sample_surface(
            trimesh.Trimesh(vertices=v, faces=sample_mesh.faces), N_SURFACE_PTS, seed=0)
        iou_m = float(iou_grids(voxelize(jnp.array(sv.astype(np.float32)), vmin, vmax),
                                ref_grid_j))
        print(f"    {label:8s}  baseline IoU = {iou_m:.4f}")
        if iou_m > best_mirror_iou:
            best_mirror_iou  = iou_m
            best_mirror_axis = axis
            mirror_label     = label

    print(f"  Best mirror: {mirror_label}  (IoU={best_mirror_iou:.4f})")

    # Apply chosen mirror to sample for all subsequent IS runs
    if best_mirror_axis is not None:
        sv_np = _mirror_verts(sample_verts_np, sample_centroid_np, best_mirror_axis)
        s_mesh_used = trimesh.Trimesh(vertices=sv_np, faces=sample_mesh.faces)
        s_mesh_used.merge_vertices()
    else:
        s_mesh_used = sample_mesh

    sv_sub, _ = trimesh.sample.sample_surface(s_mesh_used, N_SURFACE_PTS, seed=0)
    sv_sub = sv_sub.astype(np.float32)
    sample_pts_j = jnp.array(sv_sub, dtype=jnp.float32)
    centroid_j   = jnp.array(s_mesh_used.vertices, dtype=jnp.float32).mean(axis=0)

    baseline_iou = best_mirror_iou
    print(f"  Baseline IoU (no transform): {baseline_iou:.4f}")

    best_iou    = baseline_iou
    best_params: dict = {"tx": 0., "ty": 0., "tz": 0.,
                         "rx": 0., "ry": 0., "rz": 0.,
                         "sx": 1., "sy": 1., "sz": 1.}
    best_restart = "baseline"

    for r in range(n_restarts):
        print(f"\n{'='*50}")
        print(f"Restart {r+1}/{n_restarts}  (seed={seed + r})")
        print(f"{'='*50}")
        key = jax.random.key(seed + r)
        params_r, iou_r = stage_fn(key, sample_pts_j, centroid_j,
                                   ref_grid_j, vmin, vmax,
                                   run_label=f"restart {r+1}")
        print(f"\n  Restart {r+1} best IoU: {iou_r:.4f}")
        if iou_r > best_iou:
            best_iou, best_params, best_restart = iou_r, params_r, f"restart {r+1}"

    print(f"\n{'='*50}")
    print(f"  Winning: {best_restart}  mirror={mirror_label}  IoU={best_iou:.4f}  "
          f"(Δ vs baseline: {best_iou - baseline_iou:+.4f})")

    return best_params, best_iou, baseline_iou, best_mirror_axis


# ---------------------------------------------------------------------------
# Public API — single pair
# ---------------------------------------------------------------------------

def smc_align(ref_path: str, sample_path: str, output_path: str,
              seed: int = 0, n_restarts: int = 3) -> tuple[float, dict]:
    """Align one sample STL to one reference STL."""
    ref_path, sample_path, output_path = (
        Path(ref_path), Path(sample_path), Path(output_path))

    print(f"\nLoading meshes ...")
    ref    = trimesh.load(str(ref_path),    force="mesh", process=False)
    sample = trimesh.load(str(sample_path), force="mesh", process=False)
    ref.merge_vertices(); sample.merge_vertices()
    print(f"  Ref:    {len(ref.vertices):,} verts  |  "
          f"Sample: {len(sample.vertices):,} verts")
    print(f"  Surface pts (each): {N_SURFACE_PTS:,}")

    best_params, best_iou, _, mirror_axis = _run_is(ref, sample, seed=seed, n_restarts=n_restarts)

    centroid_np = np.array(sample.vertices, dtype=np.float32).mean(0)
    verts_np    = np.array(sample.vertices, dtype=np.float32)
    if mirror_axis is not None:
        verts_np = _mirror_verts(verts_np, centroid_np, mirror_axis)
    new_v = _apply_params(verts_np, centroid_np, best_params)
    result = sample.copy()
    result.vertices = new_v
    result.export(str(output_path))
    print(f"  Saved → {output_path}  ({output_path.stat().st_size/1e6:.1f} MB)")
    return best_iou, best_params


# ---------------------------------------------------------------------------
# Public API — aggregate (N bones, preserves inter-bone structure)
# ---------------------------------------------------------------------------

def smc_align_aggregate(
    ref_paths:    list,
    sample_paths: list,
    output_dir:   str | Path,
    seed:         int = 0,
    n_restarts:   int = 3,
) -> tuple[float, dict]:
    """
    Align N sample bones to N reference bones while preserving relative structure.

    One rigid+scale transform is found for the concatenated aggregate meshes,
    then applied to each sample bone individually using the AGGREGATE centroid
    as the rotation/scale pivot.  This keeps the inter-bone distances and
    orientations intact.

    Outputs are written to output_dir/<original_sample_filename>.

    Parameters
    ----------
    ref_paths, sample_paths : ordered lists of STL paths; index i is one bone pair
    output_dir              : where to save transformed sample bones
    seed, n_restarts        : IS random seed and number of independent restarts

    Returns
    -------
    best_iou   : aggregate IoU of the winning transform
    best_params: winning transform parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(ref_paths)
    assert n == len(sample_paths), "ref_paths and sample_paths must have the same length"

    print(f"\nLoading {n} ref + {n} sample meshes ...")
    ref_meshes, sample_meshes = [], []
    for p in ref_paths:
        m = trimesh.load(str(p), force="mesh", process=False)
        m.merge_vertices()
        ref_meshes.append(m)
        print(f"  ref    {Path(p).name}: {len(m.vertices):,} verts")
    for p in sample_paths:
        m = trimesh.load(str(p), force="mesh", process=False)
        m.merge_vertices()
        sample_meshes.append(m)
        print(f"  sample {Path(p).name}: {len(m.vertices):,} verts")

    # Build aggregate meshes (concatenate all bones into one mesh per side)
    ref_agg    = trimesh.util.concatenate(ref_meshes)
    sample_agg = trimesh.util.concatenate(sample_meshes)
    ref_agg.merge_vertices()
    sample_agg.merge_vertices()
    print(f"\n  Aggregate ref:    {len(ref_agg.vertices):,} verts")
    print(f"  Aggregate sample: {len(sample_agg.vertices):,} verts")
    print(f"  Surface pts (each aggregate): {N_SURFACE_PTS:,}")

    # IS on the aggregates
    best_params, best_iou, _, mirror_axis = _run_is(ref_agg, sample_agg,
                                                     seed=seed, n_restarts=n_restarts)

    # Aggregate centroid — the SAME pivot is used for every individual bone so
    # that their relative positions are preserved after transformation.
    agg_centroid_np = np.array(sample_agg.vertices, dtype=np.float32).mean(0)

    # Apply the aggregate transform to each sample bone individually
    print(f"\n  Saving {n} transformed sample bones → {output_dir}/")
    for mesh, path in zip(sample_meshes, sample_paths):
        verts_np = np.array(mesh.vertices, dtype=np.float32)
        if mirror_axis is not None:
            verts_np = _mirror_verts(verts_np, agg_centroid_np, mirror_axis)
        new_v = _apply_params(verts_np, agg_centroid_np, best_params)
        result = mesh.copy()
        result.vertices = new_v
        out_path = output_dir / Path(path).name
        result.export(str(out_path))
        print(f"    {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB)")

    return best_iou, best_params


# ---------------------------------------------------------------------------
# Public API — per-bone refinement (tx + scale only, no rotation / ty / tz)
# ---------------------------------------------------------------------------

def smc_align_per_bone(
    ref_paths:    list,
    sample_paths: list,
    output_dir:   str | Path,
    seed:         int = 0,
    n_restarts:   int = 3,
) -> None:
    """
    Refine each matched bone pair independently, searching only x-translation
    and per-axis scale.  y/z translation and all rotations are locked to 0
    so the global aggregate alignment is preserved.

    Each bone uses its own centroid as the scale pivot.

    Parameters
    ----------
    ref_paths, sample_paths : ordered lists of STL paths (already aggregate-aligned)
    output_dir              : where to save refined sample bones (overwrites in-place
                              if output_dir == directory of sample_paths)
    seed, n_restarts        : IS random seed and number of independent restarts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(ref_paths)
    assert n == len(sample_paths)

    for i, (ref_path, sample_path) in enumerate(zip(ref_paths, sample_paths)):
        print(f"\n{'='*60}")
        print(f"Per-bone refinement  {i+1}/{n}: {Path(sample_path).name}")
        print(f"  (tx + scale only — rotation and ty/tz locked)")
        print(f"{'='*60}")

        ref    = trimesh.load(str(ref_path),    force="mesh", process=False)
        sample = trimesh.load(str(sample_path), force="mesh", process=False)
        ref.merge_vertices(); sample.merge_vertices()
        print(f"  Ref:    {len(ref.vertices):,} verts  |  "
              f"Sample: {len(sample.vertices):,} verts")

        best_params, best_iou, baseline_iou, mirror_axis = _run_is(
            ref, sample,
            seed=seed, n_restarts=n_restarts,
            stage_fn=_run_3stage_tx_scale,
        )

        print(f"\n  Final IoU: {best_iou:.4f}  "
              f"(Δ vs baseline: {best_iou - baseline_iou:+.4f})")

        # Apply with INDIVIDUAL bone centroid as pivot
        centroid_np = np.array(sample.vertices, dtype=np.float32).mean(0)
        verts_np    = np.array(sample.vertices, dtype=np.float32)
        if mirror_axis is not None:
            verts_np = _mirror_verts(verts_np, centroid_np, mirror_axis)
        new_v = _apply_params(verts_np, centroid_np, best_params)
        result = sample.copy()
        result.vertices = new_v
        out_path = output_dir / Path(sample_path).name
        result.export(str(out_path))
        print(f"  Saved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GenJAX SMC alignment: maximise IoU via factorised IS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pair:
  python smc_align.py ref_1.stl sample_1.stl -o sample_1_smc.stl

  # Aggregate (preserves inter-bone structure):
  python smc_align.py --ref ref_1.stl ref_2.stl \\
                      --sample sample_1.stl sample_2.stl \\
                      -o results/
""",
    )

    # Aggregate mode: --ref and --sample accept multiple files
    parser.add_argument("--ref",    nargs="+", metavar="STL",
                        help="Reference STL file(s)")
    parser.add_argument("--sample", nargs="+", metavar="STL",
                        help="Sample STL file(s) to align")

    # Single-pair mode: positional args (kept for backward compatibility)
    parser.add_argument("ref_pos",    nargs="?", metavar="ref",
                        help="Reference STL (single-pair mode)")
    parser.add_argument("sample_pos", nargs="?", metavar="sample",
                        help="Sample STL to align (single-pair mode)")

    parser.add_argument("-o", "--output",
                        help="Output STL (single-pair) or directory (aggregate)")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--restarts", type=int, default=3,
                        help="IS restarts per run (default: 3)")
    args = parser.parse_args()

    # Determine mode
    using_named   = args.ref is not None or args.sample is not None
    using_positional = args.ref_pos is not None or args.sample_pos is not None

    if using_named and using_positional:
        parser.error("Mix of positional and --ref/--sample arguments is not allowed.")

    if using_named:
        if not args.ref or not args.sample:
            parser.error("--ref and --sample must both be provided in aggregate mode.")
        if len(args.ref) != len(args.sample):
            parser.error("--ref and --sample must have the same number of files.")
        out_dir = args.output or "smc_results"
        smc_align_aggregate(args.ref, args.sample, out_dir,
                            seed=args.seed, n_restarts=args.restarts)
    else:
        if not args.ref_pos or not args.sample_pos:
            print(__doc__); sys.exit(0)
        out = args.output or str(
            Path(args.sample_pos).with_stem(Path(args.sample_pos).stem + "_smc"))
        smc_align(args.ref_pos, args.sample_pos, out,
                  seed=args.seed, n_restarts=args.restarts)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__); sys.exit(0)
    main()
