"""
GenJAX importance sampling to refine mesh alignment via IoU maximisation.

Algorithm — 4-stage factorised SMC (full, used in aggregate mode):
  Stage 1: search translation only         (3-D, rotation=0, scale=1)
  Stage 2: search rotation only            (3-D, fix t from stage 1)
  Stage 3: search per-axis scale only      (3-D, fix t+r from stages 1-2)
  Stage 4: joint 9-D fine-tuning           (tight priors around stages 1-3 MAP)

Algorithm — per-bone refinement mode (same 4-stage as aggregate, own centroid):
  Stage 1: search translation              (3-D, rotation=0, scale=1)
  Stage 2: search rotation                 (3-D, fix t from stage 1)
  Stage 3: search per-axis scale           (3-D, fix t+r from stages 1-2)
  Stage 4: joint 9-D fine-tuning           (tight priors around stages 1-3 MAP)

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
N_PARTICLES     = 5_000   # particles per stage
N_SURFACE_PTS   = 8_000   # surface samples per mesh (uniform, seed=0 → deterministic)
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
# Point transformation (JAX)
# ---------------------------------------------------------------------------
# NOTE: jnp.array([[1,0,0],[0,c,-s],...]) with traced scalar values (c, s)
# causes XlaRuntimeError in JAX ≥0.5.x under vmap.  Use elementwise ops
# instead of matrix construction + @ to keep everything in the traced graph.
# ---------------------------------------------------------------------------

def transform_pts(pts, centroid, t, rx, ry, rz, sx, sy, sz):
    """Centre → scale → rotate (ZYX Euler: Rx then Ry then Rz) → translate."""
    # center and scale
    cx = (pts[:, 0] - centroid[0]) * sx
    cy = (pts[:, 1] - centroid[1]) * sy
    cz = (pts[:, 2] - centroid[2]) * sz

    # Rx
    crx, srx = jnp.cos(rx), jnp.sin(rx)
    cx, cy, cz = cx, crx * cy - srx * cz, srx * cy + crx * cz

    # Ry
    cry, sry = jnp.cos(ry), jnp.sin(ry)
    cx, cy, cz = cry * cx + sry * cz, cy, -sry * cx + cry * cz

    # Rz
    crz, srz = jnp.cos(rz), jnp.sin(rz)
    cx, cy, cz = crz * cx - srz * cy, srz * cx + crz * cy, cz

    return jnp.stack([cx + centroid[0] + t[0],
                      cy + centroid[1] + t[1],
                      cz + centroid[2] + t[2]], axis=1)

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

    t_vec       = jnp.stack([tx, ty, tz])
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
                run_label: str = "", stage_cb=None) -> tuple[dict, float]:
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
    if stage_cb: stage_cb(1, m1, iou1)

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
    if stage_cb: stage_cb(2, m2, iou2)

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
    if stage_cb: stage_cb(3, m3, iou3)

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
    if stage_cb: stage_cb(4, m4, iou4)

    best_iou, best_params = iou1, m1
    for iou_s, m_s in [(iou2, m2), (iou3, m3), (iou4, m4)]:
        if iou_s > best_iou:
            best_iou, best_params = iou_s, m_s
    return best_params, best_iou


# ---------------------------------------------------------------------------
# 3-stage restricted IS: x-translation + scale only (no rotation, no ty/tz)
# ---------------------------------------------------------------------------

def _run_3stage_tx_scale(key, sample_pts_j, centroid_j, ref_grid_j, vmin, vmax,
                         run_label: str = "", stage_cb=None) -> tuple[dict, float]:
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
    if stage_cb: stage_cb(1, m1, iou1)

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
    if stage_cb: stage_cb(2, m2, iou2)

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
    if stage_cb: stage_cb(3, m3, iou3)

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
            stage_fn=None,
            probe_cb=None) -> tuple[dict, float, float]:
    """
    Run multi-restart IS on a (ref, sample) mesh pair.

    Mirror strategy — probe then exploit:
      Phase 1: run one full IS restart per mirror orientation (up to 4 probes).
               This rotation-searches ALL orientations, not just the one with the
               best baseline IoU.  A mirror with lower baseline IoU can easily
               converge to a higher final IoU after rotation search (e.g. when the
               bones need a large rotation that the baseline cannot predict).
      Phase 2: run the remaining restarts (n_restarts − n_probes) from the mirror
               that produced the highest IoU in Phase 1.

    Returns
    -------
    best_params   : dict with tx/ty/tz/rx/ry/rz/sx/sy/sz
    best_iou      : IoU achieved by best_params
    baseline_iou  : IoU of the identity transform (no mirror, no rotation)
    best_mirror   : None, or 0/1/2 (axis that was mirrored before IS)
    """
    if stage_fn is None:
        stage_fn = _run_4stage

    # ── Reference grid (fixed throughout)
    rv_sub, _ = trimesh.sample.sample_surface(ref_mesh, N_SURFACE_PTS, seed=0)
    rv_sub = rv_sub.astype(np.float32)
    ref_pts_j = jnp.array(rv_sub, dtype=jnp.float32)

    rpts = np.array(ref_mesh.vertices, dtype=np.float32)
    pad  = (rpts.max(0) - rpts.min(0)) * 0.15
    vmin = jnp.array(rpts.min(0) - pad)
    vmax = jnp.array(rpts.max(0) + pad)
    ref_grid_j = voxelize(ref_pts_j, vmin, vmax)

    sample_verts_np    = np.array(sample_mesh.vertices, dtype=np.float32)
    sample_centroid_np = sample_verts_np.mean(0)

    MIRRORS = [(None, "none"), (0, "flip-x"), (1, "flip-y"), (2, "flip-z")]

    # ── Pre-compute per-mirror sample data + baseline IoUs (cheap)
    print("  Mirror baselines (unoptimised):")
    mirror_data: dict = {}   # axis → (sample_pts_j, centroid_j, baseline_iou)
    none_baseline = 0.0
    for axis, label in MIRRORS:
        v = (_mirror_verts(sample_verts_np, sample_centroid_np, axis)
             if axis is not None else sample_verts_np)
        sv, _ = trimesh.sample.sample_surface(
            trimesh.Trimesh(vertices=v, faces=sample_mesh.faces), N_SURFACE_PTS, seed=0)
        sv    = sv.astype(np.float32)
        s_pts = jnp.array(sv, dtype=jnp.float32)
        c_j   = jnp.array(v, dtype=jnp.float32).mean(axis=0)
        bl    = float(iou_grids(voxelize(s_pts, vmin, vmax), ref_grid_j))
        print(f"    {label:8s}  baseline IoU = {bl:.4f}")
        mirror_data[axis] = (s_pts, c_j, bl)
        if axis is None:
            none_baseline = bl

    # ── Phase 1: one IS restart per mirror (probe all orientations)
    n_probes = min(len(MIRRORS), n_restarts)
    probe_results = []   # (axis, label, params, iou)

    print(f"\n  Phase 1 — probing {n_probes} mirror(s) (one restart each) ...")
    for pi in range(n_probes):
        axis, label = MIRRORS[pi]
        s_pts, c_j, bl = mirror_data[axis]
        print(f"\n{'='*50}")
        print(f"Probe {pi+1}/{n_probes}  mirror={label}  baseline={bl:.4f}  (seed={seed+pi})")
        print(f"{'='*50}")
        key      = jax.random.key(seed + pi)
        _scb = (lambda si, p, iou, _ax=axis, _pi=pi: probe_cb(_pi, _ax, si, p, iou)) if probe_cb else None
        params_r, iou_r = stage_fn(key, s_pts, c_j, ref_grid_j, vmin, vmax,
                                   run_label=f"probe {pi+1} [{label}]",
                                   stage_cb=_scb)
        print(f"\n  Probe {pi+1} [{label}]: IoU = {iou_r:.4f}")
        probe_results.append((axis, label, params_r, iou_r))

    # Best mirror = highest post-IS IoU from Phase 1
    best_pi       = int(np.argmax([r[3] for r in probe_results]))
    best_axis, best_label, best_params, best_iou = probe_results[best_pi]
    best_s_pts, best_c_j, _ = mirror_data[best_axis]

    print(f"\n  Best mirror from probes: {best_label}  (IoU={best_iou:.4f})")

    # ── Phase 2: extra restarts from the winning mirror
    n_extra = n_restarts - n_probes
    if n_extra > 0:
        print(f"  Phase 2 — {n_extra} more restart(s) from mirror={best_label} ...")

    for r in range(n_extra):
        print(f"\n{'='*50}")
        print(f"Restart {n_probes+r+1}/{n_restarts}  mirror={best_label}  "
              f"(seed={seed+n_probes+r})")
        print(f"{'='*50}")
        key      = jax.random.key(seed + n_probes + r)
        _ri = n_probes + r
        _scb = (lambda si, p, iou, _ax=best_axis, _ri=_ri: probe_cb(_ri, _ax, si, p, iou)) if probe_cb else None
        params_r, iou_r = stage_fn(key, best_s_pts, best_c_j, ref_grid_j, vmin, vmax,
                                   run_label=f"restart {n_probes+r+1} [{best_label}]",
                                   stage_cb=_scb)
        print(f"\n  Restart {n_probes+r+1}: IoU = {iou_r:.4f}")
        if iou_r > best_iou:
            best_iou, best_params = iou_r, params_r

    print(f"\n{'='*50}")
    print(f"  Winning: mirror={best_label}  IoU={best_iou:.4f}  "
          f"(Δ vs identity: {best_iou - none_baseline:+.4f})")

    return best_params, best_iou, none_baseline, best_axis


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
    frame_cb=None,
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
    agg_verts_orig = [np.array(m.vertices, dtype=np.float32) for m in sample_meshes]
    agg_centroid_np = np.array(sample_agg.vertices, dtype=np.float32).mean(0)

    def _agg_probe_cb(probe_i, mirror_axis, stage_i, params, iou=None):
        if frame_cb is None:
            return
        transformed = []
        for v_orig in agg_verts_orig:
            v = v_orig.copy()
            if mirror_axis is not None:
                v = _mirror_verts(v, agg_centroid_np, mirror_axis)
            v = _apply_params(v, agg_centroid_np, params)
            transformed.append(v)
        label = f"agg  probe {probe_i+1}  stage {stage_i}/4"
        frame_cb(label, transformed, iou)

    best_params, best_iou, _, mirror_axis = _run_is(ref_agg, sample_agg,
                                                     seed=seed, n_restarts=n_restarts,
                                                     probe_cb=_agg_probe_cb)

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
# Public API — per-bone refinement (full 9-DOF per bone)
# ---------------------------------------------------------------------------

def smc_align_per_bone(
    ref_paths:    list,
    sample_paths: list,
    output_dir:   str | Path,
    seed:         int = 0,
    n_restarts:   int = 3,
    frame_cb=None,
) -> None:
    """
    Refine each matched bone pair independently with full 9-DOF search
    (translation + rotation + scale), pivoting on each bone's own centroid.

    After the aggregate alignment in step 7, each bone may still carry a
    residual rotation error — the smaller bone in particular contributes less
    to the aggregate IoU and can be left misaligned.  The previous tx+scale
    restriction caused the optimiser to compensate with extreme non-uniform
    scale factors (15–20%) instead of correcting the underlying rotation, which
    degraded both IoU and the shape comparison.

    Allowing full rotation per bone corrects this without touching the other
    bone's position.

    Parameters
    ----------
    ref_paths, sample_paths : ordered lists of STL paths (already aggregate-aligned)
    output_dir              : where to save refined sample bones
    seed, n_restarts        : IS random seed and number of independent restarts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(ref_paths)
    assert n == len(sample_paths)

    # Track current vertex state for all bones (updated as each bone finalizes).
    # Used so per-bone frames show all bones at their current best position.
    current_verts = []
    current_faces = []
    for p in sample_paths:
        m = trimesh.load(str(p), force="mesh", process=False)
        m.merge_vertices()
        current_verts.append(np.array(m.vertices, dtype=np.float32))
        current_faces.append(np.array(m.faces))

    for i, (ref_path, sample_path) in enumerate(zip(ref_paths, sample_paths)):
        print(f"\n{'='*60}")
        print(f"Per-bone refinement  {i+1}/{n}: {Path(sample_path).name}")
        print(f"  (full 9-DOF: translation + rotation + scale)")
        print(f"{'='*60}")

        ref    = trimesh.load(str(ref_path),    force="mesh", process=False)
        sample = trimesh.load(str(sample_path), force="mesh", process=False)
        ref.merge_vertices(); sample.merge_vertices()
        print(f"  Ref:    {len(ref.vertices):,} verts  |  "
              f"Sample: {len(sample.vertices):,} verts")

        _bone_verts_pre = np.array(sample.vertices, dtype=np.float32)
        _bone_centroid  = _bone_verts_pre.mean(0)

        def _bone_probe_cb(probe_i, mirror_axis, stage_i, params, iou=None,
                           _i=i, _bv=_bone_verts_pre, _bc=_bone_centroid):
            if frame_cb is None:
                return
            v = _bv.copy()
            if mirror_axis is not None:
                v = _mirror_verts(v, _bc, mirror_axis)
            v = _apply_params(v, _bc, params)
            all_verts = list(current_verts)
            all_verts[_i] = v
            label = f"bone {_i+1}/{n}  probe {probe_i+1}  stage {stage_i}/3"
            frame_cb(label, all_verts, iou)

        best_params, best_iou, baseline_iou, mirror_axis = _run_is(
            ref, sample,
            seed=seed, n_restarts=n_restarts,
            stage_fn=_run_4stage,
            probe_cb=_bone_probe_cb,
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
