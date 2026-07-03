"""
Align ONE surgical scan mesh to the atlas mesh -- NO segmentation, just
registration -- and report the fit.

Stages mirror the pipeline's atlas registration:
  STAGE 1  PCA coarse (8 mirror/sign) + similarity ICP   (segment_parts.register_points)
  STAGE 2  SMC importance-sampling refinement            (smc_align._run_is)

Reports per stage: similarity scale, ICP residual, symmetric surface Chamfer
(mm), and SMC IoU. Overlays atlas(aligned) on the surgical bone in rerun:
  grey = surgical bone, GREEN = atlas after PCA+ICP, CYAN = atlas after SMC.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from skimage import measure
sys.path.insert(0, str(Path(__file__).parent))
from segment_parts import register_points
import growth_config as cfg

PAIR, TAG = "B256M1", "sl"                       # the surgical bone
occ = Path(cfg.get(PAIR)["occ"])
ATLAS = Path(__file__).parent / "atlas.npz"
rng = np.random.default_rng(0)


def apply4(T, V):
    return (V @ T[:3, :3].T) + T[:3, 3]


def chamfer(A, B):
    """symmetric mean nearest-neighbour distance (mm) between two point sets."""
    dab = cKDTree(B).query(A)[0].mean()
    dba = cKDTree(A).query(B)[0].mean()
    return 0.5 * (dab + dba)


# ---------- surgical bone mesh (VOX already converted to mesh by vox_to_stl) ----------
scan = trimesh.load(str(occ / f"{TAG}.stl"), force="mesh", process=False)
scan.merge_vertices()
print(f"surgical mesh {PAIR} {TAG}: {len(scan.vertices):,}v {len(scan.faces):,}f  "
      f"bbox {np.round(scan.bounds[0],1)}..{np.round(scan.bounds[1],1)}", flush=True)

# ---------- atlas mesh ----------
a = np.load(str(ATLAS), allow_pickle=True)
solid = a["solid"]; pitch = float(a["pitch"]); origin = a["origin"].astype(np.float64)
av, af, _, _ = measure.marching_cubes(np.pad(solid, 1).astype(np.float32), 0.5)
av = (av - 1) * pitch + origin
atlas_mesh = trimesh.Trimesh(av, af, process=False)
atlas_surf = a["surface_points"].astype(np.float64)
print(f"atlas mesh: {len(av):,}v {len(af):,}f  bbox {np.round(av.min(0),1)}.."
      f"{np.round(av.max(0),1)}  (spans ~{(av.max(0)-av.min(0)).max():.0f} atlas-mm)",
      flush=True)

# subsampled surface points for registration + chamfer
scan_surf = scan.vertices.astype(np.float64)
scan_surf = scan_surf[rng.choice(len(scan_surf), min(8000, len(scan_surf)), replace=False)]
atlas_ss = atlas_surf[rng.choice(len(atlas_surf), min(8000, len(atlas_surf)), replace=False)]

# ---------- STAGE 1: PCA coarse + similarity ICP (atlas -> scan) ----------
T, cost, label = register_points(atlas_ss, scan_surf, seed=0)
R = T[:3, :3]; scale = float(np.linalg.norm(R, axis=0).mean()); t = T[:3, 3]
atlas_icp_v = apply4(T, atlas_mesh.vertices)
icp_surf = apply4(T, atlas_ss)
ch_icp = chamfer(icp_surf, scan_surf)
print(f"\n[STAGE 1  PCA+similarity-ICP]  [{label}]", flush=True)
print(f"  scale={scale:.4f}   ICP residual={cost:.3f} mm   "
      f"symmetric Chamfer={ch_icp:.3f} mm", flush=True)
print(f"  translation={np.round(t,2)}", flush=True)

# ---------- STAGE 2: SMC importance-sampling refinement ----------
import fast_simplification


def dec(V, F, target=10000):
    if len(F) <= target * 1.2:
        return V, F
    tr = float(np.clip(1.0 - target / len(F), 0.0, 0.99))
    return fast_simplification.simplify(V.astype(np.float64), F.astype(np.int32),
                                        target_reduction=tr)


smc_iou = smc_base = None
atlas_smc_v = None
try:
    from smc_align import _run_is
    av_d, af_d = dec(atlas_icp_v, af)
    sv_d, sf_d = dec(scan.vertices, scan.faces)
    atlas_icp_mesh = trimesh.Trimesh(av_d, af_d, process=True)
    scan_mesh = trimesh.Trimesh(sv_d, sf_d, process=True)
    params, smc_iou, smc_base, mirror_axis = _run_is(
        ref_mesh=scan_mesh, sample_mesh=atlas_icp_mesh, seed=0, n_restarts=3)
    c0 = atlas_icp_mesh.vertices.mean(0).astype(np.float64)

    def to4(M, tv):
        M4 = np.eye(4); M4[:3, :3] = M; M4[:3, 3] = tv; return M4
    Ssmc = np.eye(4)
    if mirror_axis is not None:
        D = np.eye(3); D[mirror_axis, mirror_axis] = -1
        vv = np.zeros(3); vv[mirror_axis] = 2.0 * c0[mirror_axis]
        Ssmc = to4(D, vv) @ Ssmc
    p = params
    crx, srx = np.cos(p['rx']), np.sin(p['rx'])
    cry, sry = np.cos(p['ry']), np.sin(p['ry'])
    crz, srz = np.cos(p['rz']), np.sin(p['rz'])
    Rx = np.array([[1, 0, 0], [0, crx, -srx], [0, srx, crx]])
    Ry = np.array([[cry, 0, sry], [0, 1, 0], [-sry, 0, cry]])
    Rz = np.array([[crz, -srz, 0], [srz, crz, 0], [0, 0, 1]])
    M = Rz @ Ry @ Rx @ np.diag([p['sx'], p['sy'], p['sz']])
    tsm = np.array([p['tx'], p['ty'], p['tz']])
    Ssmc = to4(M, c0 - M @ c0 + tsm) @ Ssmc
    atlas_smc_v = apply4(Ssmc, atlas_icp_v)
    smc_surf = apply4(Ssmc, icp_surf)
    ch_smc = chamfer(smc_surf, scan_surf)
    print(f"\n[STAGE 2  SMC refinement]  n_restarts=3  mirror_axis={mirror_axis}", flush=True)
    print(f"  IoU={smc_iou:.4f}  (baseline {smc_base:.4f})   "
          f"symmetric Chamfer={ch_smc:.3f} mm", flush=True)
    print(f"  SMC scale=({p['sx']:.3f},{p['sy']:.3f},{p['sz']:.3f})  "
          f"rot=({np.degrees(p['rx']):.0f},{np.degrees(p['ry']):.0f},"
          f"{np.degrees(p['rz']):.0f})deg", flush=True)
except Exception as e:
    print(f"\n[STAGE 2] SMC failed: {type(e).__name__}: {e}", flush=True)

# ---------- overlay in rerun ----------
import rerun as rr
rr.init("align_to_atlas", spawn=True)
print("\ninit ok", flush=True)
rr.log("surgical_bone", rr.Mesh3D(
    vertex_positions=scan.vertices.astype(np.float32), triangle_indices=scan.faces,
    vertex_colors=np.tile([150, 150, 150], (len(scan.vertices), 1))))
rr.log("atlas_after_ICP", rr.Mesh3D(
    vertex_positions=atlas_icp_v.astype(np.float32), triangle_indices=af,
    vertex_colors=np.tile([40, 210, 60], (len(atlas_icp_v), 1))))
if atlas_smc_v is not None:
    rr.log("atlas_after_SMC", rr.Mesh3D(
        vertex_positions=atlas_smc_v.astype(np.float32), triangle_indices=af,
        vertex_colors=np.tile([40, 220, 230], (len(atlas_smc_v), 1))))

print("\n===== SUMMARY =====", flush=True)
print(f"surgical bone: {PAIR} {TAG}", flush=True)
print(f"STAGE 1 PCA+ICP : scale={scale:.4f}  residual={cost:.3f}mm  "
      f"chamfer={ch_icp:.3f}mm  [{label}]", flush=True)
if smc_iou is not None:
    print(f"STAGE 2 SMC     : IoU={smc_iou:.4f} (baseline {smc_base:.4f})", flush=True)
print("rerun: grey=surgical bone, GREEN=atlas after ICP, CYAN=atlas after SMC.",
      flush=True)
