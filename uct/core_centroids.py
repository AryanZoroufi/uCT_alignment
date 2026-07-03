"""
Step 2 experiment (KAN-28): does ectopic growth contaminate the part-1/part-4
centroids and shift the gap midpoint?

Growth in the gap gets assigned (by the atlas nearest-label) partly to part1 and
partly to part4. On the SURGICAL leg that pulls the part centroids toward the
gap; the CONTRALATERAL leg has no growth, so it's the control.

For each leg we voxelize each part mesh to a solid, then compare its centroid
RAW vs after morphological OPENING (which strips thin growth spurs). We report:
  - how far each part centroid and the gap midpoint move,
  - the axial component of that move (along the part1->part4 axis),
  - and for SL, the downstream change in the A (full-cross-section) volume when
    the gap center is recomputed from the opened cores.

If SL moves and CL doesn't, growth is contaminating the SL centroids and we
should define the parts from opened cores before taking centroids.

Coordinate frame (native voxel index, 1 vox = vmm = 6.43um):
  native_idx = mesh_mm / (vmm * STEP)     # STEP=2 marching-cubes oversizing
Stored c1/c4 in the crop npz are surface-vertex means in this same frame.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.ndimage import (binary_opening, binary_closing, label as cc_label,
                           generate_binary_structure, iterate_structure)
sys.path.insert(0, str(Path(__file__).parent))

OCC = Path(__file__).parent / "../bones_to_recon/B256M1_occ"
STEP = 2
GT_VOX = 366110


def ball(r):
    return iterate_structure(generate_binary_structure(3, 1), r)


def solid_idx(mesh, vmm):
    """Solid voxelization of a part mesh -> (N,3) native voxel indices."""
    vg = mesh.voxelized(pitch=vmm * STEP).fill()
    return np.round(vg.points / (vmm * STEP)).astype(int)


def centroids(mesh, vmm, r):
    """Raw solid centroid vs opened-core centroid (native idx), + voxel counts."""
    idx = solid_idx(mesh, vmm)
    lo = idx.min(0)
    vol = np.zeros(idx.max(0) - lo + 1, bool)
    vi = idx - lo
    vol[vi[:, 0], vi[:, 1], vi[:, 2]] = True
    raw_c = idx.mean(0)
    if r > 0:
        op = binary_opening(vol, structure=ball(r))
        core_c = (np.argwhere(op) + lo).mean(0) if op.any() else raw_c
        core_n = int(op.sum())
    else:
        core_c, core_n = raw_c, int(vol.sum())
    return raw_c, core_c, int(vol.sum()), core_n


def a_volume(bone, nlo, c1, c4, W=15):
    """Full-cross-section (A) volume: largest CC of the closed midpoint band."""
    a = (c4 - c1); sep = float(np.linalg.norm(a)); a = a / sep
    center = sep / 2
    bxyz = np.argwhere(bone) + nlo
    t = (bxyz - c1) @ a
    band = np.abs(t - center) <= W
    g = np.zeros(bone.shape, bool)
    idx = bxyz[band] - nlo
    g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    g = binary_closing(g, iterations=2) & bone
    gl, _ = cc_label(g)
    if gl.max() == 0:
        return 0, sep
    szs = np.bincount(gl.ravel()); szs[0] = 0
    return int((gl == szs.argmax()).sum()), sep


R_SWEEP = (0, 5, 10, 15)
for tag in ("sl", "cl"):
    d = np.load(f"/tmp/{tag}_crop.npz")
    vmm = float(d["vmm"])
    c1_stored, c4_stored = d["c1"], d["c4"]
    p1 = trimesh.load(str(OCC / f"{tag}_part1.stl"), force="mesh", process=False)
    p4 = trimesh.load(str(OCC / f"{tag}_part4.stl"), force="mesh", process=False)
    print(f"\n=== {tag.upper()} ===", flush=True)
    # sanity: raw solid centroid vs stored surface-vertex centroid
    c1r0, _, n1v, _ = centroids(p1, vmm, 0)
    c4r0, _, n4v, _ = centroids(p4, vmm, 0)
    print(f"  part1 solid={n1v:,}vox raw_centroid={np.round(c1r0,1)} "
          f"(stored {np.round(c1_stored,1)}, |Δ|={np.linalg.norm(c1r0-c1_stored):.1f})",
          flush=True)
    print(f"  part4 solid={n4v:,}vox raw_centroid={np.round(c4r0,1)} "
          f"(stored {np.round(c4_stored,1)}, |Δ|={np.linalg.norm(c4r0-c4_stored):.1f})",
          flush=True)
    a_axis = (c4r0 - c1r0); a_axis /= np.linalg.norm(a_axis)
    for r in R_SWEEP:
        c1r, c1c, _, c1n = centroids(p1, vmm, r)
        c4r, c4c, _, c4n = centroids(p4, vmm, r)
        mid_raw = (c1r + c4r) / 2
        mid_core = (c1c + c4c) / 2
        shift = np.linalg.norm(mid_core - mid_raw)
        axial = float((mid_core - mid_raw) @ a_axis)
        print(f"  r={r:2d}: p1move={np.linalg.norm(c1c-c1r):4.1f} "
              f"p4move={np.linalg.norm(c4c-c4r):4.1f} vox | "
              f"|mid shift|={shift:4.1f}vox ({shift*vmm*1000:4.0f}um) axial={axial:+5.1f}vox "
              f"| core: p1={c1n:,} p4={c4n:,}", flush=True)

    if tag == "sl":
        bone = d["bone"]; nlo = d["nlo"]
        vraw, _ = a_volume(bone, nlo, c1r0, c4r0)
        c1o, _, _, _ = centroids(p1, vmm, 10)
        c4o, _, _, _ = centroids(p4, vmm, 10)
        vcore, _ = a_volume(bone, nlo, c1o, c4o)
        print(f"  DOWNSTREAM A-volume (W=15): raw-centroids={vraw:,} ({vraw/GT_VOX:.2f}xGT)  "
              f"opened-r10-centroids={vcore:,} ({vcore/GT_VOX:.2f}xGT)  "
              f"Δ={vcore-vraw:+,} vox", flush=True)
print("\ndone", flush=True)
