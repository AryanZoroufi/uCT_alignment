"""
Investigate why the registered atlas looks disproportional.

(A) Compare the atlas's SHAPE PROPORTIONS before vs after registration: PCA
    principal-axis extents of the original atlas surface vs the atlas-in-scan
    (from {tag}_atlasdbg.npz). A rigid+uniform-scale fit PRESERVES the aspect
    ratios; anisotropic scale / shear CHANGES them.
(B) Decompose the actual transform: re-run B256M1 sl registration, take the
    SimpleITK affine matrix, and report its singular values (= per-axis scale)
    and anisotropy ratio + shear. This is the smoking gun.
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).parent))

HERE = Path(__file__).parent
RECON = HERE / "../bones_to_recon"


def pca_extents(P):
    c = P.mean(0); Q = P - c
    w, V = np.linalg.eigh(Q.T @ Q / len(Q))
    o = np.argsort(w)[::-1]
    proj = Q @ V[:, o]
    return proj.max(0) - proj.min(0)


# ---------- (A) proportions before vs after ----------
a = np.load(HERE / "atlas.npz", allow_pickle=True)
orig = pca_extents(a["surface_points"].astype(float))
print("=== (A) shape proportions (PCA principal-axis extents) ===", flush=True)
print(f"ORIGINAL atlas : {np.round(orig,1)}  aspect {np.round(orig/orig[0],3)}", flush=True)
print("(long-axis extent: full atlas 1-9  vs  tibia-only 1&4  vs  scan bone)", flush=True)
for pair in ["B256M1", "B256M7"]:
    for tag in ["cl", "sl"]:
        occ = RECON / f"{pair}_occ"
        d = np.load(str(occ / f"{tag}_atlasdbg.npz"))
        av = d["atlas_verts"].astype(float); vl = d["atlas_vlabels"].astype(int)
        ext = pca_extents(av)
        tib = np.isin(vl, [1, 4])
        etib = pca_extents(av[tib]) if tib.sum() > 10 else np.zeros(3)
        bone = trimesh.load(str(occ / f"{tag}.stl"), force="mesh", process=False)
        bext = pca_extents(bone.vertices)
        print(f"{pair} {tag}: full-atlas {np.round(ext,1)} | tibia(1&4) "
              f"{np.round(etib,1)} | bone {np.round(bext,1)}   "
              f"[tibia/bone long = {etib[0]/bext[0]:.2f}x]", flush=True)

# ---------- (B) decompose the actual transform (B256M1 sl) ----------
print("\n=== (B) SimpleITK affine decomposition (B256M1 sl) ===", flush=True)
from pipeline import _segment_via_atlas
_, final_affine = _segment_via_atlas(
    RECON / "B256M1_occ" / "sl.stl", HERE / "atlas.npz", return_transform=True)
# CompositeTransform has no GetMatrix -> extract the (affine) linear part
# numerically: columns are T(e_i) - T(0). final_affine maps scan -> atlas.
o = np.array(final_affine.TransformPoint((0.0, 0.0, 0.0)))
M = np.column_stack([np.array(final_affine.TransformPoint(tuple(float(x) for x in e))) - o
                     for e in np.eye(3)])            # scan -> atlas linear
U, S, Vt = np.linalg.svd(M)
print(f"scan->atlas singular values (per-axis scale): {np.round(S,4)}", flush=True)
print(f"  ANISOTROPY (max/min) = {S.max()/S.min():.2f}   "
      f"(1.0 = perfectly uniform; >>1 = distorted)", flush=True)
print(f"atlas->scan per-axis scale (1/S): {np.round(1/S,4)}  "
      f"-> the atlas is stretched {S.max()/S.min():.1f}x more along one axis than another",
      flush=True)
print(f"det(M) = {np.linalg.det(M):.4e}  (volume scale scan->atlas)", flush=True)
# shear: how far M is from (rotation * uniform scale)
polar_R = U @ Vt
shear = np.linalg.norm(M / (S.mean()) - polar_R)
print(f"shear/anisotropy departure from pure similarity: {shear:.3f}", flush=True)
print("done", flush=True)
