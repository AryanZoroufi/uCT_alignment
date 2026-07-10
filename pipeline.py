#!/usr/bin/env python3
"""
µCT ectopic-bone-growth pipeline (one-liner).

Measures the ectopic bone growth in the cartilage gap between a tibia's two ossification centers,
for a surgical (SL) leg relative to its contralateral control (CL).

Method (locked):
  1. VOX -> surface mesh (gaussian sigma=1, Otsu, marching cubes) for SL and CL.
  2. Atlas 7-DOF chamfer segmentation -> tibia parts 1 (bottom) & 4 (top).
  3. Threshold each scan at its own Otsu + <hu-th> HU -> native tibia voxels.
  4. Register CL -> SL (similarity-ICP).
  5. Locate the growth centre from the data (ectopic bridge centroid near the part1<->part4 contact).
  6. Disc ROI (radius, height) on the voxelized-tibia PCA axis, centred at the growth centre.
  7. Growth = surgical bone in the disc that is far from the aligned control (the ectopic bridge)
     -> pore-close -> largest connected component -> alpha=1.5mm solid fill
     -> clip to the surgical bone's solid envelope -> count at native resolution.

Usage:
    python pipeline.py --sl-path <sl.VOX> --cl-path <cl.VOX>
    python pipeline.py --sl-path <sl.VOX> --cl-path <cl.VOX> --visualize
    python pipeline.py --sl-path <sl.VOX> --cl-path <cl.VOX> \
        --disc-radius 1.2 --disc-height 0.55 --hu-th 200

Defaults are the values validated on B256M1/M7 (radius 1.2 mm, full height 0.55 mm, Otsu+200).
--visualize opens the same WebGL viewer as the shared artifacts, served on localhost only.
"""
import sys
import os
import json
import base64
import argparse
import subprocess
import webbrowser
import functools
import http.server
import socketserver
from pathlib import Path
from collections import Counter
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree, Delaunay
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

UCT = Path(__file__).resolve().parent / "uct"
sys.path.insert(0, str(UCT))
from vox_to_stl import load_vox, _mesh_from_volume            # noqa: E402
from growth_peak_prep import atlas_seg, refine_7dof, euler_R, apply4  # noqa: E402
from atlas_register import register_atlas_7dof_chamfer        # noqa: E402

# ---- locked method constants -------------------------------------------------
STEP = 2                       # mesh coord = physical_mm * STEP  (1 mesh unit = 0.5 mm)
ALPHA_MM = 1.5                 # alpha-shape hull radius (physical mm)
DTOL = 0.12                    # "far from control" tolerance (mesh)
CLOSE = 3                      # pore-close the ectopic bridge
CLOSE_ENV = 4                  # close the SL occupancy before fill-holes (bone solid envelope)
ALPHA_PTS = 12000              # bridge points fed to the Delaunay
GT_VOX = 366110                # B256M1 SL Dragonfly human paint (reference; M1-specific)
STRUCT = np.ones((3, 3, 3))
rng = np.random.default_rng(0)
COL = {"sl": [95, 155, 235], "cl": [240, 150, 55]}
C_GR, C_PK, C_DS = [235, 45, 45], [40, 255, 220], [255, 225, 40]
UNIT = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], float)
CFACE = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 5, 1], [0, 4, 5],
                  [2, 3, 7], [2, 7, 6], [0, 6, 4], [0, 2, 6], [1, 5, 7], [1, 7, 3]])


# ---- geometry helpers --------------------------------------------------------
def make_mesh(grid, vmm):
    """VOX volume -> surface mesh (matches the occ meshes: gaussian sigma=1, Otsu, MC step 2)."""
    iso = float(threshold_otsu(grid[::4, ::4, ::4]))
    v, f = _mesh_from_volume(gaussian_filter(grid, 1.0), iso, STEP, vmm)
    m = trimesh.Trimesh(v, f, process=False); m.merge_vertices()
    return m


def tibia(path, offset):
    """Native tibia voxels (mesh coords) at Otsu+offset, plus the mesh and atlas part1/4 verts."""
    grid, vmm = load_vox(path); ots = float(threshold_otsu(grid[::4, ::4, ::4]))
    mesh = make_mesh(grid, vmm)
    seg = atlas_seg(mesh); p1m, p4m = seg[1], seg[4]
    n = np.vstack([p1m, p4m]) / (vmm * STEP)
    lo = np.maximum(np.floor(n.min(0)).astype(int) - 45, 0)
    hi = np.minimum(np.ceil(n.max(0)).astype(int) + 45, np.array(grid.shape))
    b = grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] > (ots + offset); del grid
    lab, _ = ndimage.label(b, STRUCT)
    bidx = np.argwhere(b)                                    # snap surface seeds to nearest bone voxel
    if len(bidx) > 1_000_000:
        bidx = bidx[rng.choice(len(bidx), 1_000_000, replace=False)]
    btree = cKDTree(bidx); blbl = lab[bidx[:, 0], bidx[:, 1], bidx[:, 2]]

    def cs(seed):
        s = seed / (vmm * STEP) - lo
        if len(s) > 5000:
            s = s[rng.choice(len(s), 5000, replace=False)]
        L = blbl[btree.query(s)[1]]
        return Counter(L).most_common(1)[0][0] if len(L) else -1
    keep = np.isin(lab, [x for x in (cs(p1m), cs(p4m)) if x > 0])
    return (np.argwhere(keep) + lo).astype(np.float64) * (vmm * STEP), vmm * STEP, mesh, p1m, p4m


def cl_to_sl(clmesh, slmesh):
    """Similarity-ICP transform mapping CL points into the SL frame."""
    cs = clmesh.vertices[rng.choice(len(clmesh.vertices), 4000, replace=False)].astype(np.float64)
    ss = slmesh.vertices[rng.choice(len(slmesh.vertices), 4000, replace=False)].astype(np.float64)
    T, _ = register_atlas_7dof_chamfer(cs, ss)
    cl7 = apply4(T, cs); dp, c7 = refine_7dof(cl7, ss); M = euler_R(dp[3:6]) * np.exp(dp[6])
    return lambda P: (apply4(T, P) - c7) @ M.T + c7 + dp[0:3]


def pca_axis(pts):
    c = pts - pts.mean(0); a = np.linalg.eigh(c.T @ c)[1][:, -1]
    t = np.array([1., 0, 0]) if abs(a[0]) < 0.9 else np.array([0., 1, 0])
    u = np.cross(a, t); u /= np.linalg.norm(u); w = np.cross(a, u)
    return a, u, w


def circumradii(pts, Sx):
    q = pts - pts.mean(0); p = q[Sx]; a, b, c, e = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    A = 2.0 * np.stack([b - a, c - a, e - a], axis=1)
    rhs = np.stack([(b * b - a * a).sum(1), (c * c - a * a).sum(1), (e * e - a * a).sum(1)], axis=1)
    good = np.abs(np.linalg.det(A)) > 1e-9; R = np.full(len(Sx), np.inf)
    if good.any():
        R[good] = np.linalg.norm(np.linalg.solve(A[good], rhs[good]) - a[good], axis=1)
    return R


def connected_bridge(sld, cld, vsize):
    """Ectopic bone (SL far from aligned CL) -> pore-close -> largest connected component."""
    if len(sld) < 10:
        return np.empty((0, 3))
    far = cKDTree(cld).query(sld)[0] > DTOL if len(cld) else np.ones(len(sld), bool)
    ecto = sld[far]
    if len(ecto) < 10:
        return np.empty((0, 3))
    idx = np.round(ecto / vsize).astype(np.int64); bl = idx.min(0); idx -= bl
    b = np.zeros(tuple(idx.max(0) + 3), bool); b[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, _ = ndimage.label(ndimage.binary_closing(b, STRUCT, iterations=CLOSE), STRUCT)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (np.argwhere(lab == sizes.argmax()) + bl).astype(np.float64) * vsize


def in_disc(pts, peak, a, u, w, RR, HH):
    dd = pts - peak
    return (np.abs(dd @ a) <= HH) & ((dd @ u) ** 2 + (dd @ w) ** 2 <= RR * RR)


def growth_volume(slp, cla, peak, a, u, w, vsize, RR, HH):
    """Bridge -> alpha=1.5mm fill -> clip to SL solid envelope. Returns (nvox, kept-voxel centres)."""
    sld = slp[in_disc(slp, peak, a, u, w, RR, HH)]
    cld = cla[in_disc(cla, peak, a, u, w, RR, HH)]
    bvox = connected_bridge(sld, cld, vsize)
    if not len(bvox):
        return 0, np.empty((0, 3))
    sub = bvox if len(bvox) <= ALPHA_PTS else bvox[rng.choice(len(bvox), ALPHA_PTS, replace=False)]
    tri = Delaunay(sub); R = circumradii(sub, tri.simplices)
    mn = np.floor(bvox.min(0) / vsize).astype(np.int64); mx = np.ceil(bvox.max(0) / vsize).astype(np.int64)
    G = np.stack(np.meshgrid(*[np.arange(mn[k], mx[k] + 1) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
    centres = G.astype(np.float64) * vsize
    simp = tri.find_simplex(centres)
    filled = centres[(simp >= 0) & (R[np.clip(simp, 0, None)] <= ALPHA_MM * STEP)]
    if not len(filled):
        return 0, filled
    fidx = np.round(filled / vsize).astype(np.int64); sidx = np.round(slp / vsize).astype(np.int64)
    cmn, cmx = fidx.min(0) - 6, fidx.max(0) + 6
    loc = sidx[((sidx >= cmn) & (sidx <= cmx)).all(1)] - cmn
    B = np.zeros(tuple(cmx - cmn + 1), bool); B[loc[:, 0], loc[:, 1], loc[:, 2]] = True
    Bsolid = ndimage.binary_fill_holes(ndimage.binary_closing(B, iterations=CLOSE_ENV))
    fl = fidx - cmn; keep = Bsolid[fl[:, 0], fl[:, 1], fl[:, 2]]
    return int(keep.sum()), filled[keep]


def growth_centre(slp, cla, a, u, w, vsize, p1m, p4m):
    """Data-driven disc centre = centroid of the ectopic bridge near the part1<->part4 contact."""
    d1, i1 = cKDTree(p1m).query(p4m); j = int(d1.argmin())
    conn = (p4m[j] + p1m[i1[j]]) / 2.0                          # closest part1<->part4 midpoint
    sg = slp[in_disc(slp, conn, a, u, w, 3.0, 1.5)]             # generous cylinder r1.5mm h0.75mm
    cg = cla[in_disc(cla, conn, a, u, w, 3.0, 1.5)]
    bvox = connected_bridge(sg, cg, vsize)
    return bvox.mean(0) if len(bvox) else conn


# ---- visualization (localhost WebGL viewer) ---------------------------------
def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _points(P, color, name, label):
    return dict(name=name, group=0, label=label, points=True,
                pos=_b64(P.astype(np.float32)), col=_b64(np.tile(color, (len(P), 1)).astype(np.uint8)))


def _cubes(centers, vsize, color, name, label):
    out = []
    for k in range(0, len(centers), 8000):
        ch = centers[k:k + 8000]; n = len(ch)
        V = (ch[:, None, :] + UNIT[None, :, :] * vsize).reshape(-1, 3)
        F = (CFACE[None, :, :] + np.arange(n)[:, None, None] * 8).reshape(-1, 3)
        out.append(dict(name=name, group=0, label=label, pos=_b64(V.astype(np.float32)),
                        col=_b64(np.tile(color, (len(V), 1)).astype(np.uint8)),
                        idx=_b64(F.astype(np.uint16)), nfaces=len(F)))
    return out


def build_and_serve(slp, cla, kept, peak, a, u, w, vsize, RR, HH, out_dir, port):
    meshes = []
    for pts, tag, lab in ((slp, "sl", 2), (cla, "cl", 3)):
        sub = pts[rng.choice(len(pts), min(45000, len(pts)), replace=False)]
        meshes.append(_points(sub - peak, COL[tag], f"{tag} full bone", lab))
    disp = kept if len(kept) <= 8000 else kept[rng.choice(len(kept), 8000, replace=False)]
    meshes += _cubes(disp - peak, vsize * 10, C_GR, "growth", 0)
    sph = trimesh.creation.icosphere(subdivisions=2, radius=0.15)
    meshes.append(dict(name="peak", group=0, label=4, nfaces=len(sph.faces),
                       pos=_b64(sph.vertices.astype(np.float32)),
                       col=_b64(np.tile(C_PK, (len(sph.vertices), 1)).astype(np.uint8)),
                       idx=_b64(sph.faces.astype(np.uint16))))
    ang = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    ring = lambda h: h * a + RR * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w)
    vert = np.vstack([np.linspace(ring(-HH)[i], ring(HH)[i], 14) for i in range(0, 160, 20)])
    meshes.append(_points(np.vstack([ring(HH), ring(-HH), vert]), C_DS, "disc", 5))
    scene = dict(groups=["specimen"], meshes=meshes,
                 title="µCT · ectopic bone growth (α-fill clipped to bone)",
                 subtitle="Surgical (blue) & control (orange) tibia; RED = growth; cyan ● = centre, "
                          "yellow ring = disc.", partslabel="Layers",
                 labelnames={"0": "Growth", "2": "SL bone", "3": "CL bone", "4": "● Centre", "5": "Disc"},
                 labelcolors={"0": C_GR, "2": COL["sl"], "3": COL["cl"], "4": C_PK, "5": C_DS},
                 labeloff=[3])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sj = out_dir / "scene.json"; sj.write_text(json.dumps(scene))
    html = out_dir / "growth_viewer.html"
    subprocess.run([sys.executable, str(UCT / "build_artifact.py"), str(sj), str(html)],
                   check=True, stdout=subprocess.DEVNULL)
    if "<title>" not in html.read_text():
        html.write_text("<title>µCT growth viewer</title>\n" + html.read_text())
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    with socketserver.TCPServer(("localhost", port), handler) as httpd:
        url = f"http://localhost:{port}/growth_viewer.html"
        print(f"\n[viewer] serving at {url}  (Ctrl+C to stop)", flush=True)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[viewer] stopped.")


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Measure ectopic tibial bone growth (SL vs CL).")
    ap.add_argument("--sl-path", "--sl_path", dest="sl_path", required=True, help="surgical (transplant) .VOX path")
    ap.add_argument("--cl-path", "--cl_path", dest="cl_path", required=True, help="contralateral (control) .VOX path")
    ap.add_argument("--disc-radius", "--disc_radius", dest="disc_radius", type=float, default=1.2,
                    help="disc radius, mm (default 1.2)")
    ap.add_argument("--disc-height", "--disc_height", dest="disc_height", type=float, default=0.55,
                    help="disc full height, mm (default 0.55)")
    ap.add_argument("--hu-th", "--hu_th", dest="hu_th", type=float, default=200.0,
                    help="bone threshold as HU offset above each scan's Otsu level (default 200)")
    ap.add_argument("--visualize", action="store_true", help="serve the WebGL viewer on localhost")
    ap.add_argument("--port", type=int, default=8000, help="localhost port for --visualize (default 8000)")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default=None, help="viewer output dir (default: ./growth_out)")
    ap.add_argument("--gt-vox", "--gt_vox", dest="gt_vox", type=float, default=None,
                    help="optional ground-truth voxel count for a ratio")
    args = ap.parse_args()

    RR = args.disc_radius * STEP        # radius mm -> mesh
    HH = args.disc_height               # full height mm == half-height in mesh units (STEP=2)

    print(f"[1/4] loading & meshing SL {args.sl_path}", flush=True)
    slp, vsize, slmesh, p1m, p4m = tibia(args.sl_path, args.hu_th)
    print(f"[2/4] loading & meshing CL {args.cl_path}", flush=True)
    clp, _, clmesh, _, _ = tibia(args.cl_path, args.hu_th)
    print("[3/4] aligning CL -> SL", flush=True)
    cla = cl_to_sl(clmesh, slmesh)(clp)
    a, u, w = pca_axis(slp)
    peak = growth_centre(slp, cla, a, u, w, vsize, p1m, p4m)
    print("[4/4] measuring growth", flush=True)
    nvox, kept = growth_volume(slp, cla, peak, a, u, w, vsize, RR, HH)
    vmm = vsize / 2.0; mm3 = nvox * vmm ** 3

    print("\n==================== GROWTH ====================")
    print(f"  disc:      radius {args.disc_radius} mm, height {args.disc_height} mm")
    print(f"  threshold: per-scan Otsu + {args.hu_th:.0f} HU")
    print(f"  growth:    {nvox} voxels  =  {mm3:.4f} mm3")
    gt = args.gt_vox or GT_VOX
    print(f"  vs GT:     {mm3 / (gt * vmm ** 3):.2f}x  (GT = {int(gt)} vox"
          f"{' [B256M1 reference]' if args.gt_vox is None else ''})")
    print("================================================", flush=True)

    if args.visualize:
        build_and_serve(slp, cla, kept, peak, a, u, w, vsize, RR, HH,
                        args.out_dir or "growth_out", args.port)


if __name__ == "__main__":
    main()
