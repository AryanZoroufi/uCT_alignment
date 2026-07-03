"""
Central per-pair config for the tibia bridge/growth measurement.

Extracted verbatim from the constants that were HARDCODED across
precompute_crop.py, recompute_736.py, recompute_variants.py, paint_bridge.py
and paint_slices.py so all downstream tools reference one source of truth.

Pure data + a tiny accessor. No heavy imports.

Per-pair fields:
  cl_vox   absolute path to the contralateral (control) .VOX
  sl_vox   absolute path to the surgical/transplant (sample) .VOX
  occ      absolute path to the *_occ dir holding {tag}_part1.stl,
           {tag}_part4.stl and {tag}.stl (occupancy meshes)
  gt_vox   Dragonfly human-paint ground-truth voxel count (None if unknown)
  step     downsample STEP used when building the occ meshes (mesh verts
           are in STEP-downsampled voxel units; native idx = verts / (vmm*STEP))
"""

VOX_BASE = ("/home/aryan/Projects/uct_backup/uCT_alignment/new data/"
            "Transplantation_21G_D14/Matrigel only/Matrigel only")
RECON_BASE = "/home/aryan/Projects/uct_backup/uCT_alignment/bones_to_recon"

# Ground truth: B256M1 SL Dragonfly human paint = 366110 vox = 0.101363 mm^3.
GT_MM3_B256M1 = 0.101363

PAIRS = {
    # Active / verified pair.
    "B256M1": {
        "cl_vox": f"{VOX_BASE}/B256M1 CL/CT_20251229_154135/CT_20251229_154135.VOX",
        "sl_vox": f"{VOX_BASE}/B256M1 SL/CT_20251229_154909/CT_20251229_154909.VOX",
        "occ": f"{RECON_BASE}/B256M1_occ",
        "gt_vox": 366110,
        "step": 2,
    },
    # Next target (the bone_4 pair). occ dir not built yet; gt unknown.
    "B256M7": {
        "cl_vox": f"{VOX_BASE}/B256M7 CL/CT_20251229_162403/CT_20251229_162403.VOX",
        "sl_vox": f"{VOX_BASE}/B256M7 SL/CT_20251229_163017/CT_20251229_163017.VOX",
        "occ": f"{RECON_BASE}/B256M7_occ",
        "gt_vox": None,
        "step": 2,
    },
}


# Chamfer-registration variants: same VOX/GT, parts read from {occ}_chamfer
# (produced by _segment_via_atlas(..., registration="chamfer")).
for _base in ("B256M1", "B256M7"):
    _c = dict(PAIRS[_base]); _c["occ"] = _c["occ"] + "_chamfer"
    PAIRS[_base + "c"] = _c


def get(pair_name):
    """Return the config dict for a pair (e.g. get('B256M1'))."""
    if pair_name not in PAIRS:
        raise KeyError(
            f"unknown pair {pair_name!r}; known pairs: {sorted(PAIRS)}")
    return PAIRS[pair_name]


if __name__ == "__main__":
    for name in sorted(PAIRS):
        p = PAIRS[name]
        print(f"[{name}] gt_vox={p['gt_vox']} step={p['step']}")
        print(f"    cl_vox={p['cl_vox']}")
        print(f"    sl_vox={p['sl_vox']}")
        print(f"    occ={p['occ']}")
