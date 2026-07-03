#!/usr/bin/env python3
"""compare_masks.py — score two boolean 3D voxel masks against each other.

Both masks must share the same shape and crop frame (native-res crop as
produced by the uCT pipeline: see /tmp/{tag}_crop.npz). Volumes use the
standard voxel pitch vmm = 0.00643 mm/voxel, so a physical volume is simply
    mm^3 = voxel_count * vmm**3

Typical use: compare a human-painted growth mask against automated candidates
(e.g. /tmp/cand736_fullxsec.npy vs /tmp/cand736_oneside.npy) and, optionally,
against the Dragonfly ground-truth voxel count.

Dependency-light: numpy only.
"""
import sys
import numpy as np

VMM = 0.00643          # mm per native voxel (pipeline convention)
GT_VOX = 366110        # B256M1 SL Dragonfly hand-paint ground truth (voxels)


def load_mask(path):
    """Load a .npy file and return a boolean 3D ndarray."""
    arr = np.load(path)
    return np.asarray(arr).astype(bool)


def compare(m1, m2, vmm=VMM):
    """Compare two boolean masks of identical shape.

    Returns a dict with voxel counts, physical volumes (mm^3), the
    intersection / union counts, Dice, IoU, and the exclusive counts.

    Raises ValueError on a shape mismatch (masks must be in the same crop
    frame for the overlap metrics to be meaningful).
    """
    m1 = np.asarray(m1).astype(bool)
    m2 = np.asarray(m2).astype(bool)
    if m1.shape != m2.shape:
        raise ValueError(
            f"mask shape mismatch: m1 {m1.shape} vs m2 {m2.shape} — "
            "both masks must be in the same crop frame (same shape)."
        )

    vox1 = int(m1.sum())
    vox2 = int(m2.sum())
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    only1 = int(np.logical_and(m1, ~m2).sum())
    only2 = int(np.logical_and(m2, ~m1).sum())

    denom_dice = vox1 + vox2
    dice = (2.0 * inter / denom_dice) if denom_dice else 1.0
    iou = (inter / union) if union else 1.0

    v3 = vmm ** 3
    return {
        "vox1": vox1,
        "vox2": vox2,
        "mm3_1": vox1 * v3,
        "mm3_2": vox2 * v3,
        "intersection": inter,
        "union": union,
        "dice": dice,
        "iou": iou,
        "only1": only1,
        "only2": only2,
    }


def vs_gt(mask, gt_vox=GT_VOX):
    """Return (vox, ratio_to_gt) for a single mask vs a ground-truth count."""
    mask = np.asarray(mask).astype(bool)
    vox = int(mask.sum())
    ratio = (vox / gt_vox) if gt_vox else float("nan")
    return vox, ratio


def _report(r, gt_vox=None, vmm=VMM):
    lines = []
    lines.append("=" * 56)
    lines.append("mask comparison")
    lines.append("=" * 56)
    lines.append(f"  vox1          : {r['vox1']:>10d}   ({r['mm3_1']:.6f} mm^3)")
    lines.append(f"  vox2          : {r['vox2']:>10d}   ({r['mm3_2']:.6f} mm^3)")
    lines.append(f"  intersection  : {r['intersection']:>10d}")
    lines.append(f"  union         : {r['union']:>10d}")
    lines.append(f"  only in m1    : {r['only1']:>10d}")
    lines.append(f"  only in m2    : {r['only2']:>10d}")
    lines.append(f"  Dice          : {r['dice']:.6f}")
    lines.append(f"  IoU           : {r['iou']:.6f}")
    if gt_vox is not None:
        lines.append("-" * 56)
        lines.append(f"  ground truth  : {gt_vox:>10d} vox "
                     f"({gt_vox * vmm**3:.6f} mm^3)")
        lines.append(f"  vox1 / GT     : {r['vox1'] / gt_vox:.6f}")
        lines.append(f"  vox2 / GT     : {r['vox2'] / gt_vox:.6f}")
    lines.append("=" * 56)
    return "\n".join(lines)


def main(argv):
    if len(argv) < 3:
        print("usage: python compare_masks.py A.npy B.npy [GT_vox_count]",
              file=sys.stderr)
        return 2
    path1, path2 = argv[1], argv[2]
    gt_vox = int(argv[3]) if len(argv) > 3 else None

    m1 = load_mask(path1)
    m2 = load_mask(path2)
    print(f"A: {path1}  shape={m1.shape}")
    print(f"B: {path2}  shape={m2.shape}")

    r = compare(m1, m2)
    print(_report(r, gt_vox=gt_vox))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
