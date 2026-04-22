"""
Parameter sweep over articular_percentile on two sets of STL files:
  1. SMC-aligned, unsmoothed  (results_CT_20260201_110806/)
  2. PCA+ICP-aligned, smoothed (results_smoothed/)  — no SMC
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import _articular_distance

SCALE  = 1.25e-4
TARGET = 0.003

def sweep(label, out_dir):
    out_dir = Path(out_dir)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'percentile':>12}  {'ref_vol':>10}  {'sam_vol':>10}  {'diff':>10}  {'inj_vol':>10}  {'ratio':>8}")
    print("-" * 70)
    for p in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        r_d, r_l, r_n, r_v = _articular_distance(str(out_dir/"ref_2.stl"),    str(out_dir/"ref_1.stl"),    p)
        s_d, s_l, s_n, s_v = _articular_distance(str(out_dir/"sample_2.stl"), str(out_dir/"sample_1.stl"), p)
        diff = r_v - s_v
        inj  = diff * SCALE
        ratio = inj / TARGET
        sign = "✓" if inj > 0 else "✗"
        print(f"{p:>11}%  {r_v:>10.1f}  {s_v:>10.1f}  {diff:>10.2f}  {inj:>10.6f}  {ratio:>7.2f}x {sign}")

sweep("SMC-aligned, unsmoothed  (sigma=0, laplacian=0)",  "results_CT_20260201_110806")
sweep("PCA+ICP-aligned, smoothed (sigma=3, laplacian=100)", "results_smoothed")
