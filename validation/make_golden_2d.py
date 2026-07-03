"""Generate the 2D golden snapshot from the CURRENT (baseline) code.

Run ONCE on the un-optimised baseline:  python -m validation.make_golden_2d
Saves validation/golden_2d.npz, which tests/test_flow2d_golden.py guards against.
Do NOT regenerate after optimising — that would defeat the regression guard.
"""
import numpy as np
from validation.golden_2d_setup import run_golden_2d

if __name__ == "__main__":
    g = run_golden_2d()
    np.savez("validation/golden_2d.npz", **g)
    print("wrote validation/golden_2d.npz")
    print(f"  n_super={int(g['n_super'])}  surf_precip={float(g['surf_precip']):.6e}")
    print(f"  theta[mean]={g['theta'].mean():.6f}  qv[mean]={g['qv'].mean():.6e}")
    print(f"  M_sorted[-3:]={g['M_sorted'][-3:]}")
