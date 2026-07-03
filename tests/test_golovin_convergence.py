"""Super-droplet convergence: the Golovin-benchmark error must fall ~ n_sd^(-1/2).

A reduced sweep for CI; the full-resolution study is validation/golovin_convergence.py.
"""
from validation.golovin_convergence import sweep, fit_slope


def test_error_decreases_and_converges_like_monte_carlo():
    n_sd_list = [128, 256, 512, 1024]
    errs = sweep(n_sd_list, n_ens=6, dt=2.0)
    # error must shrink with more super-droplets
    assert errs[-1] < errs[0] / 1.5, f"no convergence: {errs}"
    # log-log slope near the ideal Monte-Carlo -0.5 (generous band for a small ensemble)
    slope = fit_slope(n_sd_list, errs)
    assert -0.9 < slope < -0.25, f"convergence slope {slope:.2f} not ~ -0.5"
