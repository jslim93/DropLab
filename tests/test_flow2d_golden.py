"""2D-model regression guard for the performance optimisation.

The fast paths must reproduce the baseline physics state (captured in
validation/golden_2d.npz on un-optimised main) to within a very tight tolerance —
this is the operational definition of "no physics change". rtol is 1e-12 (not exact)
only to tolerate benign float re-association from vectorising/jitting; any real
physics change is orders of magnitude larger.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import pytest

from validation.golden_2d_setup import run_golden_2d

_GOLDEN = os.path.join(os.path.dirname(__file__), "..", "validation", "golden_2d.npz")


@pytest.mark.skipif(not os.path.exists(_GOLDEN), reason="golden_2d.npz not generated")
def test_2d_matches_golden():
    g = np.load(_GOLDEN)
    out = run_golden_2d()
    assert int(out["n_super"]) == int(g["n_super"]), "super-droplet count changed"
    assert np.allclose(out["theta"], g["theta"], rtol=1e-12, atol=0), "theta changed"
    assert np.allclose(out["qv"], g["qv"], rtol=1e-12, atol=0), "qv changed"
    assert np.allclose(out["M_sorted"], g["M_sorted"], rtol=1e-12, atol=0), "droplet mass changed"
    assert np.allclose(out["A_sorted"], g["A_sorted"], rtol=1e-12, atol=0), "multiplicity changed"
    assert abs(float(out["surf_precip"]) - float(g["surf_precip"])) \
        <= 1e-12 * abs(float(g["surf_precip"])), "surface precip changed"
