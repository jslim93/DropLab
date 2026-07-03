"""2D-model regression guard for the performance optimisation.

This is fundamentally a SAME-MACHINE guard: the fast/vectorised/numba paths must reproduce
the baseline physics state (captured in validation/golden_2d.npz on the un-optimised model)
bit-for-bit, so any real physics change is caught. rtol is 1e-12 (not exact) only to tolerate
benign float re-association from vectorising/jitting.

Cross-platform is a different matter. The stochastic collision step accepts a coalescence when
`p_crit > x_rand`; a ~1e-16 libm/FMA difference between CPUs flips that discrete comparison,
which changes WHICH super-droplets merge and cascades into O(1) differences in the per-droplet
mass/multiplicity arrays (bulk water still conserves). So `M_sorted`/`A_sorted` are only
meaningful on the machine the baseline was generated on. The emergent GRIDDED fields (theta, qv)
average over droplets and stay close (~1e-8 observed on x86), so off the baseline platform we
assert only those, at a loose tolerance that still catches any gross physics regression.
"""
import os
import platform
import numpy as np
import matplotlib; matplotlib.use("Agg")
import pytest

from validation.golden_2d_setup import run_golden_2d

_GOLDEN = os.path.join(os.path.dirname(__file__), "..", "validation", "golden_2d.npz")
# The baseline golden_2d.npz was generated on macOS/arm64 (Apple silicon). Full bit-identity
# is asserted there; other platforms get a gross-regression sanity check on gridded fields.
_BASELINE_PLATFORM = (platform.system() == "Darwin"
                      and platform.machine() in ("arm64", "aarch64"))


@pytest.mark.skipif(not os.path.exists(_GOLDEN), reason="golden_2d.npz not generated")
def test_2d_matches_golden():
    g = np.load(_GOLDEN)
    out = run_golden_2d()
    assert int(out["n_super"]) == int(g["n_super"]), "super-droplet count changed"

    if _BASELINE_PLATFORM:
        # same-machine bit-identity guard (the real regression protection)
        assert np.allclose(out["theta"], g["theta"], rtol=1e-12, atol=0), "theta changed"
        assert np.allclose(out["qv"], g["qv"], rtol=1e-12, atol=0), "qv changed"
        assert np.allclose(out["M_sorted"], g["M_sorted"], rtol=1e-12, atol=0), "droplet mass changed"
        assert np.allclose(out["A_sorted"], g["A_sorted"], rtol=1e-12, atol=0), "multiplicity changed"
        assert abs(float(out["surf_precip"]) - float(g["surf_precip"])) \
            <= 1e-12 * abs(float(g["surf_precip"])), "surface precip changed"
    else:
        # cross-platform: per-droplet collision trajectory is not reproducible (see module
        # docstring); assert only the emergent gridded fields, loosely, to catch gross breakage
        assert np.allclose(out["theta"], g["theta"], rtol=1e-6, atol=0), "theta grossly changed"
        assert np.allclose(out["qv"], g["qv"], rtol=1e-6, atol=0), "qv grossly changed"
