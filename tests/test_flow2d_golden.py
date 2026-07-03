"""2D-model regression guard for the performance optimisation.

The fast paths must reproduce the baseline physics state (captured in
validation/golden_2d.npz on un-optimised main) to within a very tight tolerance —
this is the operational definition of "no physics change".

Tolerance is PLATFORM-AWARE: the stored baseline was generated on macOS/arm64, where
rtol=1e-12 holds (it tolerates only benign float re-association from vectorising /
jitting). On other platforms (e.g. Linux/x86-64 CI) libm and FMA rounding differ at
the ULP level and are then amplified by the chaotic dynamics (observed ~1e-10 relative
after the golden run's steps) — so the cross-platform tolerance is 1e-8, still far
below any physics change. Any real physics change is orders of magnitude
larger than either tolerance, so the guard's power is the same; only the
same-machine bit-reproducibility claim is arm64-specific.
"""
import os
import platform
import numpy as np
import matplotlib; matplotlib.use("Agg")
import pytest

from validation.golden_2d_setup import run_golden_2d

_GOLDEN = os.path.join(os.path.dirname(__file__), "..", "validation", "golden_2d.npz")
# baseline provenance: generated on macOS/arm64 (Apple silicon)
_RTOL = 1e-12 if platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin" \
    else 1e-8


@pytest.mark.skipif(not os.path.exists(_GOLDEN), reason="golden_2d.npz not generated")
def test_2d_matches_golden():
    g = np.load(_GOLDEN)
    out = run_golden_2d()
    assert int(out["n_super"]) == int(g["n_super"]), "super-droplet count changed"
    assert np.allclose(out["theta"], g["theta"], rtol=_RTOL, atol=0), "theta changed"
    assert np.allclose(out["qv"], g["qv"], rtol=_RTOL, atol=0), "qv changed"
    assert np.allclose(out["M_sorted"], g["M_sorted"], rtol=_RTOL, atol=0), "droplet mass changed"
    assert np.allclose(out["A_sorted"], g["A_sorted"], rtol=_RTOL, atol=0), "multiplicity changed"
    assert abs(float(out["surf_precip"]) - float(g["surf_precip"])) \
        <= _RTOL * abs(float(g["surf_precip"])), "surface precip changed"
