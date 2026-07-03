"""Opt-in parallel per-cell collision (collide_parallel=True): each grid cell draws from
its own counter-based splitmix64 stream (salted by seed+step), so the run is DETERMINISTIC
for a fixed seed at any thread count — but it is a different (statistically equivalent)
random realization than the serial golden stream. Default (False) stays on the serial
global-RNG path, which tests/test_flow2d_golden.py pins bit-identical."""
import numpy as np

from droplab.flow2d_dynamic import run_flow2d_dynamic

_BASE = dict(Nx=24, Nz=24, X=1800, Z=1800, nt=40, dt=1.5, n_super=24 * 24 * 40,
             collect_every=40, seed=3, dtheta_bubble=1.2, RH0=0.95,
             N_modes=(100.0, 2.0), mu_um=(0.1, 1.0), sig=(1.8, 1.5), kappa=(1.0, 1.0))


def test_parallel_collision_is_deterministic():
    a = run_flow2d_dynamic(**_BASE, collide_parallel=True)
    b = run_flow2d_dynamic(**_BASE, collide_parallel=True)
    fa, fb = a["frames"][-1], b["frames"][-1]
    assert np.array_equal(fa["r_um"], fb["r_um"])       # same seed -> bit-identical rerun
    assert np.array_equal(fa["qc"], fb["qc"])


def test_parallel_collision_statistically_equivalent():
    s = run_flow2d_dynamic(**_BASE, collide_parallel=False)
    p = run_flow2d_dynamic(**_BASE, collide_parallel=True)
    qs = float(s["frames"][-1]["qc"].sum())
    qp = float(p["frames"][-1]["qc"].sum())
    # different RNG realization, same physics: cloud water within a loose Monte-Carlo band
    assert qp > 0.0 and qs > 0.0
    assert abs(qp - qs) / qs < 0.25
