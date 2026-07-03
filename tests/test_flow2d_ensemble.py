"""The 2D parallel ensemble must produce results bit-identical to a serial sweep —
each member's seed fixes its RNG and the parallel numba kernels are
thread-order-independent, so running members in worker processes changes nothing but
the wall time.
"""
import matplotlib; matplotlib.use("Agg")

from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.soundings import DYCOMS, DYCOMS_RADIATION
from droplab.flow2d_ensemble import run_parallel

# tiny members so the test is fast; exercises the full path (collision, sediment,
# condensation, periodic, radiation).
_CFG = dict(nt=40, dt=1.0, Nx=32, Nz=24, X=1600.0, Z=1200.0, n_super=8000,
            sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True, N_modes=(200.,),
            pert_amp=0.1, nu=6, nu_scalar=1.5, collisions=True, switch_TICE=True,
            eps=0.01, sediment=True, collect_every=100000)


def _member(seed):
    """Module-level (so it pickles to loky workers); returns a small summary."""
    out = run_flow2d_dynamic(seed=seed, **_CFG)
    return (float(out["surf_precip"]), int(out["M"].size), float(out["theta"].sum()))


def test_parallel_ensemble_bit_identical_to_serial():
    seeds = [0, 1, 2, 3]
    serial = [_member(s) for s in seeds]
    parallel = run_parallel(seeds, _member, n_jobs=2, threads_per_worker=1)
    assert parallel == serial, "parallel ensemble result differs from serial"
