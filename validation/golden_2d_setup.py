"""Deterministic 2D-model run shared by the golden generator and the regression test.

run_flow2d_dynamic is bit-reproducible with fixed seeds (np.random + numba RNG), so a
fixed config gives a reproducible final state. This is the regression anchor that
guards the 2D performance optimisation: any fast path MUST reproduce this state, so we
KNOW the physics is unchanged. Captured quantities are order-independent (grid fields
+ sorted droplet arrays) so a reordering optimisation that preserves the physics still
passes.
"""
import numpy as np

from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.soundings import DYCOMS, DYCOMS_RADIATION

# small but exercises every path: condensation substeps, collision (+TICE turb),
# sedimentation, periodic advection, cloud-top radiation.
GOLDEN_CFG = dict(nt=80, dt=1.0, Nx=40, Nz=28, X=2000.0, Z=1200.0, n_super=12000,
                  sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
                  N_modes=(200.,), pert_amp=0.1, nu=6, nu_scalar=1.5, collisions=True,
                  switch_TICE=True, eps=0.01, sediment=True, collect_every=100000, seed=3)


def run_golden_2d():
    """Run the fixed 2D config and return order-independent physics invariants."""
    out = run_flow2d_dynamic(**GOLDEN_CFG)
    return dict(
        theta=out["theta"],
        qv=out["qv"],
        surf_precip=np.float64(out["surf_precip"]),
        M_sorted=np.sort(out["M"]),
        A_sorted=np.sort(out["A"]),
        n_super=np.int64(out["M"].size),
    )
