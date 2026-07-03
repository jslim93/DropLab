"""Regression: ice super-droplets must NOT be grown by the LIQUID Koehler condensation step
-- they grow only by ice deposition. The bug grew frozen particles as liquid (wrong density
rho_liq, wrong latent heat l_v) on every ice timestep, in ADDITION to ice deposition, which
double-counted the vapour exchange and suppressed ice mass ~40% in mixed-phase clouds."""
import numpy as np
from droplab.flow2d_driver import _cond_local


def test_liquid_condensation_skips_ice():
    # one liquid + one ice super-droplet in the SAME strongly water-supersaturated cell
    M = np.array([5.0e-13, 5.0e-13])
    A = np.array([1.0e6, 1.0e6])
    Ns = np.array([1.0e-17, 1.0e-17])
    ka = np.array([0.6, 0.6])
    cidx = np.array([0, 0])
    supersat = np.array([0.05]); G = np.array([1.0e-9]); r0 = np.array([1.0e-7])
    afac = np.array([1.0e-9])
    phase = np.array([0, 1], dtype=np.int8)
    dM_cell = np.zeros(1)
    M0 = M.copy()

    _cond_local(M, A, Ns, ka, cidx, supersat, G, r0, afac, 1.0,
                True, True, True, dM_cell, phase)

    assert M[1] == M0[1]                 # ICE droplet untouched by liquid condensation
    assert M[0] != M0[0]                 # LIQUID droplet IS processed (grows/evaporates)
    # the per-cell condensed mass equals only the liquid droplet's change
    assert np.isclose(dM_cell[0], M[0] - M0[0])


def test_warm_caller_default_phase_is_all_liquid():
    # phase=None (warm-only kinematic caller) must behave as before: every drop grows
    M = np.array([5.0e-13, 5.0e-13]); A = np.array([1.0e6, 1.0e6])
    Ns = np.array([1.0e-17, 1.0e-17]); ka = np.array([0.6, 0.6]); cidx = np.array([0, 0])
    supersat = np.array([0.05]); G = np.array([1.0e-9]); r0 = np.array([1.0e-7])
    afac = np.array([1.0e-9]); dM_cell = np.zeros(1); M0 = M.copy()
    _cond_local(M, A, Ns, ka, cidx, supersat, G, r0, afac, 1.0, True, True, True, dM_cell)
    assert M[0] != M0[0] and M[1] != M0[1]   # both processed (no phase -> all liquid)
