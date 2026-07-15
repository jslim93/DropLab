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

def test_sublimated_ice_reverts_to_aerosol():
    """A crystal that sublimates down to its dry aerosol core is handed BACK to the
    aerosol population (phase -> 0, mass = dry-core liquid mass), NOT left as an inert
    M=0 ghost that permanently leaks the CCN/INP budget. A crystal that is still
    GROWING (even if momentarily below the core radius, e.g. a freshly frozen haze)
    must stay ice — the revert is gated on net sublimation this step."""
    import numpy as np
    from droplab.ice_microphysics import _ice_deposition
    from droplab.parameters import rho_ice, rho_aero, pi

    A = np.array([1.0e6, 1.0e6])
    r_dry = 0.1e-6
    Ns = np.full(2, A[0] * 4.0 / 3.0 * pi * rho_aero * r_dry ** 3)
    # SD0: a large crystal in strongly ice-SUBsaturated air -> sublimates past its core
    # SD1: a sub-core freshly frozen haze in ice-SUPERsaturated air -> should grow, stay ice
    r0_ice, r1_ice = 5.0e-6, 0.09e-6
    M = np.array([A[0] * 4.0 / 3.0 * pi * rho_ice * r0_ice ** 3,
                  A[1] * 4.0 / 3.0 * pi * rho_ice * r1_ice ** 3])
    phase = np.array([1, 1], dtype=np.int8)
    cidx = np.array([0, 1])
    S_ice = np.array([-0.9, 0.5])        # cell 0 subsaturated, cell 1 supersaturated
    G_ice = np.array([1.0e-9, 1.0e-11])
    dM = np.zeros(2)

    _ice_deposition(M, A, Ns, phase, cidx, S_ice, G_ice, 1.0, dM)

    # SD0 reverted to aerosol: liquid phase, mass = dry-core liquid mass, Ns intact
    assert phase[0] == 0
    core_liq = A[0] * 4.0 / 3.0 * pi * 1000.0 * r_dry ** 3
    assert np.isclose(M[0], core_liq, rtol=1e-6)
    # SD1 still ice and grew (freshly frozen sub-core haze not spuriously reverted)
    assert phase[1] == 1 and M[1] > A[1] * 4.0 / 3.0 * pi * rho_ice * r1_ice ** 3
    # Ns (the aerosol residual) is never touched by deposition
    assert np.allclose(Ns, A * 4.0 / 3.0 * pi * rho_aero * r_dry ** 3)
