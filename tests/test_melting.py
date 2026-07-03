"""Cold-cloud cycle -- melting: ice that reaches air warmer than 0 C reverts to liquid,
absorbing the latent heat of fusion (a heat sink, the exact opposite sign of freezing).

`_melt` is unit-tested deterministically (the physics: phase flip, mass conservation,
cold-cell no-op). The integration test uses a *shallow mixed-phase* deck -- supercooled
cloud just above a warm base, glaciated by heavy INP (Bigg) -- because that is the only
affordable regime in an idealized bubble where ice actually descends into the melt layer:
in a deep storm the homogeneous ice forms ~4 km above the 0 C level and cannot sediment
that far against the updraft within a short run (a long-stratiform problem, not a unit test).
"""
import numpy as np
import droplab.flow2d_dynamic as fd
from droplab.ice_microphysics import _melt, T_MELT


def test_melt_flips_warm_ice_and_conserves_mass():
    # [0] ice in a warm cell, [1] ice in a cold cell, [2] liquid in a warm cell
    M = np.array([2.0e-3, 3.0e-3, 1.0e-3])
    A = np.array([1.0e6, 1.0e6, 1.0e6])
    phase = np.array([1, 1, 0], dtype=np.int8)
    cidx = np.array([0, 1, 0])
    T_c = np.array([T_MELT + 5.0, T_MELT - 5.0])   # cell 0 warm, cell 1 cold
    melted = np.zeros(2)
    M0 = M.copy()

    _melt(M, A, phase, cidx, T_c, melted)

    # warm ice -> liquid; cold ice stays ice; the liquid drop is untouched
    assert phase[0] == 0 and phase[1] == 1 and phase[2] == 0
    # the melted mass is booked to the warm cell only, equal to that droplet's mass
    assert np.isclose(melted[0], M0[0]) and melted[1] == 0.0
    # a phase flip never changes the water mass (melting conserves mass)
    assert np.array_equal(M, M0)


def test_melt_is_noop_at_or_below_freezing():
    # the threshold is strict: exactly 0 C does NOT melt
    M = np.array([2.0e-3, 2.0e-3]); A = np.array([1.0e6, 1.0e6])
    phase = np.array([1, 1], dtype=np.int8); cidx = np.array([0, 1])
    T_c = np.array([T_MELT, T_MELT - 0.1]); melted = np.zeros(2)
    _melt(M, A, phase, cidx, T_c, melted)
    assert phase[0] == 1 and phase[1] == 1
    assert melted.sum() == 0.0


def _shallow_mixed_phase(**over):
    """Warm base (0-1.5 km, T>0 C) under a supercooled cloud (1.5-3.5 km) that heavy INP
    glaciates via Bigg -> ice falls ~1 km into the warm base and melts within the run."""
    snd = {"name": "shallow mixed-phase melt bed",
           "z": [0, 800, 1600, 2400, 3500, 5000],
           "theta": [281., 284., 286., 288., 290., 294.],
           "qv": [9., 7., 5.5, 4., 2.5, 1.2]}
    cfg = dict(Nx=40, Nz=48, X=4000, Z=5000, dt=2.0, nt=500, collect_every=500,
               n_super=16000, collisions=True, ice=True, sediment=True,
               freezing_mode="bigg", a_bigg=0.66, B_bigg=1.0e4, inp_n_cm3=20.0,
               homogeneous=False, dtheta_bubble=3.0, bubble_r=700., bubble_z=600.,
               eps=0.02, periodic_x=True, seed=3, sounding=snd)
    cfg.update(over)
    return cfg


def test_melting_fires_in_shallow_mixed_phase_and_conserves():
    """Instrument _melt: in a deck with a real melting level, ice sediments into the warm
    base and melts, the run stays finite and total condensate is conserved."""
    orig = fd._melt
    tally = {"melted": 0.0, "steps": 0}

    def spy(M, A, phase, cidx, T_c, melted):
        orig(M, A, phase, cidx, T_c, melted)
        s = float(melted.sum())
        tally["melted"] += s
        tally["steps"] += (s > 0.0)

    fd._melt = spy
    try:
        o = fd.run_flow2d_dynamic(**_shallow_mixed_phase())
    finally:
        fd._melt = orig

    f = o["frames"][-1]
    # a genuine melting level exists (warm base) and ice formed aloft
    assert float(o["T_col"].max()) > T_MELT
    assert f["q_ice"].max() > 0.0
    # melting demonstrably fired (ice reached >0 C air and reverted to liquid)
    assert tally["steps"] > 0 and tally["melted"] > 0.0
    # the whole field stays finite -- no blow-up from the phase changes
    assert np.isfinite(f["q_ice"]).all() and np.isfinite(f["q_liquid"]).all()
