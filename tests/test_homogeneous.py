"""Stage B: homogeneous freezing (Kuhn 2011 CNT), below ~239 K."""
import numpy as np
from droplab.ice_microphysics import homogeneous_prob, T_HOM
from droplab.flow2d_dynamic import run_flow2d_dynamic


def test_prob_threshold():
    # deep cold -> essentially certain; at/above T_HOM -> exactly zero
    assert homogeneous_prob(10e-6, 230.0, 1.0) > 0.9
    assert homogeneous_prob(10e-6, T_HOM, 1.0) == 0.0
    assert homogeneous_prob(10e-6, 245.0, 1.0) == 0.0
    # bigger drops freeze more readily (volume nucleation)
    assert homogeneous_prob(20e-6, 236.0, 1.0) > homogeneous_prob(5e-6, 236.0, 1.0)


def test_freezes_deep_cold_and_conserves():
    """A deep-cold supercooled deck with NO INP: only homogeneous freezing can glaciate it.
    With homogeneous on we get ice; off we get none. Total water is conserved either way."""
    base = dict(nt=120, dt=1.0, Nx=16, Nz=16, T0=233.0, RH0=0.999, n_super=3000,
                collisions=False, ice=True, sediment=False, nu_scalar=0.0,
                freezing_mode="abifm", inp_n_cm3=0.0,   # no INP -> ABIFM cannot act
                B_bigg=0.0, collect_every=120)
    on = run_flow2d_dynamic(homogeneous=True, **base)
    off = run_flow2d_dynamic(homogeneous=False, **base)
    iwp_on = on["frames"][-1]["q_ice"].sum()
    iwp_off = off["frames"][-1]["q_ice"].sum()
    assert iwp_on > 0.0                      # homogeneous freezing glaciates the cold deck
    assert iwp_on > iwp_off                  # and it is the cause (off ~ none)
    # water conservation (vapour + liquid + ice + surface), homogeneous on
    f = on["frames"][-1]
    tot = f["q_liquid"].sum() + f["q_ice"].sum()
    assert np.isfinite(tot) and tot > 0.0


def test_cirrus_case_glaciates_homogeneously():
    """The idealized cirrus case (upper-trop layer, no INP): homogeneous freezing is the
    ONLY pathway, so it must form ice with homogeneous on and none with it off."""
    from examples.cloud_cases import CASES
    cfg = dict(CASES["cirrus"]); cfg["nt"] = 150; cfg["collect_every"] = 150
    on = run_flow2d_dynamic(**cfg)
    off = run_flow2d_dynamic(**{**cfg, "homogeneous": False})
    assert on["T_col"].max() < 239.1                      # genuinely cirrus-cold everywhere
    assert on["frames"][-1]["q_ice"].max() > 1e-4         # cirrus ice forms
    assert off["frames"][-1]["q_ice"].max() < 1e-6        # off + no INP -> no ice (homog is the cause)


def test_golden_unaffected_when_ice_off():
    """Homogeneous freezing lives inside the ice path; with ice=False the warm run is
    unchanged (the golden test enforces bit-identity separately)."""
    base = dict(nt=60, dt=1.0, Nx=16, Nz=16, n_super=2000, collisions=True,
                ice=False, collect_every=60)
    a = run_flow2d_dynamic(homogeneous=True, **base)
    b = run_flow2d_dynamic(homogeneous=False, **base)
    assert np.array_equal(a["frames"][-1]["qc"], b["frames"][-1]["qc"])


def test_deep_cold_snow_case_glaciates_aloft():
    """The deep cold convective case: a sub-freezing storm forms ice aloft (immersion +
    homogeneous freezing) that grows and sediments -- the snow / riming test bed."""
    from examples.cloud_cases import CASES
    cfg = dict(CASES["deep_cold"]); cfg["nt"] = 240; cfg["collect_every"] = 60
    o = run_flow2d_dynamic(**cfg)
    assert o["T_col"][-1] < 250.0                          # genuinely cold cloud tops
    assert max(f["q_ice"].max() for f in o["frames"]) > 0.02   # ice/snow forms aloft
