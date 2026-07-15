import numpy as np
import pytest
from droplab.ice_microphysics import esati, qvs_ice
from droplab.condensation import esatw
from droplab.collision_soa import seed_numba_rng


def test_esati_below_esatw_when_supercooled():
    # over ice the saturation vapour pressure is LOWER than over water below 0C
    # (this gap is the engine of the Bergeron process)
    for T in (273.15, 268.15, 263.15, 253.15):
        assert esati(T) < esatw(T)


def test_esati_known_value_250K():
    # Murphy & Koop (2005) over ice at 250 K ~ 76.0 Pa
    assert esati(250.0) == pytest.approx(76.0, rel=0.02)


def test_esati_near_triple_point():
    # at the triple point ice and water saturation nearly coincide (~611 Pa)
    assert esati(273.16) == pytest.approx(611.6, rel=0.01)


def test_qvs_ice_positive_and_small():
    q = qvs_ice(263.15, 9.0e4)
    assert 0.0 < q < 0.01


from droplab.ice_microphysics import g_ice, ice_grow_r2


def test_g_ice_positive():
    assert g_ice(263.15, 9.0e4) > 0.0


def test_ice_grows_when_supersaturated_over_ice():
    # at water saturation, S_ice > 0 -> a 20 um ice sphere must grow
    T, P = 263.15, 9.0e4
    G = g_ice(T, P)
    r0 = 20e-6
    r1 = ice_grow_r2(r0, 0.05, G, 1.0)  # S_ice=5%, dt=1s
    assert r1 > r0


def test_ice_shrinks_when_subsaturated_and_clamps_nonneg():
    T, P = 263.15, 9.0e4
    G = g_ice(T, P)
    r1 = ice_grow_r2(1e-6, -1.0, G, 100.0)  # strongly subsaturated, long step
    assert r1 >= 0.0  # r^2-law clamps at zero, never NaN


from droplab.ice_microphysics import bigg_prob, _bigg_freeze


def test_bigg_no_freezing_above_zero():
    # warm drop: zero probability
    V = 4.0 / 3.0 * np.pi * (10e-6) ** 3
    assert bigg_prob(V, 275.0, 1.0, 0.66, 100.0) == 0.0


def test_bigg_prob_increases_as_it_cools():
    V = 4.0 / 3.0 * np.pi * (10e-6) ** 3
    p1 = bigg_prob(V, 268.0, 1.0, 0.66, 100.0)
    p2 = bigg_prob(V, 258.0, 1.0, 0.66, 100.0)
    assert 0.0 < p1 < p2 <= 1.0


def test_bigg_bigger_drops_freeze_more_readily():
    Vbig = 4.0 / 3.0 * np.pi * (40e-6) ** 3
    Vsmall = 4.0 / 3.0 * np.pi * (5e-6) ** 3
    assert bigg_prob(Vbig, 260.0, 1.0, 0.66, 100.0) > bigg_prob(Vsmall, 260.0, 1.0, 0.66, 100.0)


def test_bigg_freeze_fraction_statistical():
    # tests the MECHANISM: at a regime where P~0.46, the all-or-nothing frozen
    # fraction over many identical super-droplets matches bigg_prob. (Cloud drops
    # at warm mixed-phase temps barely freeze under classical Bigg — that is
    # physically correct; this test just verifies the stochastic draw, not the demo.)
    n = 20000
    A = np.full(n, 1.0e6)
    r = 15e-6
    M = np.full(n, 4.0 / 3.0 * np.pi * 1000.0 * r ** 3 * A[0])
    phase = np.zeros(n, np.int8)
    T_c = np.full(n, 243.0)
    cidx = np.zeros(n, np.int64)
    frozen_mass = np.zeros(1)
    a, B = 0.66, 1.0e5
    seed_numba_rng(0)
    _bigg_freeze(M, A, phase, cidx, T_c, 1.0, a, B, frozen_mass)
    V = M[0] / (A[0] * 1000.0)
    expected = bigg_prob(V, 243.0, 1.0, a, B)
    frac = phase.mean()
    assert frac == pytest.approx(expected, rel=0.05)
    # frozen_mass accounts for exactly the frozen super-droplets' water mass
    assert frozen_mass[0] == pytest.approx(M[0] * phase.sum(), rel=1e-12)


from droplab.ice_microphysics import ice_fall_speed
from droplab.collision import ws_drops_beard


def test_ice_falls_slower_than_equal_mass_rain():
    # a 1 mm-diameter ice sphere falls much slower than a 1 mm rain drop
    r = 0.5e-3
    A = 1.0
    M_ice = 4.0 / 3.0 * np.pi * 916.7 * r ** 3 * A
    v_ice = ice_fall_speed(M_ice, A)
    # equal-MASS liquid drop radius
    r_liq = (M_ice / (A * 4.0 / 3.0 * np.pi * 1000.0)) ** (1.0 / 3.0)
    v_rain = ws_drops_beard(r_liq, 1.0, 1000.0, 9.0e4, 270.0)
    assert 0.0 < v_ice < v_rain


def test_ice_fall_speed_monotonic_in_size():
    A = 1.0
    small = 4.0 / 3.0 * np.pi * 916.7 * (1e-4) ** 3 * A
    big = 4.0 / 3.0 * np.pi * 916.7 * (1e-3) ** 3 * A
    assert ice_fall_speed(big, A) > ice_fall_speed(small, A)


from droplab.flow2d_dynamic import run_flow2d_dynamic


def test_supercooled_layer_starts_freezing():
    # cold air + high INP efficiency (B) -> some immersion freezing within minutes
    out = run_flow2d_dynamic(nt=120, dt=1.5, Nx=24, Nz=24, T0=245.0, RH0=0.98,
                             n_super=6000, collisions=False, ice=True,
                             freezing_mode="bigg", B_bigg=1.0e6,
                             sediment=False, collect_every=40)
    assert int(out["phase"].sum()) > 0     # some super-droplets froze


def test_freezing_warms_via_latent_heat_of_fusion():
    cold = dict(nt=120, dt=1.5, Nx=24, Nz=24, T0=245.0, RH0=0.98, n_super=6000,
                collisions=False, sediment=False, freezing_mode="bigg", B_bigg=1.0e6,
                collect_every=40)
    out_ice = run_flow2d_dynamic(ice=True, **cold)
    out_noice = run_flow2d_dynamic(ice=False, **cold)
    # fusion releases heat: the ice run is no colder on domain mean
    assert out_ice["theta"].mean() >= out_noice["theta"].mean() - 1e-6


def test_ice_off_is_default_and_no_ice_forms():
    out = run_flow2d_dynamic(nt=30, dt=1.5, Nx=16, Nz=16, n_super=3000,
                             collisions=False, collect_every=10)
    # with ice off (default) there is no phase array exposed / all liquid
    assert out.get("phase") is None or int(np.asarray(out["phase"]).sum()) == 0


def test_ice_flag_threads_phase_array():
    out = run_flow2d_dynamic(nt=30, dt=1.5, Nx=16, Nz=16, n_super=3000,
                             collisions=True, ice=True, collect_every=10)
    assert "phase" in out
    assert out["phase"].shape[0] == out["M"].shape[0]   # phase survives collision/sediment masks


def test_frames_split_liquid_and_ice_and_phase():
    out = run_flow2d_dynamic(nt=60, dt=1.5, Nx=16, Nz=16, T0=258.0, RH0=0.98,
                             n_super=4000, collisions=False, ice=True,
                             sediment=True, collect_every=30)
    f = out["frames"][-1]
    for key in ("q_liquid", "q_ice", "phase"):
        assert key in f
    assert f["q_liquid"].shape == f["qc"].shape


def test_warm_run_ice_keys_absent_and_golden_safe():
    out = run_flow2d_dynamic(nt=30, dt=1.5, Nx=16, Nz=16, n_super=3000,
                             collisions=False, collect_every=10)   # ice=False default
    f = out["frames"][-1]
    assert "q_ice" not in f          # warm frames are unchanged (no new keys)


def test_glaciation_box_conserves_water_and_glaciates():
    """Supercooled deck with ice co-located with liquid: the ice grows by deposition
    (WBF) at the liquid's expense. Total water conserved; ice grows, liquid shrinks.

    Ice is INJECTED into the liquid cloud (rather than relying on spontaneous Bigg
    freezing) so the crystals sit in the ice-supersaturated, liquid-buffered air where
    WBF genuinely proceeds. The earlier Bigg-only box "glaciated" only through the
    sublimated-ice-as-M=0-ghost artifact (dynamics swept the frozen drops into dry,
    ice-SUBsaturated pockets, where correct physics now reverts them to aerosol); it
    was validating that artifact, not WBF. This co-located box tests the real thing."""
    base = dict(nt=200, dt=1.0, Nx=16, Nz=16, T0=250.0, RH0=0.999, n_super=4000,
                collisions=False, ice=True, freezing_mode="bigg", B_bigg=1.0,
                sediment=False, nu_scalar=0.0, collect_every=20)
    inp_spec = dict(t_inject=20.0, x_frac=(0.3, 0.7), z_lo=50.0, z_hi=400.0,
                    N_cm3=0.05, r_um=0.5, r_wet_um=3.0, kappa=0.0, n_super=400,
                    phase="ice")
    out = run_flow2d_dynamic(seeding=inp_spec, **base)
    fr = out["frames"]
    def total_water(f):
        # qv is kg/kg; q_liquid/q_ice frames are reported in g/kg -> back to kg/kg
        return float(f["qv"].sum() + (f["q_liquid"].sum() + f["q_ice"].sum()) / 1e3)
    tw0, tw1 = total_water(fr[0]), total_water(fr[-1])
    assert tw1 == pytest.approx(tw0, rel=2e-3)              # water conserved
    # compare the frame just AFTER injection (fr[1], step 20) with the end
    assert fr[-1]["q_ice"].sum() > fr[1]["q_ice"].sum()        # ice grew by WBF
    assert fr[-1]["q_liquid"].sum() < fr[1]["q_liquid"].sum()  # liquid fell (Bergeron)


def test_inp_seeding_injects_ice_and_glaciates():
    """INP / glaciogenic seeding: spec phase='ice' injects ice embryos (phase=1)
    into a supercooled cloud; they grow by WBF, so the seeded run glaciates far
    more than the (nearly freezing-free, low-B) baseline."""
    base = dict(nt=200, dt=1.0, Nx=16, Nz=16, T0=250.0, RH0=0.999, n_super=4000,
                collisions=False, ice=True, sediment=False, nu_scalar=0.0,
                collect_every=100, B_bigg=1.0)         # low B -> negligible spontaneous freezing
    inp_spec = dict(t_inject=20.0, x_frac=(0.3, 0.7), z_lo=50.0, z_hi=400.0,
                    N_cm3=0.05, r_um=0.5, r_wet_um=3.0, kappa=0.0, n_super=400,
                    phase="ice")
    out_seed = run_flow2d_dynamic(seeding=inp_spec, **base)
    out_base = run_flow2d_dynamic(**base)
    # the injected (tagged) super-droplets are ICE
    seeded_ice = (out_seed["tag"] > 0) & (out_seed["phase"] == 1)
    assert int(seeded_ice.sum()) > 0
    # INP seeding glaciates: more ice than the baseline that barely freezes on its own
    iwp_seed = out_seed["frames"][-1]["q_ice"].sum()
    iwp_base = out_base["frames"][-1]["q_ice"].sum()
    assert iwp_seed > iwp_base


def test_animate_mixed_phase_builds():
    import matplotlib; matplotlib.use("Agg")
    from droplab.flow2d_viz import animate_mixed_phase
    out = run_flow2d_dynamic(nt=40, dt=1.0, Nx=16, Nz=16, T0=250.0, RH0=0.999,
                             n_super=3000, collisions=False, ice=True, B_bigg=1.0e5,
                             sediment=False, collect_every=20)
    fig, anim = animate_mixed_phase(out, dt=1.0)
    assert anim is not None


def test_abifm_prob_zero_without_inp():
    from droplab.ice_microphysics import abifm_prob
    from droplab.condensation import esatw
    from droplab.ice_microphysics import esati
    T = 263.15
    assert abifm_prob(0.0, esatw(T), esati(T), 1.0, -1.35, 22.62) == 0.0


def test_abifm_prob_increases_with_inp_area_and_cold():
    from droplab.ice_microphysics import abifm_prob, esati
    from droplab.condensation import esatw
    c, m = -1.35, 22.62
    # bigger immersed INP area -> higher freezing probability
    Tw = 263.15
    p_small = abifm_prob(1e-12, esatw(Tw), esati(Tw), 1.0, c, m)
    p_big = abifm_prob(1e-9, esatw(Tw), esati(Tw), 1.0, c, m)
    assert 0.0 <= p_small < p_big <= 1.0
    # colder -> larger delta_aw -> higher J -> higher probability (same INP area)
    Tcold = 253.15
    p_cold = abifm_prob(1e-10, esatw(Tcold), esati(Tcold), 1.0, c, m)
    p_warm = abifm_prob(1e-10, esatw(Tw), esati(Tw), 1.0, c, m)
    assert p_cold > p_warm


def test_abifm_species_table_has_verified_pairs():
    from droplab.ice_microphysics import ABIFM_SPECIES
    assert ABIFM_SPECIES["default"] == (-1.35, 22.62)  # natural dust (Alpert & Knopf 2016)
    assert "dust" in ABIFM_SPECIES and "illite" in ABIFM_SPECIES


def test_inp_population_present_at_start():
    # a base INP population (inp>0) exists from t=0, sized by inp_n_cm3 / aero number
    out = run_flow2d_dynamic(nt=10, dt=1.0, Nx=8, Nz=8, n_super=6000, N_modes=(60.,),
                             collisions=False, ice=True, inp_n_cm3=6.0, collect_every=10)
    frac = (out["inp"] > 0).mean()
    assert abs(frac - 6.0 / 60.0) < 0.03          # ~10% of SDs carry INP


def test_abifm_base_inp_glaciates_and_conserves_water():
    out = run_flow2d_dynamic(nt=300, dt=1.0, Nx=8, Nz=8, T0=248.0, RH0=0.999, n_super=4000,
                             collisions=False, ice=True, freezing_mode="abifm",
                             inp_n_cm3=5.0, inp_r_um=0.8, sediment=False, nu_scalar=0.0,
                             collect_every=60)
    fr = out["frames"]
    tw = lambda f: f["qv"].sum() + (f["q_liquid"].sum() + f["q_ice"].sum()) / 1e3
    assert tw(fr[-1]) == pytest.approx(tw(fr[0]), rel=2e-3)        # water conserved
    assert fr[-1]["q_ice"].sum() > fr[0]["q_ice"].sum()           # glaciates via ABIFM
    assert fr[-1]["q_liquid"].sum() < fr[0]["q_liquid"].sum()


def test_no_inp_means_no_abifm_freezing():
    # ABIFM needs INP: with inp_n_cm3=0 no super-droplet can immersion-freeze.
    # homogeneous=False isolates the ABIFM pathway (else deep-cold tops freeze homogeneously).
    out = run_flow2d_dynamic(nt=150, dt=1.0, Nx=8, Nz=8, T0=248.0, RH0=0.999, n_super=3000,
                             collisions=False, ice=True, freezing_mode="abifm",
                             inp_n_cm3=0.0, homogeneous=False, sediment=False, collect_every=150)
    assert int(out["phase"].sum()) == 0


def test_freezing_mode_bigg_still_available():
    out = run_flow2d_dynamic(nt=120, dt=1.5, Nx=16, Nz=16, T0=245.0, RH0=0.98, n_super=4000,
                             collisions=False, ice=True, freezing_mode="bigg", B_bigg=1.0e6,
                             sediment=False, collect_every=120)
    assert int(out["phase"].sum()) > 0


def test_animate_droplets_phase_builds():
    import matplotlib; matplotlib.use("Agg")
    from droplab.flow2d_viz import animate_droplets_phase
    out = run_flow2d_dynamic(nt=40, dt=1.0, Nx=12, Nz=12, T0=248.0, RH0=0.999,
                             n_super=2000, collisions=False, ice=True,
                             inp_n_cm3=8.0, inp_r_um=2.0, collect_every=20)
    fig, anim = animate_droplets_phase(out, dt=1.0)
    assert anim is not None


def test_inp_bearing_liquid_seeding_freezes_via_abifm():
    """The OTHER seeding mode: spec inp_r_um (no phase='ice') injects LIQUID drops
    that carry an immersed INP area, so in a supercooled cloud they freeze via ABIFM.
    With no base INP, only the injected (tagged) drops can glaciate."""
    base = dict(nt=200, dt=1.0, Nx=16, Nz=16, T0=250.0, RH0=0.999, n_super=3000,
                collisions=False, ice=True, freezing_mode="abifm", inp_n_cm3=0.0,
                homogeneous=False,                        # isolate the ABIFM pathway
                sediment=False, nu_scalar=0.0, collect_every=200)
    spec = dict(t_inject=20.0, x_frac=(0.3, 0.7), z_lo=50.0, z_hi=400.0, N_cm3=0.05,
                r_um=0.5, kappa=0.5, n_super=400, inp_r_um=2.0)   # INP-bearing LIQUID
    out = run_flow2d_dynamic(seeding=spec, **base)
    seeded = out["tag"] > 0
    assert seeded.sum() > 0
    assert (out["inp"][seeded] > 0).all()                 # injected drops carry INP
    assert int((out["phase"][seeded] == 1).sum()) > 0     # and freeze via ABIFM
    # the base (untagged) population has no INP -> stays liquid
    assert int((out["phase"][~seeded] == 1).sum()) == 0
