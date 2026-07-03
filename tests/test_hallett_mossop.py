"""Cold-cloud cycle -- Hallett-Mossop rime splintering (secondary ice production).

During riming at -3..-8 C, ice throws off ~350 splinters per mg of rime. In our fixed-size
super-droplet array the splinter NUMBER is merged onto an existing ice super-droplet (exactly
like the breakup scheme's _merge_fragments_into_nearest) rather than spawning new SDs, so the
population cannot grow without bound. The host mass is left unchanged (splinters borrow mass),
so the direct operation conserves water; the extra crystals then drive faster vapour
deposition -- the physical ice-multiplication feedback (more crystals -> more ice).
"""
import numpy as np
import droplab.flow2d_dynamic as fd
from droplab.ice_microphysics import (_hallett_mossop, hm_factor, M_SPLINTER, HM_PER_KG,
                                       T_HM_LO, T_HM_PEAK, T_HM_HI)


def test_hm_factor_is_triangular_in_the_minus3_to_minus8_window():
    assert hm_factor(T_HM_PEAK) == 1.0                 # -5 C: peak
    assert hm_factor(T_HM_LO) == 0.0                   # -8 C edge: off
    assert hm_factor(T_HM_HI) == 0.0                   # -3 C edge: off
    assert hm_factor(275.0) == 0.0                     # too warm
    assert hm_factor(250.0) == 0.0                     # too cold
    # ramps linearly toward the peak from both sides
    assert 0.0 < hm_factor(266.65) < 1.0               # between -8 and -5
    assert 0.0 < hm_factor(269.15) < 1.0               # between -5 and -3


def test_splintering_conserves_mass_fixes_count_and_multiplies_number():
    # cell 0: two ice SDs + one liquid SD; cell 0 rimed at the peak temperature
    M = np.array([1.0e-6, 2.0e-6, 5.0e-7]); A = np.array([1.0e3, 1.0e3, 1.0e3])
    phase = np.array([1, 1, 0], dtype=np.int8); cidx = np.array([0, 0, 0])
    rimed = np.array([1.0e-7]); T_c = np.array([T_HM_PEAK]); out = np.zeros(1)
    M0, A0, n0 = M.copy(), A.copy(), len(M)

    _hallett_mossop(M, A, phase, cidx, rimed, T_c, out)

    assert len(M) == n0                                # super-droplet count is FIXED
    assert np.array_equal(M, M0)                       # host mass unchanged -> water conserved
    assert A.sum() > A0.sum()                          # ice crystal NUMBER multiplied
    assert phase[2] == 0 and A[2] == A0[2]             # the liquid SD is untouched
    # exactly 350 splinters per milligram of rime at the peak (1e-7 kg = 1e-4 mg -> 35)
    assert np.isclose(out[0], HM_PER_KG * 1.0e-7)


def test_splinter_cap_keeps_per_crystal_mass_physical():
    # an enormous rime mass must NOT dilute the host below one splinter per crystal
    M = np.array([1.0e-6]); A = np.array([1.0e3]); phase = np.array([1], dtype=np.int8)
    cidx = np.array([0]); rimed = np.array([1.0]); T_c = np.array([T_HM_PEAK]); out = np.zeros(1)
    _hallett_mossop(M, A, phase, cidx, rimed, T_c, out)
    assert (M[0] / A[0]) >= M_SPLINTER * (1.0 - 1e-9)


def _shallow_mixed_phase(**over):
    snd = {"name": "shallow mixed-phase H-M bed",
           "z": [0, 800, 1600, 2400, 3500, 5000],
           "theta": [281., 284., 286., 288., 290., 294.],
           "qv": [9., 7., 5.5, 4., 2.5, 1.2]}
    cfg = dict(Nx=40, Nz=48, X=4000, Z=5000, dt=2.0, nt=400, collect_every=400,
               n_super=14000, collisions=True, ice=True, sediment=True,
               freezing_mode="bigg", a_bigg=0.66, B_bigg=1.0e4, inp_n_cm3=20.0,
               homogeneous=False, dtheta_bubble=3.0, bubble_r=700., bubble_z=600.,
               eps=0.02, periodic_x=True, seed=3, sounding=snd)
    cfg.update(over)
    return cfg


def test_hallett_mossop_multiplies_ice_and_keeps_sd_count_fixed():
    """In a riming shallow mixed-phase deck, secondary ice fires; the extra crystals enhance
    deposition so ice grows faster with H-M on, and the super-droplet count is unchanged."""
    orig = fd._hallett_mossop
    tally = {"spl": 0.0, "steps": 0}

    def spy(M, A, phase, cidx, rimed, T_c, out):
        orig(M, A, phase, cidx, rimed, T_c, out)
        s = float(out.sum()); tally["spl"] += s; tally["steps"] += (s > 0.0)

    fd._hallett_mossop = spy
    try:
        on = fd.run_flow2d_dynamic(**_shallow_mixed_phase(hallett_mossop=True))
    finally:
        fd._hallett_mossop = orig
    off = fd.run_flow2d_dynamic(**_shallow_mixed_phase(hallett_mossop=False))

    fon, foff = on["frames"][-1], off["frames"][-1]
    # secondary ice demonstrably fired
    assert tally["steps"] > 0 and tally["spl"] > 0.0
    # ice multiplication: more crystals -> more deposition -> more ice mass than without H-M
    assert fon["q_ice"].sum() > foff["q_ice"].sum()
    # no super-droplet proliferation: H-M adds NUMBER to existing SDs, never new SDs, so the
    # population only ever shrinks (pruning) from the initial allocation -- it never grows.
    # (The unit test proves the operation leaves the array length untouched; here the two
    # runs differ by a handful of SDs only through divergent downstream pruning.)
    n_init = _shallow_mixed_phase()["n_super"]
    assert len(on["A"]) <= n_init and len(off["A"]) <= n_init
    assert abs(len(on["A"]) - len(off["A"])) < 0.01 * n_init
    assert np.isfinite(fon["q_ice"]).all()
