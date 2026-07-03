"""Linear Eddy Model core physics (ported from particle_model/mod_LEM.f90). The triplet map
is a PERMUTATION and diffusion is conservative, so total heat/water is conserved; mixing
reduces the variance of the subgrid (T,q) field -- the broadening mechanism."""
import numpy as np
from droplab import lem


def test_lem_coeffs_krueger():
    eta, D, D_eta, lam, tke = lem.lem_coeffs(L_turb=100.0, eps=1e-4, dz_lem=0.001, rho_air=1.0)
    assert eta == 6.0 * 0.001
    assert D > 0 and D_eta > 0 and lam > 0 and tke > 0
    # stronger dissipation -> larger diffusivity and event rate
    _, D2, _, lam2, _ = lem.lem_coeffs(100.0, 1e-3, 0.001, 1.0)
    assert D2 > D and lam2 > lam


def test_lem_coeffs_match_box_model_fortran():
    """Cross-validation against the particle_model box LEM (particle_model/mod_LEM.f90).
    The reference values below were produced by COMPILING LEM_init's arithmetic standalone
    with gfortran at the box-model defaults (L_turb=100, eps=1e-4, dz=0.001, rho=1, the
    REAL*4 muelq=1.717e-5). The Python port must reproduce them to float32 precision."""
    eta, D, D_eta, lam, tke = lem.lem_coeffs(L_turb=100.0, eps=1e-4, dz_lem=0.001, rho_air=1.0)
    # (name, value) straight from ./lem_init_check (gfortran -O2, single precision)
    ref = dict(eta=6.0000000522e-03, D=2.1544349194e+00, D_eta=1.7169999410e-05,
               lam=2.5302969360e+02, tke=4.6415898949e-02)
    got = dict(eta=eta, D=D, D_eta=D_eta, lam=lam, tke=tke)
    for k in ref:
        assert abs(got[k] - ref[k]) / abs(ref[k]) < 1e-5    # matches box Fortran (float32 floor)


def test_triplet_indices_is_a_permutation():
    tgt, src, disp = lem.triplet_indices(n_start=2, n_length=9, n_dom=20)
    assert sorted(tgt.tolist()) == sorted(src.tolist())     # bijection on the segment
    assert len(set(src.tolist())) == 9                       # each source used once
    assert disp.sum() == 0                                   # displacement n-m sums to 0 (conservative)


def test_triple_map_conserves_heat_and_water():
    rng = np.random.default_rng(0)
    eta, D, D_eta, lam, tke = lem.lem_coeffs(100.0, 1e-2, 0.001, 1.0)  # high eps -> events fire
    n = 300
    T = 263.0 + rng.standard_normal(n) * 0.1
    q = 5e-3 + rng.standard_normal(n) * 1e-4
    T0, q0 = T.sum(), q.sum()
    fired = 0
    for _ in range(200):
        disp = lem.triple_map(T, q, 0.001, 100.0, eta, lam, 0.5, rng, supersat_fluct=True)
        if np.any(disp != 0):
            fired += 1
    assert fired > 0                                         # events actually happened
    assert abs(T.sum() - T0) < 1e-9 * abs(T0)                # heat conserved (disp sums to 0)
    assert abs(q.sum() - q0) < 1e-12 * abs(q0)               # water conserved (permutation)


def test_triple_map_is_a_reordering_without_fluct():
    rng = np.random.default_rng(1)
    eta, D, D_eta, lam, tke = lem.lem_coeffs(100.0, 1e-2, 0.001, 1.0)
    q = np.arange(300.0)
    q_multiset = sorted(q.tolist())
    for _ in range(50):
        T = np.zeros(300)
        lem.triple_map(T, q, 0.001, 100.0, eta, lam, 0.5, rng, supersat_fluct=False)
    assert sorted(q.tolist()) == q_multiset                  # pure permutation: multiset preserved


def test_diffusion_conserves_and_smooths():
    T = np.zeros(100); T[50] = 1.0                           # a spike
    q = T.copy(); s0 = T.sum(); v0 = T.var()
    for _ in range(50):
        lem.diffuse(T, q, D_eta=1e-4, dz_lem=0.001, dt=0.001)
    assert abs(T.sum() - s0) < 1e-9                          # conservative
    assert T.var() < v0                                      # smoothed (variance down)


def test_sgs_velocity_ar1_stationary_variance():
    rng = np.random.default_rng(0)
    eta, D, D_eta, lam, tke = lem.lem_coeffs(100.0, 1e-3, 0.001, 1.0)
    w = np.zeros(20000)
    for _ in range(400):
        w = lem.sgs_velocity(w, D_eta, eta, dt=0.01, rng=rng)
    tke_eta = (D_eta / (eta * 0.1)) ** 2
    assert np.isclose(w.var(), tke_eta, rtol=0.15)           # AR-1 stationary var = sigma_w^2
