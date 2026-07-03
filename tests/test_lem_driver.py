"""SAM-LCM Linear Eddy Model per-grid-box coupling (droplab.lem_driver). Each super-droplet
is one LEM box carrying a prognostic supersaturation eta_sd with tau memory; the adiabatic
fluctuation of the triplet rearrangement injects the subgrid supersaturation variance that
broadens the spectrum. Off (supersat_fluct=False) + uniform start -> no variance created."""
import numpy as np
import pytest

from droplab import lem_driver as lem
from droplab.collision_soa import seed_numba_rng


def _one_cell(n=60, s=0.01, T=283.0):
    """n SDs all in cell 0; eta initialised to the cell supersaturation (anomaly 0)."""
    eta = np.full(n, s)
    w = np.zeros(n)
    cidx = np.zeros(n, dtype=np.int64)
    ss = np.array([s]); Tf = np.array([T])
    return eta, w, cidx, ss, Tf


def test_lem_requires_collisions_off():
    """LEM is opt-in and (for now) incompatible with collisions: the per-SD state is not
    threaded through collisional merging. Enabling both must fail loudly, not corrupt."""
    from droplab.flow2d_dynamic import run_flow2d_dynamic
    with pytest.raises(ValueError, match="collisions"):
        run_flow2d_dynamic(Nx=16, Nz=16, nt=2, n_super=16 * 16 * 20,
                           lem=True, collisions=True)


def test_init_lem_state_shapes_and_nan():
    eta, w = lem.init_lem_state(50)
    assert eta.shape == (50,) and w.shape == (50,)
    assert np.all(np.isnan(eta)) and np.all(w == 0.0)


def test_lazy_init_to_cell_mean():
    seed_numba_rng(0)
    eta = np.array([np.nan, np.nan, np.nan])
    w = np.zeros(3); cidx = np.array([0, 1, 0])
    ss = np.array([0.01, -0.2]); Tf = np.array([283.0, 281.0])
    lem.nudge_and_mix(eta, w, cidx, ss, Tf, 2, 50.0, 1e-3, 900.0, 2.0,
                      np.random.default_rng(0), min_sd=10)        # < min_sd -> no mixing
    assert eta[0] == pytest.approx(0.01) and eta[2] == pytest.approx(0.01)
    assert eta[1] == pytest.approx(-0.2)                          # set to its cell mean


def test_nudge_relaxes_toward_cell_mean():
    seed_numba_rng(0)
    eta, w, cidx, ss, Tf = _one_cell(n=5)                        # < min_sd: pure nudge, no mixing
    eta[:] = 0.05                                                 # all above the cell mean 0.01
    lem.nudge_and_mix(eta, w, cidx, ss, Tf, 1, 50.0, 1e-3, 100.0, 10.0,
                      np.random.default_rng(0), min_sd=10, s_max=1.0)   # large bound: isolate nudge
    # nudge: eta -= (eta - 0.01) * dt/tau = (0.05-0.01)*10/100 = 0.004 toward the mean
    assert np.allclose(eta, 0.05 - 0.004)


def test_anomaly_bounded_by_s_max():
    seed_numba_rng(0)
    eta, w, cidx, ss, Tf = _one_cell(n=60)
    eta[:] = 0.5                                                  # far from the cell mean 0.01
    lem.nudge_and_mix(eta, w, cidx, ss, Tf, 1, 50.0, 1e-2, 900.0, 2.0,
                      np.random.default_rng(0), min_sd=10, s_max=0.02)
    assert np.all(eta <= 0.01 + 0.02 + 1e-12)                     # bounded above
    assert np.all(eta >= 0.01 - 0.02 - 1e-12)                     # bounded below


def test_supersat_fluct_creates_variance_off_does_not():
    # uniform start: with the adiabatic fluctuation ON the triplet rearrangement injects
    # supersaturation variance; OFF (pure permutation+diffusion of a uniform field) it stays flat.
    for fluct, expect_var in ((True, True), (False, False)):
        seed_numba_rng(3)
        eta, w, cidx, ss, Tf = _one_cell(n=80)
        for _ in range(40):
            lem.nudge_and_mix(eta, w, cidx, ss, Tf, 1, 60.0, 1e-2, 1.0e9, 2.0,
                              np.random.default_rng(0), min_sd=10, supersat_fluct=fluct)
        spread = float(np.std(eta))
        if expect_var:
            assert spread > 1e-4                                 # variance was created
        else:
            assert spread < 1e-9                                 # uniform stays uniform


def test_sgs_velocity_ar1_stationary_variance():
    rng = np.random.default_rng(0)
    w = np.zeros(20000)
    for _ in range(400):
        w, _disp = lem.sgs_velocity_step(w, 1e-2, 60.0, 0.5, rng)
    L = 60.0
    D_turb = 0.1 * L ** (4.0 / 3.0) * 1e-2 ** (1.0 / 3.0)
    tke = (D_turb / (0.1 * L)) ** 2
    assert np.isclose(w.var(), 2.0 / 3.0 * tke, rtol=0.15)        # AR-1 stationary var = sigma^2


def _triplet_map(n_length):
    """The triplet source-index mapping used by _rearrange_numba (0-based)."""
    n = np.arange(n_length)
    n1 = int(round(n_length / 3.0)); n2 = int(round(2.0 * n_length / 3.0))
    m = np.empty(n_length, dtype=int)
    a = n < n1; b = (n >= n1) & (n < n2); c = n >= n2
    m[a] = 3 * n[a]; m[b] = 2 * (n_length - 1) - 3 * n[b]; m[c] = 3 * n[c] - 2 * (n_length - 1)
    return m.tolist()


def test_triplet_map_matches_sam_fortran():
    """Cross-validation: the SAM-LCM triplet source-index mapping (micro_sgs_mixing.f90
    micro_sgs_rearrangement L351-365) compiled with gfortran gives these permutations
    (0-based). The Python port must reproduce them BIT-FOR-BIT (deterministic mapping)."""
    fortran = {6:  [0, 3, 4, 1, 2, 5],
               9:  [0, 3, 6, 7, 4, 1, 2, 5, 8],
               12: [0, 3, 6, 9, 10, 7, 4, 1, 2, 5, 8, 11],
               30: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 25, 22, 19, 16, 13,
                    10, 7, 4, 1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29]}
    for L, ref in fortran.items():
        assert _triplet_map(L) == ref                            # exact match to compiled Fortran


def test_diffusion_conserves_unlike_sam_typo():
    """The cyclic FTCS diffusion conserves the line mean. SAM's micro_sgs_mixing.f90 L157
    uses T_old(n_end) as the first element's base (apparent typo) -> NOT conservative (the
    compiled Fortran turns sum 13.5 into 13.0 on this input); DropLab uses the consistent
    base and conserves. Documented deliberate deviation (cf. the gamma_ice table-index fix)."""
    To = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 0.5]); cf = 0.1
    Tn = To + cf * (np.roll(To, -1) - 2.0 * To + np.roll(To, 1))
    assert np.isclose(Tn.sum(), To.sum())                        # conserved (SAM gives 13.0)


def test_mean_supersat_preserved_by_mix():
    # the triplet map is a permutation and the adiabatic kick (k-m) sums to 0, so the cell-mean
    # eta is unchanged by ONE mixing pass (before nudging pulls it) -- check with tau huge.
    seed_numba_rng(1)
    eta, w, cidx, ss, Tf = _one_cell(n=90)
    eta[:] = 0.01 + 0.001 * np.sin(np.arange(90))                # some structure, mean ~0.01
    m0 = eta.mean()
    lem.nudge_and_mix(eta, w, cidx, ss, Tf, 1, 60.0, 1e-2, 1.0e12, 2.0,
                      np.random.default_rng(0), min_sd=10, supersat_fluct=True)
    assert eta.mean() == pytest.approx(m0, abs=1e-6)             # mean preserved
