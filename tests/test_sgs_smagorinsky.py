"""Smagorinsky SGS closure (droplab.sgs_smagorinsky): eddy viscosity from the resolved
strain, variable-coefficient diffusion, and the SGS dissipation rate that feeds the LEM."""
import numpy as np

from droplab import sgs_smagorinsky as sg


def test_uniform_flow_has_no_viscosity():
    uc = np.full((20, 16), 3.0); wc = np.full((20, 16), -1.0)   # no strain
    nu_t, S = sg.strain_viscosity(uc, wc, 50.0, 50.0, 0.17, periodic_x=True)
    assert np.allclose(nu_t, 0.0) and np.allclose(S, 0.0)


def test_shear_creates_viscosity_scaling_with_cs():
    z = np.linspace(0, 1, 16)
    uc = np.tile(2.0 * z, (20, 1)); wc = np.zeros((20, 16))      # du/dz shear
    nu_t, S = sg.strain_viscosity(uc, wc, 50.0, 50.0, 0.17, periodic_x=True)
    assert np.all(S[:, 1:-1] > 0) and np.all(nu_t[:, 1:-1] > 0)
    nu_t2, _ = sg.strain_viscosity(uc, wc, 50.0, 50.0, 0.34, periodic_x=True)
    assert np.allclose(nu_t2, 4.0 * nu_t)                        # nu_t ~ Cs^2


def test_div_nu_grad_constant_matches_laplacian():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((24, 18)); K = np.full_like(f, 2.5); dx, dz = 40.0, 30.0
    out = sg.div_nu_grad(f, K, dx, dz, periodic_x=True)
    # constant K: div(K grad f) = K * laplacian(f) (periodic x, no-flux z)
    lap = (np.roll(f, -1, 0) - 2 * f + np.roll(f, 1, 0)) / dx ** 2
    lapz = np.zeros_like(f)
    lapz[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / dz ** 2
    lapz[:, 0] = (f[:, 1] - f[:, 0]) / dz ** 2                   # no-flux lids
    lapz[:, -1] = (f[:, -2] - f[:, -1]) / dz ** 2
    assert np.allclose(out, 2.5 * (lap + lapz))


def test_div_nu_grad_conserves():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((20, 16)); K = np.abs(rng.standard_normal((20, 16))) + 0.1
    out = sg.div_nu_grad(f, K, 50.0, 50.0, periodic_x=True)
    assert abs(out.sum()) < 1e-9                                 # flux-form -> conservative


def test_dissipation_positive_and_grows_with_strain():
    nu_t = np.array([[1e-3, 2e-3]]); S = np.array([[0.1, 0.3]])
    eps = sg.dissipation(nu_t, S)
    assert np.all(eps > 0) and eps[0, 1] > eps[0, 0]
