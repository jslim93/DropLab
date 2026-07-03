"""Anelastic dynamical core: the variable-coefficient Poisson solver
d/dx[(1/rho0) dpsi/dx] + d/dz[(1/rho0) dpsi/dz] = rhs, and the dynamics= switch.

The solver is pinned with a manufactured solution (assemble rhs from a known psi via the
matching discrete operator, recover psi). With constant rho0 it must reduce to the spectral
Boussinesq solver. The Boussinesq path stays bit-identical (enforced by test_flow2d_golden)."""
import numpy as np
from droplab.poisson import (laplacian_anelastic_periodic_x, laplacian_anelastic_dirichlet,
                             solve_poisson_anelastic_periodic_x, solve_poisson_anelastic,
                             solve_poisson_periodic_x, solve_poisson)
from droplab.flow2d_dynamic import run_flow2d_dynamic


def _beta(Nz, dz):
    z = (np.arange(Nz) + 0.5) * dz
    rho0 = 1.2 * np.exp(-z / 8000.0)          # density falls with height
    return 1.0 / rho0


def test_anelastic_poisson_periodic_round_trip():
    Nx, Nz, dx, dz = 48, 40, 50.0, 60.0
    beta = _beta(Nz, dz)
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((Nx, Nz))
    rhs = laplacian_anelastic_periodic_x(psi, beta, dx, dz)
    rec = solve_poisson_anelastic_periodic_x(rhs, dx, dz, beta)
    assert np.max(np.abs(rec - psi)) < 1e-10


def test_anelastic_poisson_dirichlet_round_trip():
    Nx, Nz, dx, dz = 48, 40, 50.0, 60.0
    beta = _beta(Nz, dz)
    rng = np.random.default_rng(1)
    psi = rng.standard_normal((Nx, Nz))
    rhs = laplacian_anelastic_dirichlet(psi, beta, dx, dz)
    rec = solve_poisson_anelastic(rhs, dx, dz, beta)
    assert np.max(np.abs(rec - psi)) < 1e-10


def test_constant_density_reduces_to_spectral_solver():
    # constant beta -> the anelastic operator is exactly beta * Laplacian, so the solver must
    # match the spectral Boussinesq one (up to tridiagonal-vs-FFT round-off, not bit-identity).
    Nx, Nz, dx, dz = 48, 40, 50.0, 60.0
    b1 = np.ones(Nz)
    rng = np.random.default_rng(2)
    r = rng.standard_normal((Nx, Nz))
    assert np.max(np.abs(
        solve_poisson_anelastic_periodic_x(r, dx, dz, b1) - solve_poisson_periodic_x(r, dx, dz))) < 1e-8
    assert np.max(np.abs(
        solve_poisson_anelastic(r, dx, dz, b1) - solve_poisson(r, dx, dz))) < 1e-8


def test_anelastic_mode_runs_finite():
    """The anelastic dynamics path integrates a bubble without blowing up (finite fields,
    physical velocities). Convective vigour/tuning is a separate concern; this just guards
    the core's numerical health."""
    o = run_flow2d_dynamic(dynamics="anelastic", Nx=32, Nz=40, X=3200, Z=4000, dt=2.0,
                           nt=120, collect_every=120, n_super=6000, dtheta_bubble=2.0,
                           bubble_r=400., bubble_z=600., periodic_x=True, seed=1)
    f = o["frames"][-1]
    assert np.isfinite(f["qc"]).all() and np.isfinite(f["w"]).all()
    assert np.abs(f["w"]).max() < 50.0            # no runaway


def test_boussinesq_default_unchanged_by_anelastic_addition():
    """Two boussinesq runs are reproducible; the anelastic switch defaults off and does not
    perturb the existing path (bit-identity against the frozen reference: test_flow2d_golden)."""
    base = dict(Nx=24, Nz=32, nt=40, n_super=2000, collect_every=40, periodic_x=True, seed=1)
    a = run_flow2d_dynamic(**base)
    b = run_flow2d_dynamic(dynamics="boussinesq", **base)
    assert np.array_equal(a["frames"][-1]["qc"], b["frames"][-1]["qc"])


def test_anelastic_dry_instability_overturns():
    """An absolutely-unstable dry layer (theta DECREASING with height) MUST overturn. This
    pins the fix for the scalar-transport bug: anelastic uses rho0-weighted advection
    (conserves rho0*theta); plain flux-form would spuriously cool rising air (Ds/Dt=-s div V)
    and the instability would decay instead of grow."""
    snd = {"name": "unstable", "z": [0, 1000, 2000, 3000, 4000],
           "theta": [305., 302., 299., 296., 293.], "qv": [1., 1., 1., 1., 1.]}
    base = dict(Nx=32, Nz=40, X=3200, Z=4000, dt=2.0, nt=120, collect_every=20, n_super=2000,
                dtheta_bubble=0.5, bubble_r=400., bubble_z=500., periodic_x=True, seed=1,
                RH0=0.3, b_max=1.0, omega_max=0.5, sounding=snd)
    o = run_flow2d_dynamic(dynamics="anelastic", **base)
    w = [float(np.abs(f["w"]).max()) for f in o["frames"]]
    assert w[-1] > 3.0 * w[0]                 # vorticity/velocity GROWS (overturning)
    assert w[-1] > 0.5


def test_anelastic_deep_convection_is_a_localized_tower():
    """The payoff: a capped-CAPE sounding (CUMULONIMBUS) + a strong localized trigger builds an
    ISOLATED deep convective tower -- it reaches the upper troposphere yet stays narrow at its
    base (clear, subsiding air around it), i.e. a cumulonimbus, NOT a domain-filling stratiform
    layer. The capping inversion + dry free troposphere are what localize it. Impossible in the
    shallow-capped Boussinesq core."""
    from droplab.soundings import CUMULONIMBUS
    Nx, Nz, Z = 80, 72, 11500
    cfg = dict(Nx=Nx, Nz=Nz, X=16000, Z=Z, dt=2.0, nt=700, collect_every=100, n_super=20000,
               dtheta_bubble=5.0, bubble_r=1200., bubble_z=600., periodic_x=True, seed=3,
               RH0=0.5, b_max=0.6, omega_max=0.15, sounding=CUMULONIMBUS,
               ice=True, homogeneous=True, inp_n_cm3=0.5)
    o = run_flow2d_dynamic(dynamics="anelastic", **cfg)
    f = o["frames"][-1]
    z = (np.arange(Nz) + 0.5) * (Z / Nz)
    m = f["qc"].max(axis=0) > 0.05
    cloud_top = z[np.where(m)[0].max()] if m.any() else 0.0
    low_frac = (f["qc"][:, z < 4000].max(axis=1) > 0.1).sum() / Nx     # cloud-base width fraction
    peak_w = max(float(np.abs(fr["w"]).max()) for fr in o["frames"])   # peak updraft over life cycle
    assert cloud_top > 6000.0                              # genuinely DEEP (tropospheric tower)
    assert low_frac < 0.4                                  # LOCALIZED base -> tower, not a layer
    assert peak_w > 8.0                                    # vigorous convective updraft (cumulonimbus)
    assert np.isfinite(f["qc"]).all()
