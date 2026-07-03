"""MPDATA scalar-advection gates.

In the closed cumulus box (velocities vanish at all walls) MPDATA must:
  1. conserve scalar mass exactly (flux-form, no flux through walls),
  2. stay positive-definite (no spurious negatives),
  3. diffuse markedly less than plain donor-cell upwind (the whole point).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")

from droplab.flow2d import Flow2D
from droplab.mpdata import mpdata_step, upwind_step, mpdata_step_periodic_x


def _gaussian_blob(f, x0, z0, sigma=150.0, amp=1.0, base=300.0):
    xc = (np.arange(f.Nx) + 0.5) * f.dx
    zc = (np.arange(f.Nz) + 0.5) * f.dz
    XX, ZZ = np.meshgrid(xc, zc, indexing="ij")
    return base + amp * np.exp(-((XX - x0) ** 2 + (ZZ - z0) ** 2) / (2 * sigma ** 2))


def _courant(f, dt):
    return f.u * dt / f.dx, f.w * dt / f.dz


def test_mass_conserved_and_positive():
    f = Flow2D(X=2000.0, Z=2000.0, Nx=64, Nz=64, W0=2.0)
    Cx, Cz = _courant(f, dt=5.0)
    psi = _gaussian_blob(f, 1000.0, 700.0)
    m0 = psi.sum()
    for _ in range(200):
        psi = mpdata_step(psi, Cx, Cz)
    assert np.isclose(psi.sum(), m0, rtol=1e-9), f"mass drift: {psi.sum()} vs {m0}"
    assert psi.min() > 0.0, f"positivity violated: min={psi.min()}"


def test_periodic_face_velocities_divergence_free():
    """Periodic-x face velocities from psi are divergence-free per cell (needed
    for the periodic MPDATA to conserve mass)."""
    from droplab.flow2d_dynamic import _faces_from_psi_periodic
    Nx, Nz, dx, dz = 48, 32, 50.0, 40.0
    psi = np.random.default_rng(0).standard_normal((Nx, Nz))
    u, w = _faces_from_psi_periodic(psi, dx, dz)
    div = (np.roll(u, -1, axis=0) - u) / dx + (w[:, 1:] - w[:, :-1]) / dz
    assert np.max(np.abs(div)) < 1e-10


def test_periodic_mpdata_conserves_positive_wraps():
    """Periodic-x MPDATA: mass conserved, positive, and a blob advected by a
    uniform x-flow translates (and would wrap) without hitting a wall."""
    Nx, Nz = 64, 32
    Cx = np.full((Nx, Nz), 0.3)              # uniform rightward flow
    Cz = np.zeros((Nx, Nz + 1))
    xc = np.arange(Nx)
    psi = (1.0 + 0.8 * np.exp(-((xc - 10.0) ** 2) / (2 * 4.0 ** 2)))[:, None] \
        * np.ones((Nx, Nz))
    m0 = psi.sum()
    for _ in range(50):
        psi = mpdata_step_periodic_x(psi, Cx, Cz)
    assert np.isclose(psi.sum(), m0, rtol=1e-9), "mass not conserved (periodic)"
    assert psi.min() > 0, "positivity violated"
    peak = int(np.argmax(psi[:, 0]))         # started at 10, moved ~50*0.3=15 -> ~25
    assert 20 <= peak <= 30, f"blob did not advect correctly (peak at {peak})"


def test_mpdata_less_diffusive_than_upwind():
    """After advection, MPDATA keeps a sharper peak than upwind (less smearing)."""
    f = Flow2D(X=2000.0, Z=2000.0, Nx=64, Nz=64, W0=2.0)
    Cx, Cz = _courant(f, dt=5.0)
    p_mp = _gaussian_blob(f, 1000.0, 700.0, base=0.0)
    p_up = p_mp.copy()
    for _ in range(150):
        p_mp = mpdata_step(p_mp, Cx, Cz)
        p_up = upwind_step(p_up, Cx, Cz)
    # both conserve mass; MPDATA should retain a higher peak (anti-diffusion)
    assert p_mp.max() > p_up.max() * 1.1, \
        f"MPDATA not less diffusive: peak {p_mp.max():.3f} vs upwind {p_up.max():.3f}"
    assert np.isclose(p_mp.sum(), p_up.sum(), rtol=1e-6)
