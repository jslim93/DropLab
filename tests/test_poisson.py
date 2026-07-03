"""Poisson solver gate: solving lap(psi)=rhs then re-applying the discrete
Laplacian must return rhs (to round-off), with psi=0 on the cell-centered walls."""
import numpy as np

from droplab.poisson import (solve_poisson, laplacian_dirichlet,
                           solve_poisson_periodic_x, laplacian_periodic_x)


def test_poisson_inverts_laplacian():
    Nx, Nz = 48, 32
    dx, dz = 25.0, 30.0
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal((Nx, Nz))
    psi = solve_poisson(rhs, dx, dz)
    lap = laplacian_dirichlet(psi, dx, dz)
    assert np.max(np.abs(lap - rhs)) / np.max(np.abs(rhs)) < 1e-9


def test_periodic_x_poisson_inverts_laplacian():
    """Periodic-x / Dirichlet-z Poisson: solve then re-apply the matching Laplacian."""
    Nx, Nz = 64, 40
    dx, dz = 30.0, 25.0
    rng = np.random.default_rng(1)
    rhs = rng.standard_normal((Nx, Nz))
    psi = solve_poisson_periodic_x(rhs, dx, dz)
    lap = laplacian_periodic_x(psi, dx, dz)
    assert np.max(np.abs(lap - rhs)) / np.max(np.abs(rhs)) < 1e-9


def test_periodic_x_recovers_wrapping_mode():
    """A mode periodic in x (cos(2pi x/X)) and Dirichlet in z is recovered."""
    Nx, Nz = 48, 48
    dx = dz = 40.0
    i = np.arange(Nx); j = (np.arange(Nz) + 0.5)
    psi_exact = np.outer(np.cos(2 * np.pi * i / Nx), np.sin(np.pi * j / Nz))
    rhs = laplacian_periodic_x(psi_exact, dx, dz)
    psi = solve_poisson_periodic_x(rhs, dx, dz)
    assert np.max(np.abs(psi - psi_exact)) < 1e-9


def test_recovers_eigenmode():
    Nx, Nz = 40, 40
    dx = dz = 50.0
    i = (np.arange(Nx) + 0.5)
    j = (np.arange(Nz) + 0.5)
    psi_exact = np.outer(np.sin(np.pi * i / Nx), np.sin(np.pi * j / Nz))
    rhs = laplacian_dirichlet(psi_exact, dx, dz)
    psi = solve_poisson(rhs, dx, dz)
    assert np.max(np.abs(psi - psi_exact)) < 1e-9
