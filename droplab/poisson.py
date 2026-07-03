"""Fast Poisson solver for the 2D vorticity-streamfunction dynamics.

Solves  lap(psi) = rhs  on a cell-centered grid with psi = 0 on all four walls
(homogeneous Dirichlet at the half-cell-outside boundary). The eigenfunctions of
that discrete Laplacian are the DST-II basis, so a forward DST-II, a divide by
the eigenvalues, and an inverse DST-II solve it exactly in O(N log N) — no
iteration, no nullspace (Dirichlet).
"""
import numpy as np
import scipy.fft as sfft


def laplacian_dirichlet(psi, dx, dz):
    """Discrete Laplacian with psi=0 on the cell-centered walls (ghost = -edge)."""
    lap = np.empty_like(psi)
    # x-direction
    left = np.empty_like(psi)
    right = np.empty_like(psi)
    left[1:, :] = psi[:-1, :]
    left[0, :] = -psi[0, :]          # ghost outside wall: psi[-1] = -psi[0]
    right[:-1, :] = psi[1:, :]
    right[-1, :] = -psi[-1, :]
    lap = (left - 2.0 * psi + right) / dx ** 2
    # z-direction
    down = np.empty_like(psi)
    up = np.empty_like(psi)
    down[:, 1:] = psi[:, :-1]
    down[:, 0] = -psi[:, 0]
    up[:, :-1] = psi[:, 1:]
    up[:, -1] = -psi[:, -1]
    lap += (down - 2.0 * psi + up) / dz ** 2
    return lap


def solve_poisson(rhs, dx, dz):
    """Return psi with lap(psi) = rhs and psi = 0 on the walls."""
    Nx, Nz = rhs.shape
    rhs_hat = sfft.dstn(rhs, type=2, norm="ortho", axes=(0, 1))
    kx = np.arange(1, Nx + 1)
    kz = np.arange(1, Nz + 1)
    lamx = -4.0 / dx ** 2 * np.sin(kx * np.pi / (2 * Nx)) ** 2
    lamz = -4.0 / dz ** 2 * np.sin(kz * np.pi / (2 * Nz)) ** 2
    denom = lamx[:, None] + lamz[None, :]
    psi_hat = rhs_hat / denom
    return sfft.idstn(psi_hat, type=2, norm="ortho", axes=(0, 1))


def laplacian_periodic_x(psi, dx, dz):
    """Discrete Laplacian, PERIODIC in x and psi=0 (Dirichlet) on the z lids."""
    lap = (np.roll(psi, -1, axis=0) - 2.0 * psi + np.roll(psi, 1, axis=0)) / dx ** 2
    down = np.empty_like(psi); up = np.empty_like(psi)
    down[:, 1:] = psi[:, :-1]; down[:, 0] = -psi[:, 0]
    up[:, :-1] = psi[:, 1:]; up[:, -1] = -psi[:, -1]
    lap += (down - 2.0 * psi + up) / dz ** 2
    return lap


def solve_poisson_periodic_x(rhs, dx, dz):
    """Return psi with lap(psi)=rhs, PERIODIC in x and psi=0 on the z lids.
    FFT in x (periodic eigenvalues), DST-II in z (Dirichlet). No nullspace: the
    x-mean mode (lamx=0) pairs only with non-zero lamz."""
    Nx, Nz = rhs.shape
    hat = sfft.dst(rhs, type=2, norm="ortho", axis=1)     # z (Dirichlet)
    hat = np.fft.fft(hat, axis=0)                          # x (periodic)
    kx = np.arange(Nx)
    lamx = -4.0 / dx ** 2 * np.sin(np.pi * kx / Nx) ** 2
    kz = np.arange(1, Nz + 1)
    lamz = -4.0 / dz ** 2 * np.sin(kz * np.pi / (2 * Nz)) ** 2
    hat = hat / (lamx[:, None] + lamz[None, :])
    out = np.fft.ifft(hat, axis=0).real
    return sfft.idst(out, type=2, norm="ortho", axis=1)


# ----------------------------------------------------------------------------------------
# ANELASTIC Poisson  d/dx[(1/rho0) dpsi/dx] + d/dz[(1/rho0) dpsi/dz] = rhs
# ----------------------------------------------------------------------------------------
# psi is the MASS streamfunction (rho0 u = -dpsi/dz, rho0 w = dpsi/dx). Because the base
# density rho0 = rho0(z) depends only on height, the x-transform (FFT periodic / DST-II
# Dirichlet) still diagonalises the x-operator exactly; each x-wavenumber then leaves a 1-D
# variable-coefficient tridiagonal system in z (Thomas algorithm). With constant beta=1/rho0
# the operator reduces to beta * (the Dirichlet/periodic Laplacian above), so this is a
# strict generalisation of the spectral solver.

def _z_tridiag_coeffs(beta, dz):
    """Tridiagonal coefficients of d/dz[beta dpsi/dz] on a cell-centered grid with the
    Dirichlet (psi=0, ghost=-edge) lid convention. beta=1/rho0 is (Nz,). Returns sub, diagz,
    sup each (Nz,); sub[0] and sup[-1] are unused (no out-of-domain unknowns)."""
    Nz = beta.shape[0]
    bf = np.empty(Nz + 1)                       # 1/rho0 at z-faces (Nz+1 of them)
    bf[1:Nz] = 0.5 * (beta[:-1] + beta[1:])
    bf[0] = beta[0]; bf[Nz] = beta[Nz - 1]
    inv = 1.0 / dz ** 2
    sub = bf[:Nz] * inv                         # coeff of psi_{j-1}
    sup = bf[1:Nz + 1] * inv                    # coeff of psi_{j+1}
    diagz = -(bf[:Nz] + bf[1:Nz + 1]) * inv
    diagz[0] -= bf[0] * inv                     # ghost psi_{-1}=-psi_0  -> extra -bf[0]
    diagz[-1] -= bf[Nz] * inv                   # ghost psi_{Nz}=-psi_{Nz-1} -> extra -bf[Nz]
    return sub, diagz, sup


def _batched_thomas(sub, diag, sup, rhs):
    """Solve a tridiagonal system per row. sub, sup are (Nz,) (shared); diag is (Nk, Nz);
    rhs is (Nk, Nz) (real or complex). Vectorised Thomas over the Nk rows."""
    Nk, Nz = rhs.shape
    dp = diag.astype(rhs.dtype).copy()
    rp = rhs.copy()
    for j in range(1, Nz):
        w = sub[j] / dp[:, j - 1]
        dp[:, j] -= w * sup[j - 1]
        rp[:, j] -= w * rp[:, j - 1]
    out = np.empty_like(rp)
    out[:, Nz - 1] = rp[:, Nz - 1] / dp[:, Nz - 1]
    for j in range(Nz - 2, -1, -1):
        out[:, j] = (rp[:, j] - sup[j] * out[:, j + 1]) / dp[:, j]
    return out


def laplacian_anelastic_periodic_x(psi, beta, dx, dz):
    """Anelastic operator d/dx[beta dpsi/dx] + d/dz[beta dpsi/dz], PERIODIC in x, Dirichlet
    z lids (psi=0). beta=1/rho0 is (Nz,). Used to pin the solver in tests."""
    Nx, Nz = psi.shape
    lapx = (np.roll(psi, -1, axis=0) - 2.0 * psi + np.roll(psi, 1, axis=0)) / dx ** 2
    out = beta[None, :] * lapx
    bf = np.empty(Nz + 1)
    bf[1:Nz] = 0.5 * (beta[:-1] + beta[1:]); bf[0] = beta[0]; bf[Nz] = beta[Nz - 1]
    down = np.empty_like(psi); up = np.empty_like(psi)
    down[:, 1:] = psi[:, :-1]; down[:, 0] = -psi[:, 0]
    up[:, :-1] = psi[:, 1:]; up[:, -1] = -psi[:, -1]
    out += (bf[None, 1:Nz + 1] * (up - psi) - bf[None, :Nz] * (psi - down)) / dz ** 2
    return out


def laplacian_anelastic_dirichlet(psi, beta, dx, dz):
    """Anelastic operator with psi=0 on ALL walls (closed box). beta=1/rho0 is (Nz,)."""
    Nx, Nz = psi.shape
    left = np.empty_like(psi); right = np.empty_like(psi)
    left[1:, :] = psi[:-1, :]; left[0, :] = -psi[0, :]
    right[:-1, :] = psi[1:, :]; right[-1, :] = -psi[-1, :]
    out = beta[None, :] * (left - 2.0 * psi + right) / dx ** 2
    bf = np.empty(Nz + 1)
    bf[1:Nz] = 0.5 * (beta[:-1] + beta[1:]); bf[0] = beta[0]; bf[Nz] = beta[Nz - 1]
    down = np.empty_like(psi); up = np.empty_like(psi)
    down[:, 1:] = psi[:, :-1]; down[:, 0] = -psi[:, 0]
    up[:, :-1] = psi[:, 1:]; up[:, -1] = -psi[:, -1]
    out += (bf[None, 1:Nz + 1] * (up - psi) - bf[None, :Nz] * (psi - down)) / dz ** 2
    return out


def solve_poisson_anelastic_periodic_x(rhs, dx, dz, beta):
    """Solve the anelastic Poisson, PERIODIC in x and psi=0 on the z lids. FFT in x
    diagonalises the (x-constant-coeff) operator; a tridiagonal solve in z per wavenumber
    closes it. beta = 1/rho0 is (Nz,)."""
    Nx, Nz = rhs.shape
    sub, diagz, sup = _z_tridiag_coeffs(beta, dz)
    kx = np.arange(Nx)
    lamx = -4.0 / dx ** 2 * np.sin(np.pi * kx / Nx) ** 2          # (Nx,)
    diag = diagz[None, :] + beta[None, :] * lamx[:, None]         # (Nx, Nz)
    rhs_hat = np.fft.fft(rhs, axis=0)
    psi_hat = _batched_thomas(sub, diag, sup, rhs_hat)
    return np.fft.ifft(psi_hat, axis=0).real


def solve_poisson_anelastic(rhs, dx, dz, beta):
    """Solve the anelastic Poisson in a CLOSED box (psi=0 on all walls). DST-II in x
    diagonalises the x-operator; tridiagonal in z per x-mode. beta = 1/rho0 is (Nz,)."""
    Nx, Nz = rhs.shape
    sub, diagz, sup = _z_tridiag_coeffs(beta, dz)
    kx = np.arange(1, Nx + 1)
    lamx = -4.0 / dx ** 2 * np.sin(kx * np.pi / (2 * Nx)) ** 2    # (Nx,)
    diag = diagz[None, :] + beta[None, :] * lamx[:, None]
    rhs_hat = sfft.dst(rhs, type=2, norm="ortho", axis=0)
    psi_hat = _batched_thomas(sub, diag, sup, rhs_hat)
    return sfft.idst(psi_hat, type=2, norm="ortho", axis=0)
