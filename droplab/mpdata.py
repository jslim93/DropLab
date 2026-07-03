"""2D MPDATA scalar advection (Smolarkiewicz 1984).

Multidimensional Positive-Definite Advection Transport Algorithm: a donor-cell
(upwind) pass followed by one or more antidiffusive corrector passes that cancel
the upwind scheme's implicit diffusion while keeping the field non-negative.

Geometry: a CLOSED box. Velocities (hence Courant numbers) vanish at all four
walls, so flux-form advection conserves the scalar's total mass exactly and no
boundary conditions beyond zero-wall-flux are needed; transverse (cross) terms
use zero-gradient (edge) padding.

Fields are cell-centered (Nx, Nz). Courant numbers are staggered:
  Cx on x-faces  -> (Nx+1, Nz)      Cz on z-faces -> (Nx, Nz+1)
with Cx[0]=Cx[Nx]=0 and Cz[:,0]=Cz[:,Nz]=0 (closed walls).
"""
import numpy as np

_EPS = 1e-15


def _flux_x(psi, Cx):
    """Donor-cell flux on every x-face. Walls (faces 0, Nx) carry no flux."""
    F = np.zeros_like(Cx)
    Cp = np.maximum(Cx[1:-1, :], 0.0)
    Cm = np.minimum(Cx[1:-1, :], 0.0)
    F[1:-1, :] = Cp * psi[:-1, :] + Cm * psi[1:, :]
    return F


def _flux_z(psi, Cz):
    F = np.zeros_like(Cz)
    Cp = np.maximum(Cz[:, 1:-1], 0.0)
    Cm = np.minimum(Cz[:, 1:-1], 0.0)
    F[:, 1:-1] = Cp * psi[:, :-1] + Cm * psi[:, 1:]
    return F


def upwind_step(psi, Cx, Cz):
    """One donor-cell (1st-order, diffusive) flux-form advection step."""
    Fx = _flux_x(psi, Cx)
    Fz = _flux_z(psi, Cz)
    return psi - (Fx[1:, :] - Fx[:-1, :]) - (Fz[:, 1:] - Fz[:, :-1])


def _antidiff_x(psi, Cx, Cz, eps):
    """Antidiffusive Courant number on interior x-faces (Smolarkiewicz 1984),
    including the transverse (cross-derivative) term."""
    Ca = np.zeros_like(Cx)
    psiL, psiR = psi[:-1, :], psi[1:, :]                 # (Nx-1, Nz)
    A = (psiR - psiL) / (psiR + psiL + eps)
    Cxi = Cx[1:-1, :]

    # transverse psi gradient ratio (z-direction), zero-gradient padded
    pz = np.pad(psi, ((0, 0), (1, 1)), mode="edge")      # (Nx, Nz+2)
    Lp1, Rp1 = pz[:-1, 2:], pz[1:, 2:]                   # psi[i-1,j+1], psi[i,j+1]
    Lm1, Rm1 = pz[:-1, :-2], pz[1:, :-2]                 # psi[i-1,j-1], psi[i,j-1]
    B = (Lp1 + Rp1 - Lm1 - Rm1) / (Lp1 + Rp1 + Lm1 + Rm1 + eps)

    # mean transverse Courant around the face
    Czb = 0.25 * (Cz[:-1, :-1] + Cz[:-1, 1:] + Cz[1:, :-1] + Cz[1:, 1:])

    Ca[1:-1, :] = (np.abs(Cxi) - Cxi ** 2) * A - Cxi * Czb * B
    return Ca


def _antidiff_z(psi, Cx, Cz, eps):
    Ca = np.zeros_like(Cz)
    psiB, psiT = psi[:, :-1], psi[:, 1:]                 # (Nx, Nz-1)
    A = (psiT - psiB) / (psiT + psiB + eps)
    Czi = Cz[:, 1:-1]

    px = np.pad(psi, ((1, 1), (0, 0)), mode="edge")      # (Nx+2, Nz)
    Bp, Bm = px[2:, :], px[:-2, :]                       # psi[i+1,*], psi[i-1,*]
    Bcross = (Bp[:, :-1] + Bp[:, 1:] - Bm[:, :-1] - Bm[:, 1:]) / \
             (Bp[:, :-1] + Bp[:, 1:] + Bm[:, :-1] + Bm[:, 1:] + eps)

    Cxb = 0.25 * (Cx[:-1, :-1] + Cx[1:, :-1] + Cx[:-1, 1:] + Cx[1:, 1:])

    Ca[:, 1:-1] = (np.abs(Czi) - Czi ** 2) * A - Czi * Cxb * Bcross
    return Ca


def upwind_step_periodic_x(psi, Cx, Cz):
    """Donor-cell step, PERIODIC in x (Cx is (Nx,Nz): face i between cell i-1 and
    i, wrapping) and no-flux at the z lids (Cz is (Nx,Nz+1))."""
    psi_l = np.roll(psi, 1, axis=0)
    Fx = np.maximum(Cx, 0.0) * psi_l + np.minimum(Cx, 0.0) * psi   # (Nx,Nz)
    div_x = np.roll(Fx, -1, axis=0) - Fx
    Fz = _flux_z(psi, Cz)
    return psi - div_x - (Fz[:, 1:] - Fz[:, :-1])


def _antidiff_x_periodic(psi, Cx, Cz, eps):
    psiL = np.roll(psi, 1, axis=0)
    A = (psi - psiL) / (psi + psiL + eps)
    pz = np.pad(psi, ((0, 0), (1, 1)), mode="edge")
    pzL = np.roll(pz, 1, axis=0)
    Lp1, Rp1 = pzL[:, 2:], pz[:, 2:]
    Lm1, Rm1 = pzL[:, :-2], pz[:, :-2]
    B = (Lp1 + Rp1 - Lm1 - Rm1) / (Lp1 + Rp1 + Lm1 + Rm1 + eps)
    Cz_im1 = np.roll(Cz, 1, axis=0)
    Czb = 0.25 * (Cz_im1[:, :-1] + Cz_im1[:, 1:] + Cz[:, :-1] + Cz[:, 1:])
    return (np.abs(Cx) - Cx ** 2) * A - Cx * Czb * B


def _antidiff_z_periodic(psi, Cx, Cz, eps):
    Ca = np.zeros_like(Cz)
    psiB, psiT = psi[:, :-1], psi[:, 1:]
    A = (psiT - psiB) / (psiT + psiB + eps)
    Czi = Cz[:, 1:-1]
    Bp, Bm = np.roll(psi, -1, axis=0), np.roll(psi, 1, axis=0)
    Bcross = (Bp[:, :-1] + Bp[:, 1:] - Bm[:, :-1] - Bm[:, 1:]) / \
             (Bp[:, :-1] + Bp[:, 1:] + Bm[:, :-1] + Bm[:, 1:] + eps)
    Cx_ip1 = np.roll(Cx, -1, axis=0)
    Cxb = 0.25 * (Cx[:, :-1] + Cx_ip1[:, :-1] + Cx[:, 1:] + Cx_ip1[:, 1:])
    Ca[:, 1:-1] = (np.abs(Czi) - Czi ** 2) * A - Czi * Cxb * Bcross
    return Ca


def mpdata_step_periodic_x(psi, Cx, Cz, n_pass=2, eps=_EPS):
    """MPDATA advection, PERIODIC in x, no-flux z lids. Cx is (Nx,Nz)."""
    psi = upwind_step_periodic_x(psi, Cx, Cz)
    for _ in range(n_pass - 1):
        Cxa = _antidiff_x_periodic(psi, Cx, Cz, eps)
        Cza = _antidiff_z_periodic(psi, Cx, Cz, eps)
        psi = upwind_step_periodic_x(psi, Cxa, Cza)
    return psi


def mpdata_step(psi, Cx, Cz, n_pass=2, eps=_EPS):
    """Advect cell-centered scalar `psi` one step with `n_pass` MPDATA passes
    (1 = plain upwind; 2 = upwind + one antidiffusive corrector, the default)."""
    psi = upwind_step(psi, Cx, Cz)
    for _ in range(n_pass - 1):
        Cxa = _antidiff_x(psi, Cx, Cz, eps)
        Cza = _antidiff_z(psi, Cx, Cz, eps)
        psi = upwind_step(psi, Cxa, Cza)
    return psi
