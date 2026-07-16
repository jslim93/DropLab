"""Smagorinsky subgrid-scale (SGS) turbulence closure for the 2-D dynamics -- the step that
makes DropLab 'LES-flavored'. The unresolved eddies are modelled as an eddy viscosity set by
the RESOLVED strain rate (Smagorinsky 1963):

    nu_t = (Cs * Delta)^2 * |S| ,   Delta = sqrt(dx*dz) ,   |S| = sqrt(2 S_ij S_ij)

applied as a variable-coefficient diffusion div((nu+nu_t) grad .) of vorticity and scalars.
The SGS dissipation rate eps = nu_t * |S|^2 is the natural local turbulence intensity, and
feeding it to the Linear Eddy Model (droplab.lem_driver) makes the LEM self-consistent: the
subgrid supersaturation broadening is then driven by the resolved flow's own strain, exactly
as SAM couples its SGS-TKE to the LEM (tk_LCM), instead of a prescribed eps.

CAVEAT (honest): DropLab is 2-D, and 2-D turbulence has an INVERSE energy cascade, unlike
real 3-D turbulence. So this is 'LES-flavored' for education, not a true LES (which needs 3-D).

OPT-IN: smagorinsky=False -> nu_t is never computed and the dynamics are bit-identical.
"""
import numpy as np


def _ddx(f, dx, periodic_x):
    """Central d/dx on the cell-centred field f (Nx,Nz). Periodic or one-sided at x edges."""
    if periodic_x:
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)
    g = np.empty_like(f)
    g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dx)
    g[0, :] = (f[1, :] - f[0, :]) / dx
    g[-1, :] = (f[-1, :] - f[-2, :]) / dx
    return g


def _ddz(f, dz):
    """Central d/dz on the cell-centred field f (Nx,Nz), one-sided (no-flux) at the z lids."""
    g = np.empty_like(f)
    g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dz)
    g[:, 0] = (f[:, 1] - f[:, 0]) / dz
    g[:, -1] = (f[:, -1] - f[:, -2]) / dz
    return g


def strain_viscosity(uc, wc, dx, dz, Cs, periodic_x):
    """Smagorinsky eddy viscosity nu_t and strain magnitude |S| from the cell-centred
    resolved velocity (uc, wc). Returns (nu_t, Smag), both (Nx,Nz)."""
    dudx = _ddx(uc, dx, periodic_x)
    dwdz = _ddz(wc, dz)
    dudz = _ddz(uc, dz)
    dwdx = _ddx(wc, dx, periodic_x)
    s12 = 0.5 * (dudz + dwdx)
    Smag = np.sqrt(2.0 * (dudx ** 2 + dwdz ** 2) + 4.0 * s12 ** 2)   # |S| = sqrt(2 S_ij S_ij)
    delta = np.sqrt(dx * dz)
    nu_t = (Cs * delta) ** 2 * Smag
    return nu_t, Smag


def div_nu_grad(f, K, dx, dz, periodic_x):
    """Variable-coefficient diffusion div(K grad f) on the cell-centred grid, K_{i+1/2}
    = 0.5(K_i+K_{i+1}) (harmonic-free arithmetic face average). Periodic in x; no-flux at the
    z lids (matching the constant-nu vorticity Laplacian's boundary behaviour)."""
    if periodic_x:
        Kxp = 0.5 * (K + np.roll(K, -1, axis=0)); fxp = np.roll(f, -1, axis=0) - f
        Kxm = 0.5 * (K + np.roll(K, 1, axis=0)); fxm = f - np.roll(f, 1, axis=0)
        dxx = (Kxp * fxp - Kxm * fxm) / dx ** 2
    else:
        dxx = np.zeros_like(f)
        Kxp = 0.5 * (K[:-1, :] + K[1:, :])
        flux = Kxp * (f[1:, :] - f[:-1, :]) / dx ** 2
        dxx[:-1, :] += flux; dxx[1:, :] -= flux                     # no-flux at x walls
    dzz = np.zeros_like(f)
    Kzp = 0.5 * (K[:, :-1] + K[:, 1:])
    fluxz = Kzp * (f[:, 1:] - f[:, :-1]) / dz ** 2
    dzz[:, :-1] += fluxz; dzz[:, 1:] -= fluxz                       # no-flux at z lids
    return dxx + dzz


def dissipation(nu_t, Smag):
    """SGS dissipation rate eps = nu_t |S|^2 [m^2/s^3] -- the local turbulence intensity to
    feed the LEM. Floored to a small positive value so the LEM coefficients stay finite."""
    return np.maximum(nu_t * Smag ** 2, 1.0e-6)
