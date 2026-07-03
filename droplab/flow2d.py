"""Prescribed 2D kinematic flow for the cumulus framework.

A steady, non-divergent velocity field derived from a stream function so that
the DISCRETE divergence is exactly zero per grid cell (a requirement for the
MPDATA scalar advection to conserve mass). Geometry: a central updraft with
flanking downdrafts (cumulus-like), with rigid top/bottom lids (w=0) and a
horizontally periodic domain.

    cumulus     : psi = W0*(X/2pi)*sin(2pi(x-X/2)/X)*sin(pi z/Z)
                  -> central updraft, flanking downdrafts
    single_eddy : psi = W0*(X/pi)*sin(pi x/X)*sin(pi z/Z)
                  -> one rotating circulation cell (up the left, down the right)

    u = -dpsi/dz   (on x-faces),    w = +dpsi/dx   (on z-faces)

Velocities live on a staggered C-grid: u on the (Nx+1, Nz) vertical faces, w on
the (Nx, Nz+1) horizontal faces. Cell-centered values (for droplet advection and
plotting) are face averages.
"""
import numpy as np
from scipy.special import erf


class Flow2D:
    def __init__(self, X=2000.0, Z=2000.0, Nx=64, Nz=64, W0=2.0,
                 pattern="cumulus", L_thermal=500.0, z_inv=1500.0):
        self.X, self.Z = float(X), float(Z)
        self.Nx, self.Nz = int(Nx), int(Nz)
        self.W0 = float(W0)
        self.pattern = pattern
        self.L_thermal = float(L_thermal)   # updraft half-width (thermal)
        self.z_inv = float(z_inv)           # flow cap / inversion height (thermal)
        self.dx = self.X / self.Nx
        self.dz = self.Z / self.Nz

        # stream function on the corner grid (Nx+1, Nz+1)
        xc = np.linspace(0.0, self.X, self.Nx + 1)
        zc = np.linspace(0.0, self.Z, self.Nz + 1)
        XX, ZZ = np.meshgrid(xc, zc, indexing="ij")
        self.psi = self._stream(XX, ZZ)

        # face velocities from psi differences (discrete-divergence-free)
        self.u = -(self.psi[:, 1:] - self.psi[:, :-1]) / self.dz   # (Nx+1, Nz)
        self.w = (self.psi[1:, :] - self.psi[:-1, :]) / self.dx    # (Nx, Nz+1)

    def _stream(self, x, z):
        """Analytic stream function for the chosen pattern (vanishes on all walls)."""
        if self.pattern == "single_eddy":
            return (self.W0 * self.X / np.pi
                    * np.sin(np.pi * x / self.X) * np.sin(np.pi * z / self.Z))
        if self.pattern == "thermal":
            # Localized thermal: a NARROW Gaussian updraft fed by BROAD low-level
            # convergence from the moist far-field, confined below z_inv (the flow
            # caps the cloud — kinematic flow ignores buoyancy, so an inversion
            # alone would not stop it). Quiescent above z_inv and in the far field.
            xc, L, zi = self.X / 2.0, self.L_thermal, self.z_inv
            g_int = lambda xx: 0.5 * np.sqrt(np.pi) * L * erf((xx - xc) / L)
            gbar = (g_int(self.X) - g_int(0.0)) / self.X      # domain-mean of g
            Phi = (g_int(x) - g_int(0.0)) - gbar * x           # int_0^x (g - gbar)
            zfac = np.where(z <= zi, np.sin(np.pi * np.clip(z, 0, zi) / zi), 0.0)
            return self.W0 * Phi * zfac
        # default: cumulus (central updraft + flanking downdrafts)
        return (self.W0 * self.X / (2.0 * np.pi)
                * np.sin(2.0 * np.pi * (x - self.X / 2.0) / self.X)
                * np.sin(np.pi * z / self.Z))

    # -- diagnostics ---------------------------------------------------------
    def divergence(self):
        """Per-cell discrete divergence (Nx, Nz); ~machine-zero by construction."""
        dudx = (self.u[1:, :] - self.u[:-1, :]) / self.dx          # (Nx, Nz)
        dwdz = (self.w[:, 1:] - self.w[:, :-1]) / self.dz          # (Nx, Nz)
        return dudx + dwdz

    def cell_velocities(self):
        """Cell-centered (u, w), each (Nx, Nz), as face averages."""
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])               # (Nx, Nz)
        wc = 0.5 * (self.w[:, :-1] + self.w[:, 1:])               # (Nx, Nz)
        return uc, wc

    # -- interpolation to droplet positions ----------------------------------
    def interpolate(self, x, z):
        """Bilinearly sample (u, w) at droplet positions x, z (arrays, metres).

        Closed box: velocities vanish at all four walls, so coordinates are
        clamped to the domain. Returns (u_at, w_at) arrays the same shape as x."""
        uc, wc = self.cell_velocities()
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        # cell-center coordinates
        xi = (x / self.dx) - 0.5            # fractional center index in x
        zj = (z / self.dz) - 0.5            # fractional center index in z

        i0 = np.floor(xi).astype(np.int64)
        j0 = np.floor(zj).astype(np.int64)
        fx = xi - i0
        fz = zj - j0

        # clamped at all walls (closed box)
        i0m = np.clip(i0, 0, self.Nx - 1)
        i1m = np.clip(i0 + 1, 0, self.Nx - 1)
        j0c = np.clip(j0, 0, self.Nz - 1)
        j1c = np.clip(j0 + 1, 0, self.Nz - 1)

        def _bilin(field):
            f00 = field[i0m, j0c]
            f10 = field[i1m, j0c]
            f01 = field[i0m, j1c]
            f11 = field[i1m, j1c]
            return ((1 - fx) * (1 - fz) * f00 + fx * (1 - fz) * f10
                    + (1 - fx) * fz * f01 + fx * fz * f11)

        return _bilin(uc), _bilin(wc)

    def psi_at(self, x, z):
        """Analytic stream function at (x, z). For a steady non-divergent flow
        psi is a material invariant — constant along every droplet trajectory."""
        return self._stream(np.asarray(x, dtype=np.float64),
                            np.asarray(z, dtype=np.float64))

    def advect(self, x, z, dt):
        """Advance droplet positions one step with RK2 (midpoint) using the
        interpolated velocity. Positions are clamped to the closed box."""
        u1, w1 = self.interpolate(x, z)
        xm = np.clip(x + 0.5 * dt * u1, 0.0, self.X)
        zm = np.clip(z + 0.5 * dt * w1, 0.0, self.Z)
        u2, w2 = self.interpolate(xm, zm)
        xn = np.clip(x + dt * u2, 0.0, self.X)
        zn = np.clip(z + dt * w2, 0.0, self.Z)
        return xn, zn
