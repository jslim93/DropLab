"""Analytic solution of the stochastic collection equation (Smoluchowski) for
the Golovin (1963) additive kernel K(x, y) = b (x + y), with an exponential
initial mass distribution n(x, 0) = (N0 / x0) exp(-x / x0).

This closed-form solution is the standard ground-truth benchmark for
collision-coalescence schemes, because the general (e.g. gravitational) kernel
has no analytic solution. The same test validates the super-droplet method that
DropLab implements (Shima et al. 2009, Sec. 5).

Conventions
-----------
x   : single-droplet mass [kg]
N0  : initial total number concentration [m^-3]
x0  : initial mean droplet mass [kg]            (M0 = N0 * x0 = mass conc, conserved)
b   : additive-kernel constant [m^3 kg^-1 s^-1] (1.5 = 1500 cm^3 g^-1 s^-1, standard)
tau : dimensionless time, tau = b * M0 * t

References
----------
Golovin, A. M. (1963), Izv. Akad. Nauk SSSR, Ser. Geofiz. 5, 482.
Scott, W. T. (1968), J. Atmos. Sci. 25, 54.
Berry, E. X. & Reinhardt, R. L. (1974), J. Atmos. Sci. 31, 1814.
Shima, S. et al. (2009), Q. J. R. Meteorol. Soc. 135, 1307.
"""
import numpy as np
from scipy.special import ive  # exponentially scaled modified Bessel: ive(v,z)=Iv(z)exp(-|z|)

RHO_W = 1000.0      # density of liquid water [kg m^-3]
B_GOLOVIN = 1.5     # additive-kernel constant [m^3 kg^-1 s^-1] = 1500 cm^3 g^-1 s^-1


def tau(t, N0, x0, b=B_GOLOVIN):
    """Dimensionless time tau = b * M0 * t, with M0 = N0 * x0 the conserved mass conc."""
    return b * N0 * x0 * t


def number_conc(t, N0, x0, b=B_GOLOVIN):
    """Exact total number concentration N(t) = N0 exp(-tau)  [m^-3].

    Derivation: for K = b(x+y), dN/dt = -b M0 N (M0 conserved) -> N = N0 e^{-tau}.
    """
    return N0 * np.exp(-tau(t, N0, x0, b))


def n_of_mass(x, t, N0, x0, b=B_GOLOVIN):
    """Number density per unit droplet mass n(x, t)  [m^-3 kg^-1].

    Uses the identity exp(-xi(1+T)) I1(2 xi sqrt(T)) = ive(1, 2 xi sqrt(T))
    exp(-xi (1 - sqrt(T))^2), which is overflow-safe for large droplets / late T.
    Reduces to the exponential initial condition as T -> 0.
    """
    x = np.asarray(x, dtype=float)
    T = 1.0 - np.exp(-tau(t, N0, x0, b))
    xi = x / x0
    if T <= 0.0:
        return (N0 / x0) * np.exp(-xi)
    sqrtT = np.sqrt(T)
    z = 2.0 * xi * sqrtT
    return (N0 / x0) * (1.0 - T) / (xi * sqrtT) * ive(1, z) * np.exp(-xi * (1.0 - sqrtT) ** 2)


def radius_to_mass(r, rho=RHO_W):
    return 4.0 / 3.0 * np.pi * rho * np.asarray(r, dtype=float) ** 3


def mass_to_radius(x, rho=RHO_W):
    return (np.asarray(x, dtype=float) / (4.0 / 3.0 * np.pi * rho)) ** (1.0 / 3.0)


def g_lnr(r, t, N0, x0, b=B_GOLOVIN, rho=RHO_W):
    """Mass density per unit ln(r):  g(ln r) = 3 x^2 n(x)  [kg m^-3 per unit ln r].

    This is the conventional cloud-microphysics spectral plot (Berry & Reinhardt).
    """
    x = radius_to_mass(r, rho)
    return 3.0 * x ** 2 * n_of_mass(x, t, N0, x0, b)
