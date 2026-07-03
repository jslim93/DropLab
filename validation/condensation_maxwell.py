"""Maxwellian r-squared-law benchmark for diffusional droplet growth.

For a droplet growing by vapour diffusion at fixed supersaturation S, with the
curvature (Kelvin) and solute (Raoult) terms negligible and the kinetic
correction switched off (r0 = 0), the growth equation reduces to the exact
Maxwell law

    d(r^2)/dt = 2 G S   ->   r^2(t) = r0^2 + 2 G S t,

where G is the diffusional growth coefficient. This drives DropLab's production
Newton-Raphson growth integrator (``radius_liquid_euler``) and checks it
reproduces the analytic straight line in r^2 vs t.

    python -m validation.condensation_maxwell
"""
import numpy as np

from droplab.parameters import rho_liq, rv, l_v
from droplab.condensation import esatw, radius_liquid_euler


def growth_coeff_G(T, P):
    """Diffusional growth coefficient G [m^2 s^-1], identical to drop_condensation."""
    e_s = esatw(T)
    thermal_conductivity = 7.94048e-05 * T + 0.00227011
    diff_coeff = 0.211e-4 * (T / 273.15) ** 1.94 * (101325.0 / P)
    return 1.0 / (rho_liq * rv * T / (e_s * diff_coeff)
                  + (l_v / (rv * T) - 1.0) * rho_liq * l_v / (thermal_conductivity * T))


def grow_droplet(T=283.15, P=900e2, S=0.01, r0=2.0e-6, nt=100, dt=1.0):
    """Grow one droplet at fixed S with the production integrator. Returns t, r, G."""
    G = growth_coeff_G(T, P)
    r = r0
    ts, rs = [0.0], [r0]
    for n in range(nt):
        # r0(kinetic)=0, afactor=0 (no Kelvin), bfactor=0 (no solute), no ventilation/radiation
        r = radius_liquid_euler(r, dt, 0.0, G, S, 1.0, 0.0, 0.0, 1.0e-9, 0.0, 0.0)
        ts.append((n + 1) * dt); rs.append(r)
    return np.array(ts), np.array(rs), G


def main():
    T, P, S = 283.15, 900e2, 0.01
    t, r, G = grow_droplet(T, P, S)
    r2 = r ** 2
    slope_fit, intercept = np.polyfit(t, r2, 1)
    slope_exact = 2.0 * G * S
    resid = r2 - (intercept + slope_fit * t)
    lin_err = np.max(np.abs(resid)) / (r2[-1] - r2[0])
    print(f"G = {G:.4e} m^2/s,  S = {S}")
    print(f"r: {r[0]*1e6:.2f} -> {r[-1]*1e6:.2f} um over {t[-1]:.0f} s")
    print(f"slope  fitted = {slope_fit:.5e}  exact 2GS = {slope_exact:.5e}  "
          f"(rel {abs(slope_fit-slope_exact)/slope_exact:.3%})")
    print(f"max nonlinearity of r^2(t): {lin_err:.2e} of total rise")


if __name__ == "__main__":
    main()
