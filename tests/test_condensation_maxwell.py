"""Maxwell r-squared-law benchmark: the growth integrator must reproduce
r^2(t) = r0^2 + 2 G S t exactly when curvature, solute, and the kinetic
correction are switched off.
"""
import numpy as np

from validation.condensation_maxwell import grow_droplet


def test_growth_follows_maxwell_r_squared_law():
    t, r, G = grow_droplet(T=283.15, P=900e2, S=0.01, r0=2.0e-6, nt=100, dt=1.0)
    r2 = r ** 2
    slope_fit, intercept = np.polyfit(t, r2, 1)
    slope_exact = 2.0 * G * 0.01

    # slope matches the analytic 2GS
    assert abs(slope_fit - slope_exact) / slope_exact < 5e-3

    # r^2(t) is a straight line (Maxwell law) to integrator precision
    resid = r2 - (intercept + slope_fit * t)
    lin_err = np.max(np.abs(resid)) / (r2[-1] - r2[0])
    assert lin_err < 1e-6, f"r^2(t) nonlinearity {lin_err:.1e}"
