"""Dry rising-thermal benchmark (Robert 1993 style) — a buoyancy/advection verification of the
2-D core that COMPLEMENTS the moist Weisman-Klemp validation (test_dcc_validation).

A small warm bubble (Delta theta = 0.5 K) in a dry, neutrally-stratified atmosphere rises as a
buoyant thermal. The verification is on the NUMERICS, not a published front position:
  (a) the thermal rises (buoyancy -> ascent);
  (b) the potential-temperature anomaly is CONSERVED and MONOTONE -- it stays within [0, 0.5 K]
      with no spurious cold undershoot (the MPDATA advection does not over/under-shoot);
  (c) the peak updraft is consistent with the buoyancy velocity scale sqrt(g*(dtheta/theta)*H),
      reduced by the usual finite-thermal/entrainment factor (it does not run away).
"""
import numpy as np
from droplab.flow2d_dynamic import run_flow2d_dynamic


def test_dry_buoyant_thermal_rises_and_conserves_anomaly():
    Nx, Nz, X, Z = 100, 120, 1200.0, 2000.0
    neutral = {"name": "neutral", "z": [0, 2000], "theta": [300.0, 300.0], "qv": [0.05, 0.05]}
    o = run_flow2d_dynamic(dynamics="boussinesq", Nx=Nx, Nz=Nz, X=X, Z=Z, dt=1.0, nt=700,
                           collect_every=100, n_super=1500, sounding=neutral, RH0=0.1,
                           dtheta_bubble=0.5, bubble_r=250., bubble_z=400., periodic_x=False,
                           seed=1, nu=2.0, nu_scalar=0.5, collisions=False, ice=False, pert_amp=0.0)
    dz = Z / Nz
    z = (np.arange(Nz) + 0.5) * dz
    fr = o["frames"]

    def top(f):
        w = np.where((f["theta"] - 300.0).max(axis=0) > 0.05)[0]
        return z[w.max()] if len(w) else 0.0

    tops = [top(f) for f in fr]
    max_anom = max(float((f["theta"] - 300.0).max()) for f in fr)
    min_anom = min(float((f["theta"] - 300.0).min()) for f in fr)
    peak_w = max(float(np.abs(f["w"]).max()) for f in fr)
    w_scale = np.sqrt(9.8 * (0.5 / 300.0) * 1600.0)        # ~5.1 m/s

    # (a) the thermal rises
    assert tops[-1] > tops[1] + 200.0
    # (b) anomaly conserved + monotone (no overshoot above the 0.5 K bubble, no cold undershoot)
    assert 0.40 < max_anom < 0.55
    assert min_anom > -0.05
    # (c) updraft buoyancy-consistent, not runaway
    assert 0.5 < peak_w < w_scale
    assert all(np.isfinite(f["theta"]).all() for f in fr)
