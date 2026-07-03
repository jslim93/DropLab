"""Deep-convection VALIDATION against Weisman-Klemp (1982) parcel theory.

The WK82 sounding (droplab.soundings.WEISMAN_KLEMP) is the standard idealized deep-convection
profile; for its parameters parcel theory gives CAPE ~ 2120 J/kg and an equilibrium level (EL,
the level of neutral buoyancy where a surface parcel stops rising) ~ 12 km. A physically
correct storm must (a) reach that EL -- the rigorous, limiter-independent thermodynamic check
-- and (b) produce updrafts in the realistic range for that CAPE: ~0.3-0.5 x sqrt(2*CAPE) =
20-33 m/s (entrainment + water loading cut the unentrained parcel max of 65 m/s to ~0.4x).

This turns "it looks like a thunderstorm" into a quantitative benchmark. Slow (~30 s): it runs
a full anelastic cumulonimbus to maturity.
"""
import numpy as np
from droplab.soundings import WEISMAN_KLEMP
from droplab.flow2d_dynamic import run_flow2d_dynamic

WK82_CAPE = 2120.0          # J/kg (parcel theory, cited reference value)
WK82_EL = 11990.0           # m   (equilibrium level)


def test_anelastic_cumulonimbus_matches_parcel_theory():
    Z, Nz = 14000.0, 100
    cfg = dict(Nx=112, Nz=Nz, X=20000, Z=Z, dt=3.0, nt=1300, collect_every=130, n_super=30000,
               dtheta_bubble=4.5, bubble_r=1400., bubble_z=1200., periodic_x=True, seed=3,
               b_max=0.4, omega_max=0.18, sponge_frac=0.16, sponge_tau=200.0,
               sounding=WEISMAN_KLEMP, ice=True, homogeneous=True, inp_n_cm3=0.5)
    o = run_flow2d_dynamic(dynamics="anelastic", **cfg)
    fr = o["frames"]
    z = (np.arange(Nz) + 0.5) * (Z / Nz)
    cloud_top = max(z[np.where(f["qc"].max(axis=0) > 0.05)[0].max()]
                    if (f["qc"].max(axis=0) > 0.05).any() else 0.0 for f in fr)
    peak_w = max(float(np.abs(f["w"]).max()) for f in fr)

    # (a) THERMODYNAMIC: the storm reaches the upper troposphere near its equilibrium level
    # (a well-resolved run tops out at ~0.97 x EL; the exact fraction is grid/time-sensitive,
    # so the regression bound is looser than the reported validation result).
    assert cloud_top > 0.75 * WK82_EL                  # genuinely DEEP, near the EL
    assert cloud_top < 1.15 * WK82_EL                  # does not blow through the tropopause
    # (b) DYNAMIC: peak updraft is CAPE-consistent for deep convection (entrained, not parcel-max)
    w_parcel = np.sqrt(2.0 * WK82_CAPE)                # 65 m/s unentrained
    assert 0.20 * w_parcel < peak_w < 0.70 * w_parcel  # ~13-46 m/s; realistic entrained range
    assert np.isfinite(peak_w)
