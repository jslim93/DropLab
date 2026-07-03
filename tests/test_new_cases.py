"""Regime gates for the three new 2D cases — each must show its defining signature:
RICO drizzles, fog sits at the surface, and the diurnal case grows clouds with the
heating.
"""
import math
import numpy as np
import matplotlib; matplotlib.use("Agg")

from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.soundings import RICO, RICO_FORCING, FOG, BOMEX, BOMEX_FORCING


def test_rico_makes_precipitating_cumulus():
    """RICO + low maritime aerosol must grow cumulus that drizzle (drops well past
    cloud-droplet size), unlike the non-precipitating BOMEX."""
    out = run_flow2d_dynamic(nt=800, dt=1.5, Nx=64, Nz=56, X=3200.0, Z=4000.0,
                             n_super=30000, sounding=RICO, forcing=RICO_FORCING,
                             N_modes=(50.,), pert_amp=0.1, nu=12, nu_scalar=1.5,
                             collisions=True, switch_TICE=True, eps=0.01, sediment=True,
                             collect_every=100000, seed=3)
    r = np.where(out["A"] > 0, (out["M"] / (out["A"] * 4 / 3 * np.pi * 1000.)) ** (1 / 3), 0.0)
    assert out["frames"][-1]["qc"].max() > 0.5, "no cumulus formed"
    # drops grow well past cloud-droplet size (~10 um) via collision-coalescence —
    # the warm-rain process a polluted/non-precipitating cloud would not sustain.
    # (the full-resolution run reaches mm drizzle; this short test reaches ~30 um.)
    assert r.max() * 1e6 > 25.0, "drops did not grow past cloud size — RICO should precipitate"


def test_fog_forms_at_the_surface():
    """Surface cooling in a stable, near-saturated layer must condense fog AT the
    ground (cloud water concentrated in the lowest cells, ~none aloft)."""
    out = run_flow2d_dynamic(nt=1000, dt=1.0, Nx=48, Nz=40, X=1600.0, Z=600.0,
                             n_super=20000, sounding=FOG, T0=283.0, RH0=0.99,
                             surface_cool=-6.0e-3, periodic_x=True, N_modes=(200.,),
                             pert_amp=0.02, nu=4, nu_scalar=0.2, collisions=False,
                             sediment=False, b_max=0.05, omega_max=0.02,
                             collect_every=100000, seed=3)
    qc = out["frames"][-1]["qc"]                      # (Nx, Nz)
    prof = qc.mean(axis=0)
    assert qc.max() > 0.1, "no fog formed"
    assert prof[:4].sum() > 5.0 * prof[4:].sum() + 1e-9, "fog is not surface-confined"


def test_diurnal_heating_grows_clouds():
    """With diurnal surface heating, cloud cover must rise from the (low-sun) morning
    to the (high-sun) midday — the diurnal growth of continental cumulus."""
    P = 14400.0
    out = run_flow2d_dynamic(nt=2000, dt=2.0, Nx=64, Nz=64, X=3200.0, Z=3000.0,
                             n_super=25000, sounding=BOMEX, forcing=BOMEX_FORCING,
                             diurnal_period=P, N_modes=(200.,), pert_amp=0.1, nu=14,
                             nu_scalar=1.5, collisions=False, sediment=True,
                             collect_every=100, seed=3)
    cf = {}
    for f in out["frames"]:
        t = f["step"] * 2.0
        cf[t] = float((f["qc"].max(axis=1) > 0.05).mean())
    morning = max(v for t, v in cf.items() if 400 < t < 1000)    # solar ~0.3
    midday = max(v for t, v in cf.items() if 3000 < t < 4000)    # solar ~1.0
    assert midday > morning, f"no diurnal growth (midday {midday} vs morning {morning})"


def _cloud_tilt(out):
    """Horizontal displacement between the cloud's base and top x-centroids (m)."""
    f = out["frames"][-1]; qc = f["qc"]; flow = out["flow"]
    xc = (np.arange(flow.Nx) + 0.5) * flow.dx
    per_lev = qc.sum(axis=0)
    lev = np.where(per_lev > 0.1 * per_lev.max())[0]
    if len(lev) < 2:
        return 0.0
    base = (qc[:, lev[0]] * xc).sum() / qc[:, lev[0]].sum()
    top = (qc[:, lev[-1]] * xc).sum() / qc[:, lev[-1]].sum()
    return top - base


def test_arctic_mixed_phase_glaciates():
    from examples.cloud_cases import CASES
    from droplab.flow2d_dynamic import run_flow2d_dynamic
    # reduced config that nucleates ice via ABIFM on the base INP population (real
    # cold MOSAiC sounding; ABIFM is efficient at ~-20 C so a realistic INP suffices).
    cfg = dict(CASES["arctic"])
    cfg.update(Nx=48, Nz=32, n_super=12000, nt=600, collect_every=300, collisions=False,
               inp_n_cm3=0.5, inp_r_um=4.0)
    out = run_flow2d_dynamic(**cfg)
    assert (out["inp"] > 0).sum() > 0                  # base INP population present
    assert int(out["phase"].sum()) > 0                 # ice nucleated (ABIFM immersion)
    assert out["frames"][-1]["q_ice"].sum() > 0.0      # ice water present


def test_wind_shear_tilts_convection():
    """A linear wind shear must tilt the updraft (base and top x-centroids well
    separated); without shear the thermal stays upright."""
    cfg = dict(nt=300, dt=2.0, Nx=96, Nz=64, X=4800.0, Z=3000.0, n_super=25000,
               periodic_x=True, RH0=0.93, z_bl=600.0, z_inv=1900.0, dtheta_inv=4.0,
               dtheta_bubble=2.5, bubble_z=500.0, bubble_r=500.0, N_modes=(200.,),
               nu=16, nu_scalar=1.5, collisions=False, sediment=True, b_max=0.18,
               omega_max=0.05, collect_every=100000, seed=3)
    upright = _cloud_tilt(run_flow2d_dynamic(wind_shear=0.0, **cfg))
    sheared = _cloud_tilt(run_flow2d_dynamic(wind_shear=5.0e-3, **cfg))
    assert abs(sheared) > 300.0, f"shear did not tilt the cloud (tilt={sheared:.0f} m)"
    assert abs(sheared) > abs(upright) + 200.0, f"sheared {sheared:.0f} vs upright {upright:.0f}"
