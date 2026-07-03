"""Climate-intervention diagnostic gates.

Checks the optical-depth identity (the two equivalent tau formulas agree) and the
Twomey DIRECTION: splitting the same liquid water into more, smaller droplets must
shrink the effective radius and raise the optical depth and albedo.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")

from droplab.parameters import rho_liq, pi
from droplab.flow2d import Flow2D
from droplab.climate_diag import column_optics


def _drops_for_lwc(n_drops, r_um, A_each, flow):
    """Place n_drops identical super-droplets of radius r_um, multiplicity A_each,
    all in column 0 of the domain."""
    r = r_um * 1e-6
    M = np.full(n_drops, 4.0 / 3.0 * pi * rho_liq * r ** 3 * A_each)
    A = np.full(n_drops, float(A_each))
    x = np.full(n_drops, 0.5 * flow.dx)               # column 0
    z = np.full(n_drops, 0.5 * flow.dz)
    return M, A, x, z


def test_tau_identity_matches_lwp_reff_form():
    """tau = (2*pi/area)*sum(A r^2) must equal 3*LWP/(2*rho_w*r_eff)."""
    flow = Flow2D(X=1000.0, Z=1000.0, Nx=20, Nz=20)
    M, A, x, z = _drops_for_lwc(50, r_um=12.0, A_each=1e8, flow=flow)
    o = column_optics(M, A, x, z, flow, depth=1.0)
    i = 0
    tau_direct = o["tau"][i]
    tau_from_lwp = 3.0 * o["lwp"][i] / (2.0 * rho_liq * o["reff"][i])
    assert np.isclose(tau_direct, tau_from_lwp, rtol=1e-10), \
        f"tau formulas disagree: {tau_direct} vs {tau_from_lwp}"


def test_twomey_more_smaller_drops_brighten():
    """Same liquid water path, but twice the droplets at smaller radius (r/2^(1/3))
    keeps LWP fixed while halving the volume per drop -> r_eff down, tau & albedo up."""
    flow = Flow2D(X=1000.0, Z=1000.0, Nx=20, Nz=20)
    # clean: N drops of radius r0
    Mc, Ac, xc, zc = _drops_for_lwc(40, r_um=14.0, A_each=1e8, flow=flow)
    clean = column_optics(Mc, Ac, xc, zc, flow)
    # polluted: SAME total water, split into 2x the drops at smaller radius
    r_small = 14.0 / 2.0 ** (1.0 / 3.0)               # half the volume each
    Mp, Ap, xp, zp = _drops_for_lwc(40, r_um=r_small, A_each=2e8, flow=flow)
    poll = column_optics(Mp, Ap, xp, zp, flow)

    assert np.isclose(clean["lwp_mean"], poll["lwp_mean"], rtol=1e-9), "LWP should be unchanged"
    assert poll["reff_mean"] < clean["reff_mean"], "polluted r_eff must be smaller"
    assert poll["tau_mean"] > clean["tau_mean"], "polluted optical depth must be larger"
    assert poll["albedo_mean"] > clean["albedo_mean"], "polluted albedo must be higher (Twomey)"


def test_mcb_brightens_full_model():
    """End-to-end: injecting sea-salt CCN into a running stratocumulus deck must,
    in the domain mean, shrink the effective radius and raise the albedo (Twomey).
    Same turbulence seed for both runs so the chaotic part cancels in the mean."""
    from droplab.flow2d_dynamic import run_flow2d_dynamic
    from droplab.soundings import DYCOMS, DYCOMS_RADIATION
    from droplab.climate_diag import twomey_report
    cfg = dict(dt=1.0, Nx=48, Nz=32, X=2400.0, Z=1200.0, n_super=24000,
               sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
               N_modes=(60.,), pert_amp=0.1, nu=6, nu_scalar=1.5, collisions=True,
               switch_TICE=True, eps=0.01, sediment=True, collect_every=100000, seed=1)
    spec = dict(t_inject=40.0, x_frac=(0.0, 1.0), z_lo=50.0, z_hi=500.0,
                N_cm3=600.0, r_um=0.1, kappa=1.2, n_super=8000)
    base = run_flow2d_dynamic(nt=600, **cfg)
    seeded = run_flow2d_dynamic(nt=600, seeding=spec, **cfg)
    flow = Flow2D(X=2400.0, Z=1200.0, Nx=48, Nz=32)
    rep = twomey_report(base, seeded, flow)
    assert rep["d_reff_um"] < 0.0, f"MCB did not shrink r_eff (Δ={rep['d_reff_um']:.2f} µm)"
    assert rep["d_albedo"] > 0.0, f"MCB did not brighten the cloud (Δ={rep['d_albedo']:.4f})"
    assert rep["forcing_wm2"] < 0.0, "MCB forcing should be a cooling (negative)"


def test_precip_vs_nonprecip_aerosol_control():
    """Aerosol controls the drizzle: a CLEAN deck (few CCN, big drops) drizzles hard
    and stays dim (precipitating); a POLLUTED deck (many CCN, small drops) suppresses
    drizzle and brightens (non-precipitating). Same meteorology."""
    from droplab.flow2d_dynamic import run_flow2d_dynamic
    from droplab.soundings import DYCOMS, DYCOMS_RADIATION
    cfg = dict(dt=1.0, Nx=48, Nz=32, X=2400.0, Z=1200.0, n_super=24000,
               sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
               pert_amp=0.1, nu=6, nu_scalar=1.5, collisions=True, switch_TICE=True,
               eps=0.01, sediment=True, collect_every=100000, seed=5)
    flow = Flow2D(X=2400.0, Z=1200.0, Nx=48, Nz=32)
    clean = run_flow2d_dynamic(nt=800, N_modes=(20.,), **cfg)
    poll = run_flow2d_dynamic(nt=800, N_modes=(400.,), **cfg)
    oc = column_optics(clean["M"], clean["A"], clean["x"], clean["z"], flow)
    op = column_optics(poll["M"], poll["A"], poll["x"], poll["z"], flow)

    # Drizzle formation via rain-sized-drop MASS (drops > 40 um), not surface-precip timing:
    # surf_precip is unreliable below ~20-min runs (it can even reverse sign across platforms),
    # so it is the wrong cross-platform metric. The mass of large drops aloft is the robust,
    # bulk signature of collision-coalescence and shows the clean/polluted contrast directly.
    def _rain_mass(o):
        M, A = np.asarray(o["M"]), np.asarray(o["A"])
        r_um = np.where(A > 0, (M / (A * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0), 0.0) * 1e6
        return float(M[r_um > 40.0].sum())
    assert _rain_mass(clean) > _rain_mass(poll), "clean deck must form more drizzle"
    assert oc["reff_mean"] > op["reff_mean"], "clean drops must be larger"
    assert op["albedo_mean"] > oc["albedo_mean"] + 0.1, "polluted deck must be markedly brighter"


def test_haze_excluded_from_optics():
    """Sub-micron haze (r < r_min) must not contribute to the cloud optics."""
    flow = Flow2D(X=1000.0, Z=1000.0, Nx=20, Nz=20)
    M, A, x, z = _drops_for_lwc(100, r_um=0.1, A_each=1e10, flow=flow)  # all haze
    o = column_optics(M, A, x, z, flow)
    assert o["tau_mean"] == 0.0, "haze leaked into optical depth"
    assert o["albedo_mean"] == 0.0, "haze produced spurious albedo"
