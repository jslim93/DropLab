"""Aerosol-cloud-interaction (ACI) diagnostics for the 2D stratocumulus model.

These turn small ensembles over background aerosol number into the standard
sensitivities a climate scientist reports. The lever is the cloud-condensation-
nucleus number N_a: more CCN split the same liquid water into more, smaller
droplets (the Twomey effect), brightening the cloud and suppressing drizzle.

Sensitivities (fitted as least-squares slopes in log space over an ensemble):
  ACI_N = d ln N_d / d ln N_a    activation efficiency      (0..1; <1, competition)
  ACI_r = -d ln r_eff / d ln N_a Twomey radius susceptibility (ideal 1/3 at fixed LWP)
  S_pop = -d ln P / d ln N_a      precipitation susceptibility (>0; aerosol suppresses)

This module ONLY consumes the existing model API (it imports, never modifies, the
physics core or climate_diag). New diagnostics live here so the breakup/physics
lanes stay conflict-free.
"""
import numpy as np

from droplab.parameters import rho_liq, pi
from droplab.climate_diag import column_optics, toa_forcing
from droplab.flow2d_dynamic import run_flow2d_dynamic


def cloud_droplet_number(M, A, x, z, flow, depth=1.0, r_min_um=1.0):
    """Domain-mean cloud-droplet number concentration N_d [cm^-3].

    Counts only activated drops (r > r_min, mirroring column_optics) and divides
    the per-column real-droplet count by the cloud volume dx*depth*H, where H is
    the vertical extent of the activated population (the cloud-layer thickness).

    Returns dict(nd_col [cm^-3, length Nx], nd_mean [cm^-3, over cloudy columns]).
    """
    Nx = flow.Nx
    col = np.clip((x / flow.dx).astype(np.int64), 0, Nx - 1)
    r = np.where(A > 0.0, (M / (A * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0), 0.0)
    cloud = r > (r_min_um * 1e-6)

    nd_col = np.zeros(Nx)
    if not cloud.any():
        return dict(nd_col=nd_col, nd_mean=0.0)

    H = max(float(z[cloud].max() - z[cloud].min()), flow.dz)   # cloud-layer thickness [m]
    counts = np.zeros(Nx)
    np.add.at(counts, col[cloud], A[cloud])                    # real drops per column
    nd_col = counts / (flow.dx * depth * H) / 1e6              # m^-3 -> cm^-3
    cloudy = counts > 0
    nd_mean = float(nd_col[cloudy].mean()) if cloudy.any() else 0.0
    return dict(nd_col=nd_col, nd_mean=nd_mean)


def rain_water_path(M, A, x, z, flow, depth=1.0, r_rain_um=40.0):
    """Domain-mean rain-water path RWP [kg/m^2] from rain-category drops (r > 40 um).

    Column-integrated rain mass divided by horizontal area. This is the precip
    proxy used for precipitation susceptibility: it is far less Monte-Carlo-noisy
    at low N_d than accumulated surface precip, and it registers drizzle as soon
    as rain-size drops form (rather than only after they reach the ground).

    Returns dict(rwp_col [kg/m^2, length Nx], rwp_mean [kg/m^2]).
    """
    Nx = flow.Nx
    col = np.clip((x / flow.dx).astype(np.int64), 0, Nx - 1)
    r = np.where(A > 0.0, (M / (A * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0), 0.0)
    rain = r > (r_rain_um * 1e-6)
    rwp_col = np.zeros(Nx)
    if rain.any():
        np.add.at(rwp_col, col[rain], M[rain])
    rwp_col = rwp_col / (flow.dx * depth)                       # kg/m^2
    return dict(rwp_col=rwp_col, rwp_mean=float(rwp_col.mean()))


def _loglog_slope(xa, ya):
    """Least-squares slope of ln(ya) vs ln(xa) and its r^2."""
    lx, ly = np.log(np.asarray(xa, float)), np.log(np.asarray(ya, float))
    m, b = np.polyfit(lx, ly, 1)
    pred = m * lx + b
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(m), float(r2)


def lwp_susceptibility(Nd, lwp):
    """LWP adjustment dln(LWP)/dln(N_d) — the LWP-N_d relationship (Gryspeerdt 2019).

    Positive = ascending / precipitation-suppression branch (aerosol suppresses
    drizzle, the cloud retains water). Negative = descending / entrainment-drying
    branch (smaller drops -> enhanced cloud-top evaporation -> the cloud dries).
    The full curve is the "inverted V"; a single slope reports the net tendency, so
    inspect the LWP(N_d) array for the shape. Returns (slope, r2).
    """
    return _loglog_slope(Nd, lwp)


def aci_susceptibility(N_list, runner, n_jobs=1):
    """Fit ACI sensitivities from an ensemble.

    runner(N_a) must return (N_d, r_eff, albedo, precip, lwp). Precipitation
    susceptibility is fitted only over runs with precip > 0 (clean clouds drizzle,
    polluted ones do not — that contrast IS the susceptibility). The LWP
    susceptibility ACI_L = dln(LWP)/dln(N_d) is the inverted-V relationship.

    The N_list members are independent runs; pass n_jobs != 1 to fan them across
    worker processes (each pinned to one numba thread to avoid oversubscription).
    Results are bit-identical to the serial sweep.
    """
    Na = np.asarray(N_list, float)
    if n_jobs == 1:
        results = [runner(N) for N in N_list]
    else:
        from droplab.flow2d_ensemble import run_parallel
        results = run_parallel(list(N_list), runner, n_jobs=n_jobs)
    Nd, reff, alb, prc, lwp = ([] for _ in range(5))
    for nd, re, ab, pr, lw in results:
        Nd.append(nd); reff.append(re); alb.append(ab); prc.append(pr); lwp.append(lw)
    Nd, reff, alb, prc, lwp = map(np.asarray, (Nd, reff, alb, prc, lwp))

    aci_N, r2_N = _loglog_slope(Na, Nd)
    slope_r, r2_r = _loglog_slope(Na, reff)
    pmask = prc > 0
    if pmask.sum() >= 2:
        slope_p, r2_p = _loglog_slope(Na[pmask], prc[pmask])
        S_pop = -slope_p
    else:
        S_pop, r2_p = float("nan"), float("nan")
    lmask = (Nd > 0) & (lwp > 0)
    if lmask.sum() >= 2:
        ACI_L, r2_L = lwp_susceptibility(Nd[lmask], lwp[lmask])
    else:
        ACI_L, r2_L = float("nan"), float("nan")
    amask = alb > 0
    if amask.sum() >= 2:
        S_albedo, r2_A = _loglog_slope(Na[amask], alb[amask])
    else:
        S_albedo, r2_A = float("nan"), float("nan")

    return dict(
        ACI_N=aci_N, ACI_r=-slope_r, S_pop=S_pop, ACI_L=ACI_L, S_albedo=S_albedo,
        r2=dict(N=r2_N, r=r2_r, precip=r2_p, lwp=r2_L, albedo=r2_A),
        Na=Na, Nd=Nd, reff=reff, albedo=alb, precip=prc, lwp=lwp,
    )


def cloud_radiative_effect(albedo, S0=1361.0, mu0=0.5):
    """Short-wave cloud radiative effect [W/m^2] = -S0 * mu0 * albedo.

    Negative = cooling (the cloud reflects sunlight). Monotonically decreasing in
    albedo. S0 = solar constant, mu0 = cosine of the solar zenith angle.
    """
    return -S0 * mu0 * np.asarray(albedo, float)


def erfaci(albedo_pi, albedo_pd):
    """Effective radiative forcing from aerosol-cloud interaction [W/m^2].

    Between a clean pre-industrial (PI) and polluted present-day (PD) aerosol
    loading: dF = toa_forcing(albedo_pd - albedo_pi). PD brighter -> negative
    (cooling).
    """
    return float(toa_forcing(albedo_pd - albedo_pi))


def erfaci_decomposition(pi_out, pd_out, depth=1.0, g_asym=0.85):
    """Split ERFaci into the intrinsic Twomey term and the LWP-adjustment term.

    Computed PER COLUMN (cloud albedo is nonlinear in optical depth, so the split
    must be done column-by-column then domain-averaged, not from domain means). The
    Twomey counterfactual gives every present-day (polluted) column the
    pre-industrial (clean) LWP while keeping its polluted effective radius, isolating
    the brightening due to droplet number alone; the remainder is the LWP adjustment
    (the inverted-V response). The two terms sum exactly to the total ERFaci.

    pi_out, pd_out are run_flow2d_dynamic outputs that MUST share geometry and should
    use the same seed so columns align. Returns the three forcings [W/m^2] (negative
    = cooling) plus the domain-mean albedos used.
    """
    pi = column_optics(pi_out["M"], pi_out["A"], pi_out["x"], pi_out["z"],
                       pi_out["flow"], depth=depth, g_asym=g_asym)
    pd = column_optics(pd_out["M"], pd_out["A"], pd_out["x"], pd_out["z"],
                       pd_out["flow"], depth=depth, g_asym=g_asym)
    gp = 1.0 - g_asym
    lwp_pi, reff_pd = pi["lwp"], pd["reff"]
    # per-column Twomey counterfactual albedo: polluted reff at clean LWP
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(reff_pd) & (reff_pd > 0) & (lwp_pi > 0)
        tau_T = np.where(ok, 3.0 * lwp_pi / (2.0 * rho_liq * np.where(ok, reff_pd, 1.0)), 0.0)
    alb_T = (gp * tau_T) / (2.0 + gp * tau_T)            # = 0 where the cloud is removed

    A_pi = float(np.mean(pi["albedo"]))
    A_pd = float(np.mean(pd["albedo"]))
    A_T = float(np.mean(alb_T))
    rf_twomey = float(toa_forcing(A_T - A_pi))
    erf_adjust = float(toa_forcing(A_pd - A_T))
    return dict(
        RFaci_Twomey=rf_twomey,                          # droplet-number brightening
        ERFaci_adjustment=erf_adjust,                    # LWP (inverted-V) response
        ERFaci_total=rf_twomey + erf_adjust,             # == toa_forcing(A_pd - A_pi)
        albedo_pi=A_pi, albedo_pd=A_pd, albedo_twomey=A_T,
    )


def cloud_albedo_direct(tau, mu0, g_asym=0.85):
    """Conservative two-stream cloud albedo for a DIRECT beam at cosine-zenith mu0:
    R = a / (a + 2*mu0),  a = (1-g)*tau. Low sun (small mu0) -> longer slant path
    -> brighter. At mu0=1 this equals DropLab's column_optics albedo convention."""
    a = (1.0 - g_asym) * np.asarray(tau, float)
    return a / (a + 2.0 * mu0)


def cloud_albedo_diffuse(tau, g_asym=0.85):
    """Conservative two-stream cloud albedo for ISOTROPIC (diffuse) incidence:
    R_diff = 2 * integral_0^1 R_dir(mu) mu dmu = a - (a^2/2) ln((a+2)/a),  a=(1-g)tau.
    Brighter than overhead-direct (effective slant), darker than low-sun-direct."""
    a = (1.0 - g_asym) * np.asarray(tau, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rd = a - 0.5 * a ** 2 * np.log((a + 2.0) / np.where(a > 0, a, 1.0))
    return np.where(a > 0, Rd, 0.0)


def diffusion_brightening(out, delta_f_diff=0.2, mu0=0.7, depth=1.0, g_asym=0.85, S0=1361.0):
    """Gristey-2025 SAI 'diffusion-brightening' of low cloud, on a DropLab cloud field.

    Stratospheric aerosol scatters incoming sunlight, converting a fraction
    delta_f_diff of the DIRECT beam into DIFFUSE light WITHOUT touching the cloud's
    microphysics. Because cloud albedo depends on the incidence angle distribution,
    this changes per-column albedo by Delta = delta_f_diff*(R_diffuse - R_direct(mu0))
    (albedo is linear in the diffuse fraction at fixed tau, mu0). The shortwave
    'bonus' forcing is Delta_CRE = -S0*mu0*<Delta>.

    Sign is mu0-dependent: a bonus brightening (cooling) when the sun is high
    (mu0 > ~0.65), which can reverse at low sun. 1D two-stream, plane-parallel;
    delta_f_diff is a prescribed input, not derived from a stratospheric model.
    Reference: Gristey et al. (2025), GRL, doi:10.1029/2024GL113914.
    """
    o = column_optics(out["M"], out["A"], out["x"], out["z"], out["flow"],
                      depth=depth, g_asym=g_asym)
    tau = o["tau"]
    Rdir = cloud_albedo_direct(tau, mu0, g_asym)
    Rdif = cloud_albedo_diffuse(tau, g_asym)
    dR = float(np.mean(delta_f_diff * (Rdif - Rdir)))
    return dict(
        d_albedo=dR, d_CRE=-S0 * mu0 * dR, mu0=mu0, delta_f_diff=delta_f_diff,
        albedo_direct_mean=float(np.mean(Rdir)), albedo_diffuse_mean=float(np.mean(Rdif)),
    )


def make_runner(depth=1.0, **base_kwargs):
    """Build a runner(N_a) -> (N_d, r_eff, albedo, precip) over run_flow2d_dynamic.

    base_kwargs are the fixed configuration (resolution, sounding, forcing, seed);
    only N_modes is swept. The SAME seed across the ensemble keeps the slopes clean
    (aerosol is the only thing that changes).
    """
    base_kwargs.pop("N_modes", None)
    base_kwargs.pop("depth", None)

    def runner(N):
        out = run_flow2d_dynamic(N_modes=(float(N),), depth=depth, **base_kwargs)
        opt = column_optics(out["M"], out["A"], out["x"], out["z"], out["flow"], depth=depth)
        ndd = cloud_droplet_number(out["M"], out["A"], out["x"], out["z"], out["flow"], depth=depth)
        rwp = rain_water_path(out["M"], out["A"], out["x"], out["z"], out["flow"], depth=depth)
        cloudy = opt["lwp"] > 1e-4
        lwp_in = float(opt["lwp"][cloudy].mean()) if cloudy.any() else 0.0   # in-cloud LWP [kg/m^2]
        # precip proxy = rain-water path (drizzle-development length required; see make_runner docs)
        return ndd["nd_mean"], opt["reff_mean"], opt["albedo_mean"], rwp["rwp_mean"], lwp_in

    return runner
