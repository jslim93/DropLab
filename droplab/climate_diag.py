"""Climate-intervention diagnostics for the 2D stratocumulus model.

The lever in marine cloud brightening (MCB) and precipitation cloud-seeding is the
aerosol number: more cloud condensation nuclei split the SAME liquid water into
MORE, SMALLER droplets. Smaller droplets scatter sunlight more efficiently per unit
water (the Twomey effect), so the cloud's albedo rises and it reflects more sunlight
back to space — a cooling. These diagnostics turn the super-droplet population into
the quantities a climate scientist reads: effective radius, optical depth, and the
short-wave cloud albedo.

Key relations (cloud column i):
  LWP_i   = sum_drops M / (dx*depth)                      liquid water path  [kg/m^2]
  r_eff_i = sum(A r^3) / sum(A r^2)                       effective radius   [m]
  tau_i   = (2*pi/(dx*depth)) * sum(A r^2)                optical depth (Q_ext~2)
          = 3*LWP_i / (2*rho_w*r_eff_i)                   (identical — see module docs)
  albedo  = (1-g)*tau / (2 + (1-g)*tau)                   two-stream, conservative
with asymmetry g~0.85 for liquid cloud. Only droplets with r > r_min (activated
cloud drops, not haze) contribute to the optics.
"""
import numpy as np

from droplab.parameters import rho_liq, pi


def column_optics(M, A, x, z, flow, depth=1.0, g_asym=0.85, r_min_um=1.0):
    """Per-x-column cloud optics from the super-droplet population.

    Returns a dict with arrays of length Nx (one value per column):
      lwp   liquid water path           [kg/m^2]
      reff  effective radius            [m]   (NaN where the column is clear)
      tau   short-wave optical depth    [-]
      albedo  cloud albedo              [-]
    plus scalar domain means: albedo_mean, tau_mean, reff_mean, lwp_mean.
    """
    Nx = flow.Nx
    col = np.clip((x / flow.dx).astype(np.int64), 0, Nx - 1)
    r = np.where(A > 0.0, (M / (A * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0), 0.0)
    cloud = r > (r_min_um * 1e-6)                      # activated drops only

    area = flow.dx * depth                              # horizontal area per column [m^2]
    Ar2 = np.zeros(Nx); Ar3 = np.zeros(Nx); Msum = np.zeros(Nx)
    np.add.at(Ar2, col[cloud], A[cloud] * r[cloud] ** 2)
    np.add.at(Ar3, col[cloud], A[cloud] * r[cloud] ** 3)
    np.add.at(Msum, col[cloud], M[cloud])

    lwp = Msum / area
    with np.errstate(invalid="ignore", divide="ignore"):
        reff = np.where(Ar2 > 0.0, Ar3 / Ar2, np.nan)
    tau = (2.0 * pi / area) * Ar2
    gp = 1.0 - g_asym
    albedo = (gp * tau) / (2.0 + gp * tau)

    cloudy = lwp > 1e-4                                 # columns with real cloud
    return dict(
        lwp=lwp, reff=reff, tau=tau, albedo=albedo,
        albedo_mean=float(albedo.mean()),              # whole-domain (clear cols -> 0)
        tau_mean=float(tau.mean()),
        reff_mean=float(np.nanmean(reff[cloudy])) if cloudy.any() else float("nan"),
        lwp_mean=float(lwp.mean()),
        cloud_fraction=float(cloudy.mean()),
    )


def optics_from_frame(frame, flow, depth=1.0, **kw):
    """Per-column optics from a captured animation frame (which stores droplet
    radius and multiplicity rather than mass). Used to plot albedo(t)."""
    r = frame["r_um"] * 1e-6
    A = frame["A"]
    M = A * 4.0 / 3.0 * pi * rho_liq * r ** 3
    return column_optics(M, A, frame["x"], frame["z"], flow, depth, **kw)


def toa_forcing(d_albedo, S0=1361.0, T_atm=0.75):
    """Rough top-of-atmosphere shortwave forcing from a cloud-albedo change:
    dF = -S0/4 * T_atm * d_albedo  [W/m^2] (negative = cooling)."""
    return -(S0 / 4.0) * T_atm * d_albedo


SIGMA_SB = 5.670374419e-8                                   # Stefan-Boltzmann [W m^-2 K^-4]


def lw_emissivity(lwp_gm2, iwp_gm2, kappa_liq=0.13, kappa_ice=0.06):
    """Broadband long-wave cloud emissivity from the water/ice path (grey approximation,
    Stephens 1978-style): eps = 1 - exp(-kappa_liq*LWP - kappa_ice*IWP), mass absorption
    coefficients in m^2 g^-1. eps -> 1 for an optically thick cloud, 0 for clear sky."""
    return 1.0 - np.exp(-kappa_liq * np.asarray(lwp_gm2) - kappa_ice * np.asarray(iwp_gm2))


def cloud_radiative_effect(M, A, x, z, flow, T_col, phase=None, depth=1.0, mu0=0.5,
                           S0=1361.0, T_atm=0.75, g_asym=0.85, r_min_um=1.0,
                           kappa_liq=0.13, kappa_ice=0.06):
    """Per-column SHORT-WAVE, LONG-WAVE, and NET cloud radiative effect [W m^-2].

    SW (reflection, cooling, <=0): SWCRE = -(S0/4)*mu0*T_atm*albedo, albedo from the liquid
    optics (Twomey). mu0 in [0,1] is the daylight factor (0 = polar night -> no SW).
    LW (greenhouse, warming, >=0): LWCRE = eps_lw * sigma * (T_sfc^4 - T_top^4): the cloud
    emits from its cold top instead of the warm surface, cutting outgoing long-wave; eps_lw
    from the liquid+ice path. NET = SW + LW: low warm decks cool (SW wins), high/cold or
    polar-night clouds warm (LW wins). Clear columns contribute 0.
    """
    Nx, Nz = flow.Nx, flow.Nz
    col = np.clip((x / flow.dx).astype(np.int64), 0, Nx - 1)
    kz = np.clip((z / flow.dz).astype(np.int64), 0, Nz - 1)
    r = np.where(A > 0.0, (M / (A * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0), 0.0)
    is_ice = (phase == 1) if phase is not None else np.zeros(M.shape[0], bool)
    cloud = r > (r_min_um * 1e-6)

    area = flow.dx * depth
    Mliq = np.zeros(Nx); Mice = np.zeros(Nx)
    np.add.at(Mliq, col[cloud & ~is_ice], M[cloud & ~is_ice])
    np.add.at(Mice, col[cloud & is_ice], M[cloud & is_ice])
    lwp_gm2 = Mliq / area * 1.0e3                            # kg/m^2 -> g/m^2
    iwp_gm2 = Mice / area * 1.0e3

    sw = column_optics(M, A, x, z, flow, depth, g_asym, r_min_um)
    swcre = -(S0 / 4.0) * mu0 * T_atm * sw["albedo"]        # per column (<=0)

    ktop = np.full(Nx, -1, dtype=np.int64)
    np.maximum.at(ktop, col[cloud], kz[cloud])
    eps = lw_emissivity(lwp_gm2, iwp_gm2, kappa_liq, kappa_ice)
    T_sfc = float(T_col[0])
    Ttop = np.where(ktop >= 0, T_col[np.clip(ktop, 0, Nz - 1)], T_sfc)
    lwcre = np.where(ktop >= 0, eps * SIGMA_SB * (T_sfc ** 4 - Ttop ** 4), 0.0)

    net = swcre + lwcre
    cloudy = (lwp_gm2 + iwp_gm2) > 0.1
    return dict(
        swcre=swcre, lwcre=lwcre, net=net,
        swcre_mean=float(swcre.mean()), lwcre_mean=float(lwcre.mean()),
        net_mean=float(net.mean()),
        lwp_gm2=lwp_gm2, iwp_gm2=iwp_gm2,
        cloud_fraction=float(cloudy.mean()),
    )


def twomey_report(base_out, seeded_out, flow, depth=1.0, **kw):
    """Compare a baseline run with a seeded run and quantify the brightening.

    Returns a dict with both columns' domain-mean optics and the deltas a climate
    study reports: change in effective radius, optical depth, albedo, and the
    short-wave radiative forcing dF = -S0/4 * (1-A_atm) * d(albedo) as a rough
    top-of-atmosphere estimate (S0=1361 W/m^2, clear-sky transmittance ~0.75)."""
    b = column_optics(base_out["M"], base_out["A"], base_out["x"], base_out["z"],
                       flow, depth, **kw)
    s = column_optics(seeded_out["M"], seeded_out["A"], seeded_out["x"], seeded_out["z"],
                      flow, depth, **kw)
    dalbedo = s["albedo_mean"] - b["albedo_mean"]
    forcing = toa_forcing(dalbedo)                      # W/m^2 (negative = cooling)
    return dict(
        base=b, seeded=s,
        d_reff_um=(s["reff_mean"] - b["reff_mean"]) * 1e6,
        d_tau=s["tau_mean"] - b["tau_mean"],
        d_albedo=dalbedo,
        forcing_wm2=forcing,
    )


def radiative_report(base_out, seeded_out, flow, depth=1.0, mu0=0.5, **kw):
    """Short-wave / long-wave / net cloud radiative effect of an intervention.

    Reports each run's domain-mean SW, LW, and net CRE [W m^-2] and the seeded-minus-base
    deltas, so a warm-cloud intervention (which mostly changes SW) and a mixed-phase one
    (which can change LW through ice and cloud-top temperature) are evaluated on the same
    footing. Needs T_col and phase in the run outputs (both exposed by run_flow2d_dynamic)."""
    b = cloud_radiative_effect(base_out["M"], base_out["A"], base_out["x"], base_out["z"],
                               flow, base_out["T_col"], phase=base_out.get("phase"),
                               depth=depth, mu0=mu0, **kw)
    s = cloud_radiative_effect(seeded_out["M"], seeded_out["A"], seeded_out["x"],
                               seeded_out["z"], flow, seeded_out["T_col"],
                               phase=seeded_out.get("phase"), depth=depth, mu0=mu0, **kw)
    return dict(
        base=b, seeded=s,
        d_sw=s["swcre_mean"] - b["swcre_mean"],
        d_lw=s["lwcre_mean"] - b["lwcre_mean"],
        d_net=s["net_mean"] - b["net_mean"],
    )
