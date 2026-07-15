"""Shape-resolving ICE HABIT prediction, ported faithfully from SAM6.10.10.LCM_JS
(MICRO_LAGRANGE/micro_cond.f90, micro_sedi.f90) -- the Chen & Lamb (1994) inherent-
growth-ratio habit model with the Shima et al. (2020) mass-distribution-hypothesis axis
evolution and the Welss & Seifert (2023) Böhm aspect-ratio fall speed.

Each ice super-droplet carries (a_axis, c_axis, rho_app): the equatorial and polar
semi-axes [m] and the apparent (bulk) density [kg/m^3]. The aspect ratio phi = c/a sets
the habit -- phi<1 oblate (plate), phi>1 prolate (column), phi=1 sphere. As ice grows by
vapour deposition the inherent growth ratio Gamma(T) partitions the volume change between
the a- and c-axes, so the shape EVOLVES with temperature (plates near -15 C, columns
colder/warmer) -- visualisable directly as ellipses.

OPT-IN: `habit=True`. With it off the a/c/rho arrays are None and ice stays single-sphere,
bit-for-bit the previous behaviour (golden gate).

References (verbatim from SAM-LCM):
  Chen & Lamb (1994) JAS 51, 1206  -- inherent growth ratio, capacitance, density.
  Shima et al. (2020) GMD 13, 4107 -- mass-distribution-hypothesis axis update.
  Welss & Seifert (2023); Böhm (1989, 1992) -- aspect-ratio terminal velocity.
  Wells et al. (2024) -- the Gamma(T) lookup table used here.
"""
import numpy as np

from droplab.parameters import rho_ice, rho_liq, rho_aero, l_s, rv, pi, g, muelq
from droplab._igr_table import IGR_T, IGR_G

R_MIN_AXIS = 1.0e-5            # 10 um: below this an ice particle grows isometrically (sphere)
M_FLOOR = 4.0 / 3.0 * pi * rho_ice * (0.001e-6) ** 3   # mass floor (0.001 um sphere)


# --------------------------------------------------------------------------------------
# inherent growth ratio Gamma(T)  -- Chen & Lamb (1994) Fig.3, Wells et al. (2024) table
# --------------------------------------------------------------------------------------
def gamma_ice(Tc):
    """Inherent growth ratio Gamma as a function of temperature Tc [deg C]. Gamma<1 favours
    the a-axis (plates), Gamma>1 the c-axis (columns). Linear interpolation on the 293-pt
    Wells (2024) table over -30..0 C. (SAM's gamma_ice has a legacy index-stride bug that
    under-samples the fine table; we use the correct linear interpolation -- DropLab is not
    bit-compared to SAM, and the intent is the Wells table value at T.)"""
    gam = np.interp(np.asarray(Tc, float), IGR_T, IGR_G)
    gam = np.where(Tc < -30.0, 1.28, gam)          # below the table (SAM constant)
    return gam


# --------------------------------------------------------------------------------------
# capacitance C(a,c)  -- Chen & Lamb (1994) Eqs 39-40
# --------------------------------------------------------------------------------------
def capacitance(a, c):
    """Electrostatic-analogue capacitance [m] of an oblate/prolate spheroid (a=equatorial,
    c=polar semi-axis). dm/dt = 4*pi*C*G*s_i."""
    a = np.asarray(a, float); c = np.asarray(c, float)
    phi = c / np.maximum(a, 1e-30)
    out = a.copy()                                 # sphere default (phi==1)
    ob = phi < 1.0 - 1e-9
    pr = phi > 1.0 + 1e-9
    if np.any(ob):
        eps = np.sqrt(np.clip(1.0 - phi[ob] ** 2, 0.0, 1.0))
        out[ob] = a[ob] * eps / np.arcsin(np.clip(eps, 0.0, 1.0 - 1e-12))
    if np.any(pr):
        eps = np.sqrt(np.clip(1.0 - phi[pr] ** (-2), 0.0, 1.0))
        out[pr] = c[pr] * eps / np.log((1.0 + eps) * phi[pr])
    return out


# --------------------------------------------------------------------------------------
# growth coefficients (reuse DropLab's g_ice thermodynamics for consistency)
# --------------------------------------------------------------------------------------
def _growth_coeffs(T, P):
    """Return (C_pre, D): the mass-growth prefactor C_pre = rho_ice * g_ice [kg/m/s, so
    dm/dt = 4 pi C C_pre s_i] and the vapour diffusivity D [m^2/s], using the SAME K, D,
    e_si as droplab.ice_microphysics.g_ice so the habit thermodynamics match the rest of
    DropLab's ice."""
    K = 7.94048e-5 * T + 0.00227011
    D = 0.211e-4 * (T / 273.15) ** 1.94 * (101325.0 / P)
    e_si = np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)
    Fd = rv * T / (e_si * D)
    Fk = (l_s / (rv * T) - 1.0) * l_s / (K * T)
    C_pre = 1.0 / (Fd + Fk)                         # = rho_ice * g_ice  (mass convention)
    return C_pre, D


# --------------------------------------------------------------------------------------
# deposition density  -- Chen & Lamb (1994) Eq.42 / Jensen & Harrington (2015)
# --------------------------------------------------------------------------------------
def deposition_density(gam, drho, a):
    """Density [kg/m^3] of newly deposited ice. drho is the vapour-excess (kg/m^3); plates
    (Gamma<1) below 100 um deposit at solid-ice density, otherwise the C&L94 empirical fit
    (the *1000 puts drho on the g/cm^3 scale of the fit)."""
    rd = rho_ice * np.exp(-3.0 * np.maximum(drho * 1000.0 - 0.05, 0.0) / np.maximum(gam, 1e-6))
    return np.where((gam < 1.0) & (a < 100e-6), rho_ice, rd)


# --------------------------------------------------------------------------------------
# ventilation  -- Chen & Lamb (1994) Eqs 29-30
# --------------------------------------------------------------------------------------
def ice_ventilation(a, c, w, T, rho_air, cap):
    """Returns (f_vent_mass, f_vent_axes). f_vent_mass scales the mass growth; f_vent_axes
    modifies Gamma -> Gamma* in the axis update."""
    dyn = muelq * ((273.15 + 110.4) / (T + 110.4)) * (T / 273.15) ** 1.5     # Sutherland
    D_air = dyn / rho_air
    D_v = 0.211e-4 * (T / 273.15) ** 1.94 * (101325.0 / (rho_air * 287.0 * T))
    N_Re = np.minimum(2.0 * np.maximum(a, c) * w / D_air, 120.0)
    N_Sc = D_air / D_v
    phi = c / np.maximum(a, 1e-30)
    XX = N_Sc ** (1.0 / 3.0) * np.sqrt(N_Re)
    XX = np.where(phi >= 1.0, XX * phi ** (-1.0 / 3.0), XX * phi ** (1.0 / 6.0))
    capr = np.maximum(cap, 1e-30)
    lo = XX < 1.0
    fvm = np.where(lo, 1.0 + 0.14 * XX ** 2, 0.86 + 0.28 * XX)
    num = np.where(lo, 1.0 + 0.14 * XX ** 2 * np.sqrt(c / capr), 0.86 + 0.28 * XX * np.sqrt(c / capr))
    den = np.where(lo, 1.0 + 0.14 * XX ** 2 * np.sqrt(a / capr), 0.86 + 0.28 * XX * np.sqrt(a / capr))
    fva = num / np.maximum(den, 1e-30)
    return fvm, fva


# --------------------------------------------------------------------------------------
# the habit step: grow mass by capacitance, then evolve axes + apparent density
# --------------------------------------------------------------------------------------
def grow_and_shape(m1, a, c, rho_app, T, P, S_ice, dt, rho_air, w, esatw_over_esati):
    """One depositional habit step for a population of ice particles (per REAL particle).

    m1 [kg] single-particle ice mass, a/c [m] semi-axes, rho_app [kg/m^3], T [K], P [Pa],
    S_ice ice supersaturation (e/esati-1), dt [s], rho_air [kg/m^3], w [m/s] fall speed,
    esatw_over_esati = esatw/esati (for the water-saturation cap on the habit SS).

    Returns (m_new, a_new, c_new, rho_new, dm) -- all per single particle; dm = m_new-m1.
    Mass growth uses the m^(2/3) prognostic (Chen-Lamb); axes follow the Shima (2020) MDH
    with the inherent growth ratio Gamma(T)."""
    C_pre, D = _growth_coeffs(T, P)
    cap = capacitance(a, c)
    fvm, fva = ice_ventilation(a, c, w, T, rho_air, cap)
    # mass growth (m^(2/3) prognostic), uncapped S_ice
    dm23 = 8.0 / 3.0 * pi * C_pre * (cap / np.maximum(m1, M_FLOOR) ** (1.0 / 3.0)) * fvm * S_ice
    m_new = np.maximum(np.maximum(m1, M_FLOOR) ** (2.0 / 3.0) + dt * dm23, M_FLOOR ** (2.0 / 3.0)) ** 1.5
    dm = m_new - m1
    # axis + density update (acd_update), water-saturation-capped SS for the density
    Tc = T - 273.15
    gam = gamma_ice(Tc)
    gam = np.where(np.maximum(a, c) < R_MIN_AXIS, 1.0, gam)        # tiny -> isometric
    s_cap = np.minimum(S_ice, esatw_over_esati - 1.0)             # Miller-Young cap
    drho = C_pre * s_cap / np.maximum(D, 1e-30)
    rho_dep = deposition_density(gam, drho, a)
    V_ini = m1 / np.maximum(rho_app, 1e-30)
    dV = np.where(dm < 0.0, dm / np.maximum(rho_app, 1e-30), dm / np.maximum(rho_dep, 1e-30))
    dlogV = np.clip(dV / np.maximum(V_ini, 1e-300), -2.0, 2.0)   # stability limiter (no overflow)
    gam_ast = gam * fva
    # MDH axis update (Shima 2020 Eqs 26-27); reset to sphere if sub-10um
    small = np.minimum(a, c) < R_MIN_AXIS
    r_sph = (np.maximum(m_new, M_FLOOR) / (4.0 / 3.0 * pi * rho_ice)) ** (1.0 / 3.0)
    a_new = np.where(small, r_sph, a * np.exp(dlogV / (2.0 + gam_ast)))
    c_new = np.where(small, r_sph, c * np.exp(dlogV * gam_ast / (2.0 + gam_ast)))
    rho_new = np.where(small, rho_ice,
                       np.clip(m_new / np.maximum(V_ini + dV, 1e-300), 50.0, rho_ice))
    return m_new, a_new, c_new, rho_new, dm


# --------------------------------------------------------------------------------------
# Böhm aspect-ratio terminal velocity  -- micro_sedi.f90 (Welss & Seifert 2023, App. A)
# --------------------------------------------------------------------------------------
def _N_Re_boehm(phi, q, m_ice, mu, rho_air):
    """Reynolds number from Welss (2023) Eqs A1a-A1j (X0=2.8e6)."""
    X0 = 2.8e6
    X = 8.0 * m_ice * g * rho_air / (pi * mu ** 2 * np.maximum(phi, 1.0) * q ** 0.25)
    k1 = np.maximum(0.82 + 0.18 * phi, 0.85)
    k2 = 0.37 + 0.63 / phi
    k3 = 1.33 / (np.maximum(np.log10(phi), 0.0) + 1.19)
    k_phi = np.minimum(np.minimum(k1, k2), k3)
    Gphi = np.clip(3.76 - 8.41 * phi + 9.18 * phi ** 2 - 3.53 * phi ** 3, 1.0, 1.98)
    CDPs = np.maximum(0.292 * k_phi * Gphi, 0.492 - 0.2 / np.sqrt(phi))
    CDP = np.maximum(1.0, q * (1.46 * q - 0.46) * CDPs)
    CDPp = CDP * (1.0 + 1.6 * (X / X0) ** 2) / (1.0 + (X / X0) ** 2)
    CD0 = 4.5 * k_phi ** 2 * np.maximum(phi, 1.0)
    beta = np.sqrt(1.0 + CDP / (6.0 * k_phi) * np.sqrt(X / CDPp)) - 1.0
    gam = (CD0 - CDP) / (4.0 * CDP)
    return 6.0 * k_phi / CDPp * beta ** 2 * (1.0 + 2.0 * beta * np.exp(-beta * gam)
                                             / ((2.0 + beta) * (1.0 + beta)))


def boehm_fallspeed(m_ice, a, c, rho_app, T, rho_air):
    """Terminal velocity [m/s] of a habit-resolved ice particle (single particle mass
    m_ice [kg]) via the Welss/Böhm aspect-ratio drag (micro_sedi.f90 sedi_Boehm)."""
    a = np.asarray(a, float); c = np.asarray(c, float)
    dyn = muelq * ((273.15 + 110.4) / (T + 110.4)) * (T / 273.15) ** 1.5
    phi = c / np.maximum(a, 1e-30)
    pro = phi > 1.0
    A_CE = np.where(pro, pi * a * c, pi * a ** 2)
    Aproj = np.where(pro, A_CE, A_CE * ((1.0 - phi) * (rho_app / rho_ice) + phi))
    q = Aproj / np.maximum(A_CE, 1e-30)
    Re_pro = _N_Re_boehm(phi, q, m_ice, dyn, rho_air)
    Dchar = 2.0 * a
    w_pro = dyn * Re_pro / (rho_air * np.maximum(Dchar, 1e-30))
    # prolate: blend with the cylinder branch (Welss Eqs 31-32)
    Re_cyl = _N_Re_boehm(phi, 4.0 / (pi * np.maximum(phi, 1e-30)), m_ice, dyn, rho_air)
    w_cyl = dyn * Re_cyl / (rho_air * np.maximum(Dchar, 1e-30))
    fx = np.exp(-0.3 * (phi - 1.0))
    return np.where(pro, fx * w_pro + (1.0 - fx) * w_cyl, w_pro)


# --------------------------------------------------------------------------------------
# driver-facing helpers (operate in place on the super-droplet shape arrays)
# --------------------------------------------------------------------------------------
def init_ice_shape(hab, M, A, phase):
    """Seed any ice super-droplet with no shape yet (hab[:,0]<=0) as a sphere of its current
    ice mass: a=c=(m1/(4/3 pi rho_ice))^(1/3), rho_app=rho_ice. Catches every ice-creation
    path (freezing, homogeneous, riming) in one call. `hab` is (N,3)=[a,c,rho]; mutates it."""
    new = (phase == 1) & (A > 0.0) & (M > 0.0) & (hab[:, 0] <= 0.0)
    if not np.any(new):
        return
    r = (M[new] / A[new] / (4.0 / 3.0 * pi * rho_ice)) ** (1.0 / 3.0)
    hab[new, 0] = r
    hab[new, 1] = r
    hab[new, 2] = rho_ice


def reset_melted_shape(hab, phase):
    """Liquid super-droplets carry no shape; zero it so a re-freeze re-seeds a sphere."""
    hab[phase == 0, :] = 0.0


def deposit_habit(M, A, Ns, phase, cidx, hab,
                  T_flat, P_flat, S_ice_flat, rho_air_flat, eswi_flat, dt):
    """Habit-resolving ice deposition (replaces the spherical r^2-law when habit is on):
    grows each ice super-droplet's mass by capacitance, evolves (a,c,rho_app)=hab columns,
    and returns the per-super-droplet mass change dM [kg SD-total] for the vapour/heat
    budget. Mutates M, `phase` and `hab`.

    Sublimation floors at the dry aerosol core (same convention as the spherical
    _ice_deposition and the liquid path's r_aero floor): a crystal sublimating to its
    core reverts to AEROSOL -- phase -> 0, M = liquid-convention dry-core mass
    (= Ns * rho_liq/rho_aero), shape cleared -- instead of becoming an inert M=0 ghost."""
    dM = np.zeros(M.shape[0])
    ice = np.flatnonzero((phase == 1) & (A > 0.0) & (M > 0.0) & (hab[:, 0] > 0.0))
    if ice.size == 0:
        return dM
    c = cidx[ice]
    m1 = M[ice] / A[ice]
    w = boehm_fallspeed(m1, hab[ice, 0], hab[ice, 1], hab[ice, 2], T_flat[c], rho_air_flat[c])
    m_new, a_new, c_new, rho_new, dm1 = grow_and_shape(
        m1, hab[ice, 0], hab[ice, 1], hab[ice, 2], T_flat[c], P_flat[c],
        S_ice_flat[c], dt, rho_air_flat[c], w, eswi_flat[c])
    # dry-core masses per particle: Ns = 4/3 pi rho_aero r_dry^3 A  =>  closed forms
    m_core_ice = (rho_ice / rho_aero) * Ns[ice] / A[ice]     # core in ice convention
    m_core_liq = (rho_liq / rho_aero) * Ns[ice] / A[ice]     # core in liquid convention
    # revert ONLY when sublimating (m_new < m1) AND the ice has retreated to the core;
    # a freshly frozen sub-core haze crystal that is GROWING must stay ice (see the
    # spherical _ice_deposition for the same gate).
    gone = (m_new < m1) & (m_new <= m_core_ice)
    M[ice] = np.where(gone, m_core_liq, m_new) * A[ice]
    hab[ice, 0] = np.where(gone, 0.0, a_new)
    hab[ice, 1] = np.where(gone, 0.0, c_new)
    hab[ice, 2] = np.where(gone, 0.0, rho_new)
    if gone.any():
        phase[ice[gone]] = 0                                  # back to the aerosol pool
    dM[ice] = M[ice] - m1 * A[ice]                            # exact vapour credit
    return dM


def freeze_to_sphere(r_liq):
    """When a liquid drop of radius r_liq [m] freezes, seed an ice sphere of equal mass:
    a = c = r_liq*(rho_liq/rho_ice)^(1/3), rho_app = rho_ice. Returns (a, c, rho)."""
    a = r_liq * (rho_liq / rho_ice) ** (1.0 / 3.0)
    return a, a.copy() if hasattr(a, "copy") else a, np.full_like(np.asarray(a, float), rho_ice)
