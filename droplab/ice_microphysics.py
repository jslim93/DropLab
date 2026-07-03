"""Ice / mixed-phase microphysics for the 2D Lagrangian model (DropLab v2).

Self-contained: saturation over ice, Bigg (1953) stochastic immersion freezing,
depositional growth (r^2-law), and ice fall speed. All gated by `ice=False` in the
driver, so the warm v1 path is bit-identical when ice is off.
"""
import numpy as np
from numba import njit, prange, vectorize

from droplab.parameters import rho_liq, rho_ice, rv, r_a, l_s, l_f, pi


def esati(T):
    """Saturation vapour pressure over ICE [Pa], Murphy & Koop (2005), valid
    110-273.16 K. Below 0 C this is LESS than e_sat over water -> WBF driver."""
    return np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)


@vectorize(["float64(float64)"], cache=True)
def _esati_v(T):
    """Grid-vectorised esati (numba ufunc), for per-cell use in the driver."""
    return np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)


def qvs_ice(T, P):
    """Saturation mixing ratio over ice [kg/kg]."""
    e = esati(T)
    return (r_a / rv) * e / (P - e)


@njit(cache=True)
def g_ice(T, P):
    """Ice deposition growth coefficient G_ice [m^2/s] in the r dr/dt = G_ice*S_ice
    convention (mirrors the liquid G in flow2d_dynamic._thermo, with rho_ice, l_s,
    e_si). K and D match the liquid path exactly for consistency."""
    K = 7.94048e-5 * T + 0.00227011                       # air thermal conductivity
    D = 0.211e-4 * (T / 273.15) ** 1.94 * (101325.0 / P)  # vapour diffusivity
    e_si = np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)
    Fd = rv * T / (e_si * D)
    Fk = (l_s / (rv * T) - 1.0) * l_s / (K * T)
    return 1.0 / (rho_ice * (Fd + Fk))


@njit(cache=True)
def ice_grow_r2(r, S_ice, G_ice, dt):
    """One analytic deposition step (r^2-law). Stable & non-iterative: integrating
    r dr/dt = G_ice*S_ice over dt gives r^2 += 2*G_ice*S_ice*dt. Clamps at zero."""
    r2 = r * r + 2.0 * G_ice * S_ice * dt
    return np.sqrt(r2) if r2 > 0.0 else 0.0


@njit(cache=True)
def bigg_prob(V, T, dt, a, B):
    """Bigg (1953) immersion-freezing probability for ONE drop of volume V over dt.
    J(T) = B*(exp(a*(T0-T)) - 1) is the volume nucleation rate; P = 1 - exp(-V*J*dt).
    Zero at/above the melting point."""
    if T >= 273.15:
        return 0.0
    J = B * (np.exp(a * (273.15 - T)) - 1.0)
    return 1.0 - np.exp(-V * J * dt)


@njit(cache=True)
def _bigg_freeze(M, A, phase, cidx, T_c, dt, a, B, frozen_mass):
    """All-or-nothing per super-droplet immersion freezing. For each LIQUID
    super-droplet, draw u~U(0,1); if u < P the WHOLE super-droplet flips to ice
    (phase 1) and its water mass M[i] is added to frozen_mass[cell] (for the
    latent-heat-of-fusion source). Serial (RNG order = reproducible), like collision.
    M is conserved across the phase change (water mass = ice mass)."""
    for i in range(M.shape[0]):
        if phase[i] != 0 or A[i] <= 0.0 or M[i] <= 0.0:
            continue
        T = T_c[i]
        if T >= 273.15:
            continue
        V = M[i] / (A[i] * rho_liq)                 # single-drop volume
        P = 1.0 - np.exp(-V * B * (np.exp(a * (273.15 - T)) - 1.0) * dt)
        if np.random.random() < P:
            phase[i] = 1
            frozen_mass[cidx[i]] += M[i]


@njit(cache=True)
def ice_fall_speed(M, A):
    """Terminal fall speed of an ice particle [m/s] from a Locatelli & Hobbs (1974)
    style power law v = 0.69 * D_mm^0.41 on the spherical-equivalent diameter (using
    rho_ice). The L&H constants take D in MILLIMETRES (1 mm aggregate -> ~0.69 m/s,
    realistic snow; 2 mm -> ~0.9 m/s), so convert the SI radius to mm. Per-droplet,
    RNG-free. (SAM6-LCM uses a fuller Best-number/Boehm drag scheme; this is the MVP
    approximation — still far slower than equal-mass rain.)"""
    if A <= 0.0 or M <= 0.0:
        return 0.0
    r = (M / (A * 4.0 / 3.0 * pi * rho_ice)) ** (1.0 / 3.0)
    D_mm = 2.0 * r * 1.0e3
    return 0.69 * D_mm ** 0.41


@njit(parallel=True, cache=True)
def _ice_deposition(M, A, phase, cidx, S_ice_c, G_ice_c, dt, dM_i):
    """Per-ICE-droplet depositional growth via the r^2-law against the cell's
    over-ice supersaturation. Parallel over droplets (each writes its own M[i],
    dM_i[i] -> no race; pure function -> thread-order-independent, bit-identical)."""
    for i in prange(M.shape[0]):
        if phase[i] != 1 or A[i] <= 0.0 or M[i] <= 0.0:
            dM_i[i] = 0.0
            continue
        c = cidx[i]
        r = (M[i] / (A[i] * 4.0 / 3.0 * pi * rho_ice)) ** 0.33333333333
        r2 = r * r + 2.0 * G_ice_c[c] * S_ice_c[c] * dt
        r = np.sqrt(r2) if r2 > 0.0 else 0.0
        M_old = M[i]
        M[i] = A[i] * 4.0 / 3.0 * pi * rho_ice * r ** 3.0
        dM_i[i] = M[i] - M_old


@njit(cache=True)
def _scatter_dM_ice(dM_i, cidx, dM_cell):
    """Serial ascending-order scatter of per-droplet ice mass change into per-cell
    totals (bit-reproducible sum order)."""
    for i in range(dM_i.shape[0]):
        dM_cell[cidx[i]] += dM_i[i]


# ABIFM (water-Activity-Based Immersion Freezing Model; Knopf & Alpert 2013):
#   log10( J_het [cm^-2 s^-1] ) = c + m * delta_aw,   delta_aw = 1 - e_si/e_sw.
# Each entry is (c, m). Values are SOURCE-VERIFIED from the SAM6-LCM config
# (micro_vars.f90 default + its two commented alternates). The SAM code does NOT name
# these to a mineral, so they are kept generic; m is per unit water-activity
# difference (dimensionless — the SAM "degC^-1" comment is a known unit-label error).
# Add named-mineral pairs (kaolinite / illite / feldspar, Knopf & Alpert 2013) here.
ABIFM_SPECIES = {
    "default": (-4.0, 35.0),     # SAM micro_vars.f90 default
    "abifm_a": (-8.61, 53.32),   # SAM alternate 1 (more active)
    "abifm_b": (-1.35, 22.62),   # SAM alternate 2
}


@njit(cache=True)
def abifm_prob(inp_area, esw, esi, dt, c, m):
    """ABIFM per-drop immersion-freezing probability over dt for an immersed INP
    surface area `inp_area` [m^2]. delta_aw = 1 - e_si/e_sw (Koop water-activity
    criterion at water saturation); J = 10^(c + m*delta_aw) * 1e4 [m^-2 s^-1];
    P = 1 - exp(-J*dt*inp_area). Zero if no INP."""
    if inp_area <= 0.0:
        return 0.0
    daw = 1.0 - esi / esw
    Js = 10.0 ** (c + m * daw) * 1.0e4
    return 1.0 - np.exp(-Js * dt * inp_area)


@njit(cache=True)
def _abifm_freeze(M, A, phase, inp, cidx, esw_c, esi_c, T_c, dt, c, m, frozen_mass):
    """All-or-nothing per-super-droplet ABIFM immersion freezing (serial -> RNG
    order reproducible). esw_c/esi_c/T_c are per-cell (flat, indexed by cidx); only
    supercooled (T<0C) liquid super-droplets that carry INP (inp>0) can freeze. On
    freezing phase->1 and the water mass is added to frozen_mass[cell] (fusion source)."""
    for i in range(M.shape[0]):
        if phase[i] != 0 or A[i] <= 0.0 or M[i] <= 0.0 or inp[i] <= 0.0:
            continue
        ci = cidx[i]
        if T_c[ci] >= 273.15:
            continue
        daw = 1.0 - esi_c[ci] / esw_c[ci]
        Js = 10.0 ** (c + m * daw) * 1.0e4
        P = 1.0 - np.exp(-Js * dt * inp[i])
        if np.random.random() < P:
            phase[i] = 1
            frozen_mass[ci] += M[i]


# Homogeneous freezing (Kuhn et al. 2011 classical nucleation theory, as in SAM6-LCM).
# Below ~239 K supercooled liquid freezes spontaneously without an ice nucleus; this is
# the dominant pathway at cold convective tops and in cirrus. Constants from Kuhn (2011).
T_HOM = 239.1                      # K — homogeneous freezing only acts below this
_HOM_NV = 3.35e28                  # water molecule number conc [m^-3]
_HOM_AV = -2.323e-18               # fit parameter 1 [J]
_HOM_BV = -1.075e-20              # fit parameter 2 [J/K]
_HOM_DW = -2.13e-20               # free-energy perturbation [J]
_HOM_SRF = 2.5e-9                  # surface-layer thickness [m]
_BOLTZ = 1.380649e-23             # Boltzmann [J/K]
_PLANCK = 6.62607015e-34          # Planck [J s]


@njit(cache=True)
def homogeneous_prob(r, T, dt):
    """Kuhn (2011) classical-nucleation-theory homogeneous-freezing probability for ONE
    drop of radius r [m] over dt. Volume nucleation rate J_v = nv*k*T/h*exp((-av+bv*T)/kT);
    total rate Js = J_v * V * (1 + 3*(srf/r)*exp(-dw/kT)); P = 1 - exp(-dt*Js). Zero at or
    above T_HOM."""
    if T >= T_HOM or r <= 0.0:
        return 0.0
    V = 4.18879020478639 * r ** 3.0                         # (4/3) pi r^3
    Jv = _HOM_NV * _BOLTZ * T / _PLANCK * np.exp((-_HOM_AV + _HOM_BV * T) / (_BOLTZ * T))
    surf = 1.0 + 3.0 * (_HOM_SRF / r) * np.exp(-_HOM_DW / (_BOLTZ * T))
    Js = Jv * V * surf
    return 1.0 - np.exp(-dt * Js)


@njit(cache=True)
def _homogeneous_freeze(M, A, phase, cidx, T_c, dt, frozen_mass):
    """All-or-nothing per-super-droplet homogeneous freezing below T_HOM (serial -> RNG
    order reproducible). T_c is per-cell (flat, indexed by cidx). On freezing phase->1 and
    the water mass is added to frozen_mass[cell] (latent-heat-of-fusion source); mass is
    conserved across the phase change."""
    for i in range(M.shape[0]):
        if phase[i] != 0 or A[i] <= 0.0 or M[i] <= 0.0:
            continue
        ci = cidx[i]
        T = T_c[ci]
        if T >= T_HOM:
            continue
        r = (M[i] / (A[i] * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0)
        P = homogeneous_prob(r, T, dt)
        if np.random.random() < P:
            phase[i] = 1
            frozen_mass[ci] += M[i]


T_MELT = 273.15                    # K — ice in air warmer than this melts to liquid


@njit(cache=True)
def _melt(M, A, phase, cidx, T_c, melted_mass):
    """Instantaneous melting: an ice super-droplet sitting in a cell warmer than T_MELT
    reverts to liquid (phase->0). Mass M and solute Ns are untouched (the phase flag is the
    only change), so freeze<->melt is fully mass-conserving. The melted mass is accumulated
    in melted_mass[cell] as a latent-heat-of-fusion SINK -- melting cools the air, the exact
    opposite sign of freezing. Serial loop keeps it deterministic (no RNG here: melting is
    all-or-nothing, not stochastic)."""
    for i in range(M.shape[0]):
        if phase[i] != 1 or A[i] <= 0.0 or M[i] <= 0.0:
            continue
        ci = cidx[i]
        if T_c[ci] > T_MELT:
            phase[i] = 0
            melted_mass[ci] += M[i]


# Hallett-Mossop rime-splintering (secondary ice production). Splinters are thrown off
# DURING riming, only in the -3..-8 C window, peaking at -5 C; ~350 per milligram of rime.
T_HM_LO = 265.15                   # K (-8 C)  edge: no splintering at/below
T_HM_PEAK = 268.15                 # K (-5 C)  peak splinter yield
T_HM_HI = 270.15                   # K (-3 C)  edge: no splintering at/above
HM_PER_KG = 3.5e8                  # splinters per kg of rime accreted (Hallett & Mossop 1974)
M_SPLINTER = 1.0e-13               # kg — initial ice mass of one splinter (~3 micron crystal)


@njit(cache=True)
def hm_factor(T):
    """Triangular Hallett-Mossop temperature efficiency: 0 outside -8..-3 C, 1 at -5 C."""
    if T <= T_HM_LO or T >= T_HM_HI:
        return 0.0
    if T <= T_HM_PEAK:
        return (T - T_HM_LO) / (T_HM_PEAK - T_HM_LO)
    return (T_HM_HI - T) / (T_HM_HI - T_HM_PEAK)


@njit(cache=True)
def _hallett_mossop(M, A, phase, cidx, rimed, T_c, splinters_out):
    """Secondary ice from rime splintering. For each cell that rimed inside the -3..-8 C
    window, N_HM = HM_PER_KG * rimed_mass * f_HM(T) new splinter crystals are produced and
    -- exactly like the breakup scheme's _merge_fragments_into_nearest -- their NUMBER is
    added to the multiplicity of an EXISTING ice super-droplet rather than spawning a new
    one, so the super-droplet count is fixed (no unbounded proliferation). The host SD's
    mass M is left unchanged (the splinters borrow mass from it), so total water is exactly
    conserved; the host's representative crystal size shifts down -- the standard cost of a
    fixed-count scheme -- and a cap keeps the per-crystal mass at or above one splinter.

    Two passes keep it O(n + cells): pass 1 finds, per cell, the ice SD with the most
    splinter-hosting capacity (largest M/M_SPLINTER - A, i.e. the rimer/graupel); pass 2
    adds the (capped) splinter number to it. splinters_out[cell] records the count added."""
    n = M.shape[0]
    ncell = rimed.shape[0]
    best_idx = np.full(ncell, -1, dtype=np.int64)
    best_cap = np.zeros(ncell)
    for i in range(n):
        if phase[i] != 1 or A[i] <= 0.0 or M[i] <= 0.0:
            continue
        c = cidx[i]
        cap = M[i] / M_SPLINTER - A[i]      # how many extra splinter-sized crystals this SD can host
        if cap > best_cap[c]:
            best_cap[c] = cap
            best_idx[c] = i
    for c in range(ncell):
        r = rimed[c]
        if r <= 0.0:
            continue
        f = hm_factor(T_c[c])
        if f <= 0.0:
            continue
        i = best_idx[c]
        if i < 0:
            continue
        add = HM_PER_KG * r * f
        if add > best_cap[c]:              # cap: don't dilute below one splinter per crystal
            add = best_cap[c]
        if add <= 0.0:
            continue
        A[i] += add                        # number multiplication; M[i] unchanged -> mass conserved
        splinters_out[c] += add
