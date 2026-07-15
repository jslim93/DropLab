"""Toy non-inductive electrification + lightning for the 2D dynamic model.

OPT-IN DIAGNOSTIC PASS. Nothing here runs unless `electrification=True` is passed to
`run_flow2d_dynamic`; with it off the `charge` array is None and the simulation is
bit-for-bit identical (the golden gate). Charge exerts NO force on the droplets or the
dynamics — it is a read-only diagnostic of the microphysics the model already computes.

Physics (see docs/ELECTRIFICATION_DESIGN.md):
  * Non-inductive charging on riming graupel-crystal rebounds. Saunders single
    charge-reversal temperature -> a clean dipole: below `q_rev_T` graupel charges
    negative (and the lofted crystals positive), above it the reverse.
  * Charge is a per-super-droplet scalar. Graupel sediments, crystals loft, so the
    vertical dipole forms FROM the existing Lagrangian transport -- nothing extra.
  * Electric field by reusing the streamfunction Poisson solver: lap(phi) = -rho_q/eps0,
    E = -grad phi. Breakdown where |E| > E_crit(z) triggers a discharge flash.

Two SELF-CONSISTENCY invariants (NOT evidence of realism -- any conserving scheme passes
them; see docs/ELECTRIFICATION_AUDIT.md):
  * Deposition is charge-conserving (each event adds +Q and -Q) -> domain net charge = 0.
  * Flash neutralisation pulls channel charges toward their local mean, which preserves
    their sum exactly -> domain net charge stays 0 through a flash too.

REALISM CAVEAT: this is a physically ILLUSTRATIVE toy, not predictive. The charging
magnitude (charge_coeff) is a tuned knob with no anchor in laboratory per-collision charge;
graupel-vs-crystal is a mass-rank proxy (one ice category); the field uses a grounded-box BC
in 2-D (absolute |E| is not physical); and the discharge traces a STATIC potential field
(it is not re-solved as an equipotential leader). Quantitative results track the knobs, not
storm physics. Full assessment + citations: docs/ELECTRIFICATION_AUDIT.md.

References (for later verification):
  Takahashi, T. (1978). Riming electrification as a charge generation mechanism in
    thunderstorms. J. Atmos. Sci. 35(8), 1536-1548.
  Reynolds, S. E., Brook, M. & Gourley, M. F. (1957). Thunderstorm charge separation.
    J. Meteorol. 14(5), 426-436.
  Saunders, C. P. R. & Peck, S. L. (1998). Laboratory studies of the influence of the
    rime accretion rate on charge transfer during crystal/graupel collisions.
    J. Geophys. Res. 103(D12), 13949-13956.
  Kasemir, H. W. (1960). A contribution to the electrostatic theory of a lightning
    discharge. J. Geophys. Res. 65(7), 1873-1878.  (bidirectional, net-neutral leader)
  Niemeyer, L., Pietronero, L. & Wiesmann, H. J. (1984). Fractal dimension of dielectric
    breakdown. Phys. Rev. Lett. 52(12), 1033-1036.  (the DBM discharge)
  Mansell, E. R., MacGorman, D. R., Ziegler, C. L. & Straka, J. M. (2002). Simulated
    three-dimensional branched lightning in a numerical thunderstorm model.
    J. Geophys. Res. 107(D9), 4075.  (DBM in a storm model + charge neutralisation)

Scope: a single ice category (graupel vs crystal = mass-rank proxy), a single charge-
reversal temperature (dipole, no tripole), 2-D, diagnostic only. NOT a validated
electrification scheme -- an instructive, equation-based toy. See the design doc.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from droplab.poisson import solve_poisson, solve_poisson_periodic_x
from droplab.parameters import rho_ice, pi

EPS0 = 8.8541878128e-12          # vacuum permittivity [F/m]
T_FREEZE = 273.15                # K

# --- non-inductive charging constants (physical, from the lab/field literature) ---------
DQ_NI = 5.0e-15        # characteristic charge separated per rebounding graupel-crystal
                       # collision [C] ~ 5 fC. Lab range 1-100+ fC (Takahashi 1978;
                       # Saunders & Peck 1998; Saunders, Keith & Mitzeva 1991). A single
                       # representative magnitude (NOT a knob tuned to make flashes).
EPS_SEP = 0.1          # fraction of graupel-crystal collisions that REBOUND and separate
                       # charge (vs aggregate). O(0.1); the one efficiency parameter.
D_GRAUPEL = 2.0e-4     # graupel/crystal size boundary [m] (0.2 mm): ice larger than this
                       # is the dense rimer (graupel), smaller is a vapour-grown crystal.
E_COLL_GC = 1.0        # graupel-crystal collision efficiency (~1 for large graupel)


def _ice_fallspeed(D):
    """Terminal fall speed of an ice particle of diameter D [m] -> [m/s]. Lump-graupel /
    snow power law v = 124 D^0.66 (Locatelli & Hobbs 1974); graupel (large D) falls faster
    than crystals (small D), so their differential velocity drives the rebounding
    collisions that separate charge."""
    return 124.0 * D ** 0.66


# --------------------------------------------------------------------------------------
# charging law: Saunders & Peck (1998) critical rime-accretion-rate (RAR) curve
# --------------------------------------------------------------------------------------
# Table 1 of Saunders & Peck (1998), J. Geophys. Res. 103(D12), 13949-13956: a 6th-order
# polynomial fit (in cloud temperature T, degC) to their Figure 6 -- the laboratory-measured
# boundary in (RAR, T) space between positive and negative graupel charging. RAR = EW*V
# [g m^-2 s^-1] is the rime accretion rate (EW = effective liquid water content swept by the
# graupel [g m^-3], V = graupel/crystal relative fall speed [m/s]); RAR at/above the curve ->
# graupel charges POSITIVE, below it -> NEGATIVE (p. 13955: "graupel to charge negatively at
# temperatures as high as -2.3 degC at values of RAR below around 1 g m^-2 s^-1; at higher
# values of RAR the rimer charges positively"). Coefficients transcribed directly from Table 1.
_RAR_CRIT_COEFFS = (1.0, 7.9262e-2, 4.4847e-2, 7.4754e-3, 5.4686e-4, 1.6737e-5, 1.7613e-7)
_RAR_CRIT_T_LO, _RAR_CRIT_T_HI = -36.2, -2.3   # degC, the experimental range (Saunders & Peck 1998)


def critical_rar(T_celsius):
    """Critical rime accretion rate [g m^-2 s^-1] for graupel charge-sign reversal at
    cloud temperature T_celsius (degC): Saunders & Peck (1998) Table 1 polynomial fit to
    their Figure 6. Valid only for -36.2 <= T <= -2.3 degC (the paper: "this equation is
    valid only in the temperature range investigated") -- T is clamped to that range
    before evaluation, and the result is floored at 0 (RAR cannot be negative; the raw
    6th-order fit dips slightly negative right at the cold edge of the fitted range)."""
    Tc = np.clip(T_celsius, _RAR_CRIT_T_LO, _RAR_CRIT_T_HI)
    rar = sum(c * Tc ** k for k, c in enumerate(_RAR_CRIT_COEFFS))
    return np.maximum(rar, 0.0)


def graupel_charge_sign(T, q_rev_T, RAR=None):
    """Sign of the charge the GRAUPEL (heavy, riming) ice acquires. The lighter
    rebounding crystals take the opposite sign.

    When `RAR` [g m^-2 s^-1] (rime accretion rate = EW*V) is given, uses the literature
    Saunders & Peck (1998) critical-RAR(T) curve (see critical_rar()): RAR at/above the
    curve -> positive, below it -> negative. This is the real (RAR, T)-dependent boundary,
    not a single reversal temperature.

    Falls back to a single fixed reversal temperature q_rev_T (below it negative, above
    it positive) when RAR is None -- e.g. no local effective-liquid-water estimate is
    available at the call site. This fallback is the older, coarser convention."""
    if RAR is not None:
        return 1.0 if RAR >= critical_rar(T - 273.15) else -1.0
    return -1.0 if T < q_rev_T else 1.0


# --------------------------------------------------------------------------------------
# charge separation (the charging pass)
# --------------------------------------------------------------------------------------
def deposit_charge(charge, M, A, phase, cidx, Nc, T_flat, qsc_flat,
                   q_sc_min, q_rev_T, dt, V_cell, Nx, Nz, dz, charge_eff=EPS_SEP,
                   sweep_depth=300.0, T_lo=243.15, T_hi=271.15, hab=None,
                   rho_air_flat=None):
    """Separate charge by NON-INDUCTIVE graupel-crystal collisions, SWEEP-based.

    Real non-inductive charging is a sweep-out: graupel falls THROUGH a population of ice
    crystals, colliding and rebounding along its fall path. The charging rate is therefore
    a function of the graupel and crystal number DENSITIES in a region, not of two discrete
    super-droplets landing in the exact same grid cell (Helsdon & Farley 1987; Ziegler et
    al. 1991; Mansell et al. 2005). Requiring exact co-location is a sparse-super-droplet
    artifact -- and it fails entirely once organized (sheared) flow displaces graupel from
    crystals, even though graupel is still falling through crystals one cell below.

    So: classify ice by a physical size (single-particle D > D_GRAUPEL = graupel, else
    crystal). Bin graupel and crystal number densities to the grid and smooth each in the
    VERTICAL only over `sweep_depth` (~the depth a graupel sweeps crystals as it falls;
    NOT a free knob -- the fall-through interaction scale). A graupel super-droplet then
    charges in proportion to the crystal density it sweeps,

        dq_graupel(i)  =  -s * DQ_NI * charge_eff * dt * K * A_i * n_crystal_sweep(cell_i)

    (K = gravitational kernel from representative graupel/crystal sizes & fall speeds; s the
    charge-reversal sign), and crystals receive the equal-and-opposite total, distributed by
    the graupel density they were swept by -> the domain net stays exactly zero. Vertical-
    only smoothing is physical: graupel sweeps crystals directly below it; crystals displaced
    horizontally (by shear) are NOT swept, so shear-separated regions correctly do not charge.

    Magnitude physically anchored (collision rate x lab per-collision charge DQ_NI~5 fC).
    Kept simplifications (labelled): single ice category sized into graupel/crystal,
    representative kernel. Mutates `charge` in place.

    `rho_air_flat`, when given (Nc,), lets the charge-SIGN law use the real Saunders &
    Peck (1998) critical-rime-accretion-rate curve (see graupel_charge_sign()) instead of
    a single fixed reversal temperature: the representative supercooled liquid water
    content EW [g/m^3] = qsc * rho_air is combined with the already-computed graupel/
    crystal relative fall speed into RAR = EW*V. Falls back to the single-q_rev_T rule
    when rho_air_flat is None (no local air-density estimate at the call site).

    `hab`, when given (habit=True runs), is the (N,3)=[a_axis,c_axis,rho_app] shape state
    from droplab.ice_habit: rho_app is the particle's ACTUAL apparent density, which
    riming/deposition growth can push well away from bulk solid ice (rho_ice) -- fluffy,
    low-density crystals/aggregates are geometrically LARGER than a fixed-rho_ice sphere of
    the same mass would suggest, so using rho_ice for ALL ice mis-sizes the graupel/crystal
    split. Falls back to rho_ice for any particle with no shape yet (hab[:,2]<=0) or when
    hab is None (habit=False, or the pre-habit call sites)."""
    is_ice = (phase == 1) & (A > 0) & (M > 0)
    if not is_ice.any():
        return charge
    rho_class = np.full_like(M, rho_ice)
    if hab is not None:
        has_shape = is_ice & (hab[:, 2] > 0.0)
        rho_class[has_shape] = hab[has_shape, 2]
    r = np.zeros_like(M)
    r[is_ice] = (M[is_ice] / (A[is_ice] * (4.0 / 3.0) * pi * rho_class[is_ice])) ** (1.0 / 3.0)
    is_g = is_ice & (2.0 * r > D_GRAUPEL)
    is_c = is_ice & (2.0 * r <= D_GRAUPEL) & (r > 0.0)
    if not is_g.any() or not is_c.any():
        return charge
    # number densities per cell, then vertical-only sweep smoothing (graupel falls through)
    ng = np.zeros(Nc); np.add.at(ng, cidx[is_g], A[is_g]); ng /= V_cell
    nc = np.zeros(Nc); np.add.at(nc, cidx[is_c], A[is_c]); nc /= V_cell
    sig = max(sweep_depth / dz, 0.5)
    ng_sw = gaussian_filter1d(ng.reshape(Nx, Nz), sig, axis=1, mode="nearest").ravel()
    nc_sw = gaussian_filter1d(nc.reshape(Nx, Nz), sig, axis=1, mode="nearest").ravel()
    active = (qsc_flat > q_sc_min) & (T_flat > T_lo) & (T_flat < T_hi)   # charging zone
    if not active.any():
        return charge
    # representative collection kernel from domain-mean graupel/crystal sizes
    rg = float(np.average(r[is_g], weights=A[is_g])); rc = float(np.average(r[is_c], weights=A[is_c]))
    v_rel = abs(_ice_fallspeed(2.0 * rg) - _ice_fallspeed(2.0 * rc))   # graupel/crystal relative speed
    K = pi * (rg + rc) ** 2 * v_rel * E_COLL_GC
    coef = DQ_NI * charge_eff * dt * K
    w_active = (ng + nc)[active] + 1e-30
    T_rep = float(np.average(T_flat[active], weights=w_active))
    RAR = None
    if rho_air_flat is not None:
        EW = float(np.average(qsc_flat[active] * rho_air_flat[active], weights=w_active)) * 1e3
        RAR = EW * v_rel                                # g m^-2 s^-1 (Saunders & Peck 1998)
    s = graupel_charge_sign(T_rep, q_rev_T, RAR=RAR)
    g_act = np.flatnonzero(is_g & active[cidx])        # graupel in the charging zone
    c_all = np.flatnonzero(is_c)
    if g_act.size == 0:
        return charge
    qg = coef * A[g_act] * nc_sw[cidx[g_act]]          # graupel sweeps nearby crystals
    # crystals receive the equal-and-opposite total, distributed over ALL crystals that were
    # swept by graupel (ng_sw>0) -- including any in adjacent non-active cells -- so the
    # balance is exact regardless of the active gate (charge conserved to machine precision)
    wc = A[c_all] * ng_sw[cidx[c_all]]
    if wc.sum() <= 0.0:
        return charge                                  # no crystals were swept -> no charging
    charge[g_act] += s * qg                            # graupel gets sign s, total s*sum(qg)
    charge[c_all] += -s * float(qg.sum()) * (wc / wc.sum())   # -s*sum(qg) onto crystals
    return charge


# --------------------------------------------------------------------------------------
# field: charge density -> potential -> E
# --------------------------------------------------------------------------------------
def charge_density(charge, cidx, Nx, Nz, cell_volume):
    """Grid the per-super-droplet charge to a cell-centered charge density [C/m^3]."""
    rho = np.zeros(Nx * Nz)
    np.add.at(rho, cidx, charge)
    return (rho / cell_volume).reshape(Nx, Nz)


def solve_potential(rho_q, dx, dz, periodic_x):
    """Electric potential from Gauss's law lap(phi) = -rho_q/eps0, reusing the
    streamfunction Poisson solver (phi = 0 on the grounded walls / z-lids). The solver
    returns psi with lap(psi) = rhs, so we pass rhs = -rho_q/eps0."""
    rhs = -rho_q / EPS0
    return solve_poisson_periodic_x(rhs, dx, dz) if periodic_x else solve_poisson(rhs, dx, dz)


def _ddx(f, dx, periodic_x):
    if periodic_x:
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)
    d = np.zeros_like(f)
    d[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dx)
    d[0, :] = (f[1, :] - f[0, :]) / dx
    d[-1, :] = (f[-1, :] - f[-2, :]) / dx
    return d


def _ddz(f, dz):
    d = np.zeros_like(f)
    d[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dz)
    d[:, 0] = (f[:, 1] - f[:, 0]) / dz
    d[:, -1] = (f[:, -1] - f[:, -2]) / dz
    return d


def efield(phi, dx, dz, periodic_x):
    """E = -grad(phi). Returns (Ex, Ez, |E|), all (Nx, Nz)."""
    Ex = -_ddx(phi, dx, periodic_x)
    Ez = -_ddz(phi, dz)
    return Ex, Ez, np.hypot(Ex, Ez)


# --------------------------------------------------------------------------------------
# breakdown + discharge flash
# --------------------------------------------------------------------------------------
def breakdown_field(rho_air, rho0, E_breakdown):
    """Altitude-reduced breakdown threshold E_crit(z) = E_breakdown * rho(z)/rho0
    (runaway breakdown falls with air density). rho_air is (Nx, Nz) or (Nz,)."""
    return E_breakdown * (rho_air / rho0)


def dbm_leader(phi, init_ij, dx, dz, rng, eta=3.0, max_cells=180, ground_j=0):
    """Stochastic dielectric-breakdown leader on the grid.

    Implements the dielectric-breakdown model (DBM) of Niemeyer, Pietronero & Wiesmann
    (1984): the discharge is a connected channel that grows one cell at a time from the
    initiation cell, where each candidate boundary cell i is added with probability

        P_i  =  |phi_i - phi_0|^eta  /  Sum_j |phi_j - phi_0|^eta

    over the current boundary cells j. Larger eta -> more filamentary / less space-filling
    (eta=1 is bushy, D~1.7 in 2D; the default eta=3 here is deliberately filamentary so the
    channel reads as a forked bolt rather than a Lichtenberg blob). Growth stops at the
    ground row or `max_cells`. 8-connected so the tree can branch diagonally.

    SIMPLIFICATION vs a real leader (Mansell et al. 2002): phi_0 is FROZEN at the
    initiation cell and the potential field is NEVER re-solved as the channel grows. So
    this traces the STATIC pre-existing potential gradient -- it does not dynamically
    screen the field the way a conducting equipotential leader does. See
    docs/ELECTRIFICATION_AUDIT.md. Consequences: intracloud dominance here is partly a
    numerical artifact (max_cells cap + greedy eta + grounded-box BC), not a physically
    earned IC:CG ratio.

    References:
      Niemeyer, L., Pietronero, L. & Wiesmann, H. J. (1984), "Fractal dimension of
        dielectric breakdown", Phys. Rev. Lett. 52(12), 1033-1036.
      Mansell, E. R., MacGorman, D. R., Ziegler, C. L. & Straka, J. M. (2002),
        "Simulated three-dimensional branched lightning in a numerical thunderstorm
        model", J. Geophys. Res. 107(D9), 4075.

    Returns (cells, edges, reached_ground): the list of channel cells (i,j), the list of
    (parent_ij, child_ij) tree edges, and whether it reached the ground."""
    Nx, Nz = phi.shape
    i0, j0 = init_ij
    phi0 = float(phi[i0, j0])
    in_ch = np.zeros((Nx, Nz), bool)
    in_ch[i0, j0] = True
    cells = [(i0, j0)]
    edges = []
    nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    cand = {}                                   # candidate cell -> parent cell

    def _add_candidates(ci, cj):
        for di, dj in nbrs:
            ni = (ci + di) % Nx                  # periodic in x
            nj = cj + dj
            if nj < 0 or nj >= Nz:               # clamped in z
                continue
            if in_ch[ni, nj] or (ni, nj) in cand:
                continue
            cand[(ni, nj)] = (ci, cj)

    _add_candidates(i0, j0)
    reached_ground = (j0 == ground_j)
    while cand and len(cells) < max_cells and not reached_ground:
        keys = list(cand.keys())
        w = np.array([abs(phi[i, j] - phi0) for (i, j) in keys]) ** eta
        s = w.sum()
        p = w / s if s > 0 else np.full(len(keys), 1.0 / len(keys))
        ci, cj = keys[int(rng.choice(len(keys), p=p))]
        parent = cand.pop((ci, cj))
        in_ch[ci, cj] = True
        cells.append((ci, cj))
        edges.append((parent, (ci, cj)))
        _add_candidates(ci, cj)
        if cj == ground_j:
            reached_ground = True
    return cells, edges, reached_ground


def flash(charge, cidx, Nx, Nz, phi, Emag, rho_air, rho0, dx, dz, rng,
          E_breakdown=1.5e5, eta=3.0, max_cells=180, flash_neutralize=0.7,
          flash_radius=2):
    """If |E| exceeds the local breakdown threshold somewhere, fire ONE discharge.

    Initiation is at the cell of maximum field (the most probable breakdown point); the
    channel then propagates as a stochastic dielectric-breakdown leader (`dbm_leader`)
    along the potential field, branching and seeking the ground. Super-droplets within
    `flash_radius` grid cells of the channel are neutralised by pulling their charge
    toward that region's mean, c -> (1-f)c + f*mean(c) -- a real leader drains the charge
    reservoir in a VOLUME around the channel (Mansell et al. 2002), not just the 1-cell
    line, and this is what collapses the field after a flash so the storm must recharge
    before the next one. The pull preserves the sum EXACTLY, so the domain net charge is
    unchanged (the bidirectional-leader idea of Kasemir 1960: a leader carries no net
    charge, it only redistributes it). Mutates `charge` in place.

    Returns {segments, cells, q_neutralized, Emax, grounded} or None if no breakdown.
    `segments` is an (n,4) array of (x0,z0,x1,z1) channel edges in metres for drawing."""
    Ecrit = np.broadcast_to(breakdown_field(rho_air, rho0, E_breakdown), Emag.shape)
    Emax = float(Emag.max())
    if not np.any(Emag > Ecrit):                 # no breakdown anywhere -> no flash
        return None

    init_ij = np.unravel_index(int(np.argmax(Emag)), Emag.shape)
    cells, edges, grounded = dbm_leader(phi, init_ij, dx, dz, rng, eta=eta,
                                        max_cells=max_cells)
    # dilate the channel by flash_radius cells (periodic x, clamped z): the leader's
    # neutralisation volume
    in_ch = np.zeros((Nx, Nz), dtype=bool)
    for (i, j) in cells:
        in_ch[i, j] = True
    r = int(max(flash_radius, 0))
    if r > 0:
        dil = in_ch.copy()
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di == 0 and dj == 0:
                    continue
                sh = np.roll(in_ch, di, axis=0)          # periodic in x
                if dj > 0:
                    dil[:, dj:] |= sh[:, :-dj]
                elif dj < 0:
                    dil[:, :dj] |= sh[:, -dj:]
                else:
                    dil |= sh
        in_ch = dil
    chan_flat = np.flatnonzero(in_ch.ravel())
    near = np.isin(cidx, chan_flat)              # super-droplets in the discharge volume
    q_neutralized = 0.0
    if near.any():
        cn = charge[near]
        mean = cn.mean()
        q_neutralized = float(flash_neutralize * np.abs(cn - mean).sum())
        charge[near] = (1.0 - flash_neutralize) * cn + flash_neutralize * mean
    segments = np.array([[(pi + 0.5) * dx, (pj + 0.5) * dz, (ci + 0.5) * dx, (cj + 0.5) * dz]
                         for (pi, pj), (ci, cj) in edges]) if edges else np.empty((0, 4))
    return {"segments": segments, "cells": cells, "q_neutralized": q_neutralized,
            "Emax": Emax, "grounded": grounded}
