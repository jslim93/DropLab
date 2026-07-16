"""Per-grid-box Linear Eddy Model coupling. The line/box structure is SAM6.10.10.LCM_JS
(SRC/MICRO_LAGRANGE/micro_sgs_mixing.f90 + micro_sgs_uvw.f90); the broadening source is the
box model's adiabatic supersaturation fluctuation (particle_model/mod_LEM.f90,
switch_supersat_fluct_LEM=.TRUE.).

Design (the authoritative SAM-LCM reference): ONE super-droplet IS one LEM grid box. The SDs
inside a resolved grid cell form the 1-D vertical LEM line, spaced dz_sgs = dz_cell / n_SD
(micro_sgs_mixing.f90 L64). Each SD carries a PROGNOSTIC supersaturation eta_sd (SAM's iss)
with MEMORY: nudged toward the grid-box mean on a slow tau ~ 900 s (L228-237), NOT set to it
instantly. Within a box the eta line is homogenised by eddy diffusion + triplet-map
rearrangement, both SUBSTEP-STABILISED (dt_diff = 0.2 dz_sgs^2 / D_mol, L73-79). Each SD then
condenses against its OWN eta_sd, and the SDs are transported by the resolved flow plus a
per-SD AR-1 subgrid velocity Wsgs (micro_sgs_uvw.f90) whose variance scales with turbulence.

Broadening source: the triplet rearrangement displaces fluid elements (k-m) cells up/down the
line and applies the adiabatic supersaturation change (k-m)*dz_sgs*ds/dz -- the box model's
switch_supersat_fluct_LEM. This is SELF-CONTAINED (no resolved gradient needed), which is how
the 0-D parcel model broadens; SAM turns it off and relies on resolved transport, but at the
coarse DropLab grid the self-contained source is the one that works. The effect is cumulative:
each droplet's supersaturation random-walks (~1%), so Var(r^2) grows ~linearly in time and the
spectrum broadens with cloud depth/age.

OPT-IN: lem=False -> these arrays are None and nothing runs -> bit-identical (golden gate).
"""
import math
import numpy as np
from numba import njit

from droplab.parameters import muelq, l_v, g, cp, rv


def init_lem_state(n_super):
    """Per-SD prognostic supersaturation (NaN = uninitialised -> set to the SD's cell mean
    on first touch) and the AR-1 subgrid vertical velocity. Mirrors particle_var(:,ieta)
    and particle_var(:,iWsgs)."""
    eta_sd = np.full(n_super, np.nan)
    w_sgs = np.zeros(n_super)
    return eta_sd, w_sgs


@njit(cache=True)
def _rearrange_numba(eta, s0, n_last, dz_sgs, eta_k, L_turb, dz_cell, dsdz):
    """One triplet-map rearrangement of the cell line eta[s0:s0+n_last] (cyclic). Compresses
    the chosen segment 3x and lays down three copies, middle reversed (SAM micro_sgs_mixing
    L272-395). If dsdz>0, a fluid element displaced (k-m) cells up cools adiabatically and its
    supersaturation rises by (k-m)*dz_sgs*dsdz -- the box model's switch_supersat_fluct_LEM,
    the self-contained broadening source. (k-m) sums to 0 over the segment -> mean preserved."""
    if n_last <= 6 or L_turb < eta_k:
        return
    n_start = int(np.random.random() * n_last)
    if L_turb / eta_k < 1.01:
        length = eta_k
    else:
        hi = min(L_turb, dz_cell)
        length = eta_k
        for _ in range(10000):
            length = (eta_k ** (-5.0 / 3.0) - np.random.random()
                      * (eta_k ** (-5.0 / 3.0) - L_turb ** (-5.0 / 3.0))) ** (-0.6)
            if eta_k <= length <= hi:
                break
    n_length = int(math.floor(length / dz_sgs))
    n_length = int(round(n_length / 3.0)) * 3
    if n_length < 6:
        n_length = 6
    hi2 = int(math.floor(n_last / 3.0)) * 3
    if n_length > hi2:
        n_length = hi2
    if n_length < 6:
        return
    seg = np.empty(n_length)                              # copy so reads are unaffected by writes
    for k in range(n_length):
        seg[k] = eta[s0 + (n_start + k) % n_last]
    n1 = int(round(n_length / 3.0))
    n2 = int(round(2.0 * n_length / 3.0))
    for k in range(n_length):
        if k < n1:
            mm = 3 * k
        elif k < n2:
            mm = 2 * (n_length - 1) - 3 * k
        else:
            mm = 3 * k - 2 * (n_length - 1)
        val = seg[mm]
        if dsdz != 0.0:
            val += (k - mm) * dz_sgs * dsdz
        eta[s0 + (n_start + k) % n_last] = val


@njit(cache=True)
def _mix_kernel(eta, starts, counts, ssg, Tg, epsg, dz_cell, dt, supersat_fluct, min_sd,
                rho_air, s_active):
    """Per-cell SAM-LCM mixing (micro_sgs_mixing L63-196), serial over cell-sorted groups.
    For each CLOUDY cell (mean supersaturation >= s_active) with > min_sd SDs: substep-stable
    cyclic FTCS diffusion of the eta line plus triplet-map rearrangements. Clear/subsaturated
    cells are skipped (no growing droplets there, so no broadening to resolve) -- this is the
    dominant cost saving, since most of the domain is clear air."""
    for gi in range(starts.shape[0]):
        n_last = counts[gi]
        if n_last <= min_sd or ssg[gi] < s_active:       # skip small or clear cells
            continue
        s0 = starts[gi]
        eps = epsg[gi]                                    # per-cell dissipation rate
        dz_sgs = dz_cell / n_last
        eta_k = 6.0 * dz_sgs
        L_turb = dz_cell                                 # SAM L67: max(min(smix, dz_LCM), 1e-3)
        if L_turb < eta_k:
            continue
        D_turb = 0.1 * L_turb ** (4.0 / 3.0) * eps ** (1.0 / 3.0)
        D_mol = D_turb * (eta_k / L_turb) ** (4.0 / 3.0)
        if D_mol < muelq / rho_air:
            D_mol = muelq / rho_air
        lam = 54.0 / 5.0 * D_turb / L_turb ** 3 * (L_turb / eta_k) ** (5.0 / 3.0)
        dt_diff = 0.2 * dz_sgs ** 2 / D_mol
        n_tsteps = int(math.ceil(dt / dt_diff))
        if n_tsteps < 1:
            n_tsteps = 1
        dt_diff = dt / n_tsteps
        dt_turb_inv = lam * dz_cell
        n_rsteps = int(math.ceil(dt_diff * dt_turb_inv))
        if n_rsteps < 1:
            n_rsteps = 1
        if supersat_fluct:
            dsdz = (ssg[gi] + 1.0) * l_v * g / (rv * Tg[gi] ** 2 * cp)
        else:
            dsdz = 0.0
        c = dt_diff * D_mol / dz_sgs ** 2                # FTCS coefficient (<= 0.2, stable)
        tmp = np.empty(n_last)
        p_accept = dt_diff * dt_turb_inv / n_rsteps
        for _ in range(n_tsteps):
            for i in range(n_last):                      # cyclic FTCS diffusion
                ip = s0 + (i + 1) % n_last
                im = s0 + (i - 1) % n_last
                tmp[i] = eta[s0 + i] + c * (eta[ip] - 2.0 * eta[s0 + i] + eta[im])
            for i in range(n_last):
                eta[s0 + i] = tmp[i]
            for _ in range(n_rsteps):
                if p_accept > np.random.random():
                    _rearrange_numba(eta, s0, n_last, dz_sgs, eta_k, L_turb, dz_cell, dsdz)


def nudge_and_mix(eta_sd, w_sgs, cidx, supersat_flat, T_flat, n_cells, dz_cell, eps,
                  tau_ndg, dt, rng, rho_air=1.0, min_sd=10, s_max=0.02, supersat_fluct=True,
                  s_active=-0.02):
    """Advance the per-SD prognostic supersaturation one resolved step: (1) lazy-init new SDs
    to their cell mean, (2) nudge every SD toward its cell mean on tau_ndg (memory) and bound
    the subgrid anomaly to +-s_max, (3) per cell with > min_sd SDs, homogenise eta by
    substep-stable diffusion + triplet rearrangement with the adiabatic supersaturation
    fluctuation (the broadening source). Mutates eta_sd in place.

    s_max bounds |eta_sd - cell mean| to a physical subgrid value (~1-2%): at true LES
    resolution SAM keeps this small implicitly; at the coarse DropLab grid a cell can
    transition clear<->cloud faster than the SD nudges, so we cap it (documented)."""
    m = np.isnan(eta_sd)
    if m.any():
        eta_sd[m] = supersat_flat[cidx[m]]
    eta_sd -= (eta_sd - supersat_flat[cidx]) * (dt / tau_ndg)        # nudge (memory)
    sc = supersat_flat[cidx]
    np.clip(eta_sd, sc - s_max, sc + s_max, out=eta_sd)             # bound subgrid anomaly

    eps_arr = np.full(n_cells, float(eps)) if np.ndim(eps) == 0 else np.asarray(eps, float)
    order = np.argsort(cidx, kind="stable")
    cs = cidx[order]
    uniq, starts = np.unique(cs, return_index=True)                 # cs sorted -> contiguous groups
    counts = np.diff(np.append(starts, cs.shape[0]))
    eta_sorted = np.ascontiguousarray(eta_sd[order])
    _mix_kernel(eta_sorted, starts.astype(np.int64), counts.astype(np.int64),
                supersat_flat[uniq], T_flat[uniq], eps_arr[uniq], float(dz_cell), float(dt),
                bool(supersat_fluct), int(min_sd), float(rho_air), float(s_active))
    eta_sd[order] = eta_sorted


def sgs_velocity_step(w_sgs, eps, dz_cell, dt, rng):
    """Per-SD AR-1 (Langevin) subgrid vertical velocity (SAM micro_sgs_uvw.f90 L26-47).
    sigma = sqrt(2/3 tke), tke = (D_turb/(0.1 L))^2 (Deardorff), RL = exp(-dt/(tk/tke)).
    Returns the updated w_sgs and the per-SD vertical displacement w_sgs*dt [m]. eps may be a
    scalar (prescribed) or a per-SD array (strain-derived from the Smagorinsky closure), so
    the subgrid velocity -- and the broadening -- scale with the LOCAL resolved turbulence."""
    L = max(dz_cell, 1.0e-3)
    D_turb = 0.1 * L ** (4.0 / 3.0) * np.asarray(eps, float) ** (1.0 / 3.0)
    tke = (D_turb / (0.1 * L)) ** 2
    sigma = np.sqrt(2.0 / 3.0 * tke)
    RL = np.exp(-dt / (D_turb / np.maximum(tke, 1.0e-30)))
    w_new = RL * w_sgs + np.sqrt(1.0 - RL ** 2) * sigma * rng.standard_normal(w_sgs.shape)
    w_new = np.where(tke < 1.0e-4, 0.0, w_new)                       # laminar cells: no SGS wind
    w_sgs[:] = w_new
    return w_sgs, w_sgs * dt
