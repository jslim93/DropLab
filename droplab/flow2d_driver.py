"""2D kinematic cumulus driver: Lagrangian super-droplets two-way coupled to a
2D Eulerian (theta, q_v) field advected by a prescribed non-divergent flow.

Loop per step:
  1. advect droplet positions in the flow (RK2),
  2. advect theta, q_v with MPDATA,
  3. condense: each droplet grows in ITS cell's (T, q, P); the condensed mass is
     scattered back as a vapor sink + latent-heat source for that cell,
  4. (optional) grid-cell-local LSM collision,
  5. collect diagnostics.

Reuses the validated growth physics (`radius_liquid_euler`) per droplet and the
validated `collide_soa` per cell — this module only adds the 2D bookkeeping.
"""
import numpy as np
from numba import njit, vectorize, prange

from droplab.parameters import (p0, r_a, cp, rv, l_v, g, rho_liq, rho_ice, rho_aero, pi,
                              vanthoff_aero, molecular_weight_water,
                              molecular_weight_aero, activation_radius_ts,
                              seperation_radius_ts)
from droplab.condensation import esatw, sigma_air_liq, radius_liquid_euler
from droplab.collision import ws_drops_beard
from droplab.aero_init import r_equi, r_equi_arr
from droplab.flow2d import Flow2D
from droplab.mpdata import mpdata_step
from droplab.collision_soa import collide_soa, seed_numba_rng, _collision_kernel, _sm64_next

# Compiled ufuncs over the grid — same polynomials as the scalar @njit esatw /
# sigma_air_liq (bit-identical), but ~100x faster than np.vectorize (a Python
# loop). This is the hot path: called for every cell, every condensation substep.
@vectorize(["float64(float64)"], cache=True)
def _esatw_v(T):
    dT = T - 273.15
    result = (6.11239921 + dT * (0.443987641 + dT * (0.142986287e-1 + dT * (
        0.264847430e-3 + dT * (0.302950461e-5 + dT * (0.206739458e-7 + dT * (
            0.640689451e-10 + dT * (-0.952447341e-13 + -0.976195544e-15 * dT))))))))
    return result * 100.0


@vectorize(["float64(float64)"], cache=True)
def _sigma_v(tabs):
    tc = tabs - 273.15
    result = (75.93 + 0.115 * tc + 6.818e-2 * tc ** 2 + 6.511e-3 * tc ** 3
              + 2.933e-4 * tc ** 4 + 6.283e-6 * tc ** 5 + 5.285e-8 * tc ** 6)
    return result * 1.0e-3


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------
def _rh_profile(zc, RH0, z_bl, RH_top, dz_trans=250.0):
    """Cumulus moisture stratification: a moist boundary layer (RH0) below z_bl
    smoothly capped by drier free-tropospheric air (RH_top) above it."""
    return RH_top + (RH0 - RH_top) * 0.5 * (1.0 - np.tanh((zc - z_bl) / dz_trans))


def _base_state(flow, T0, P0, RH0, z_bl=600.0, RH_top=0.2, z_inv=None,
                dtheta_inv=0.0, gamma_theta=0.0, sounding=None):
    """Hydrostatic base state, returning (theta, qv, P_col, T_col, theta0, RH_col).

    With `sounding` (a dict of z/theta/qv from droplab.soundings, e.g. BOMEX) the
    profiles are interpolated from that REAL sounding — the cloud is then capped
    by the observed inversion. Otherwise theta is well-mixed in the boundary layer
    and stably stratified above (gamma_theta, plus an optional z_inv/dtheta_inv
    jump), and q_v follows the moist-BL/dry-aloft RH profile. P(z) integrates the
    hydrostatic relation for the height-varying theta."""
    kap = r_a / cp
    theta0 = T0 * (p0 / P0) ** kap
    zc = (np.arange(flow.Nz) + 0.5) * flow.dz

    if sounding is not None:
        theta_col = np.interp(zc, sounding["z"], sounding["theta"])
        qv_col = np.interp(zc, sounding["z"], sounding["qv"]) * 1e-3   # g/kg -> kg/kg
        theta0 = float(theta_col[0])
    else:
        theta_col = theta0 + gamma_theta * np.maximum(zc - z_bl, 0.0)
        if z_inv is not None and dtheta_inv > 0.0:
            theta_col = theta_col + dtheta_inv * np.clip((zc - z_inv) / 200.0, 0.0, 1.0)

    # Exner from hydrostatic integration: dpi/dz = -g/(cp*theta)
    inv_theta = 1.0 / theta_col
    exner = (P0 / p0) ** kap - g / cp * (np.cumsum(inv_theta) * flow.dz
                                         - 0.5 * inv_theta * flow.dz)
    P_col = p0 * exner ** (1.0 / kap)
    T_col = theta_col * exner
    es_col = _esatw_v(T_col)
    if sounding is None:
        RH_col = _rh_profile(zc, RH0, z_bl, RH_top)
        qv_col = RH_col * es_col / (P_col - RH_col * es_col) * r_a / rv
    else:
        e_a = qv_col * P_col / (qv_col + r_a / rv)
        RH_col = np.clip(e_a / es_col, 0.0, 0.999)
    theta = np.tile(theta_col, (flow.Nx, 1))
    qv = np.tile(qv_col, (flow.Nx, 1))
    return theta, qv, P_col, T_col, theta0, RH_col


def _init_droplets(flow, n_super, N_modes, mu_um, sig, kappa, T_col, RH_col,
                   depth, seed):
    """Aerosol everywhere (it is, physically); each super-droplet's initial wet
    radius is its equilibrium at the LOCAL relative humidity of its height, so
    haze stays haze in the dry layer and only activates where lifted to saturation."""
    rng = np.random.default_rng(seed)
    N_modes = np.asarray(N_modes, float)
    mu = np.log(np.asarray(mu_um, float) * 1e-6)
    sg = np.log(np.asarray(sig, float))
    kappa = np.asarray(kappa, float)

    counts = (n_super * N_modes / N_modes.sum()).astype(int)
    counts[0] += n_super - counts.sum()
    r_dry, kap = [], []
    for k in range(len(N_modes)):
        r_dry.extend(rng.lognormal(mu[k], sg[k], counts[k]))
        kap.extend([kappa[k]] * counts[k])
    r_dry = np.asarray(r_dry)
    kap = np.asarray(kap)

    V_dom = flow.X * flow.Z * depth
    N_real = N_modes.sum() * 1e6 * V_dom            # N_modes given per cm^3
    A = np.full(n_super, max(round(N_real / n_super), 1.0))
    Ns = 4.0 / 3.0 * pi * rho_aero * r_dry ** 3 * A

    x = rng.uniform(0, flow.X, n_super)
    z = rng.uniform(0, flow.Z, n_super)
    iz = np.clip((z / flow.dz).astype(int), 0, flow.Nz - 1)
    S_loc = RH_col[iz] - 1.0                          # local subsaturation
    # vectorized equilibrium radius (bit-identical numba port of r_equi; the per-droplet
    # Python loop was ~1/3 of a short run's wall time at 200k super-droplets)
    r_wet = np.maximum(r_dry, r_equi_arr(S_loc, T_col[iz], r_dry, rho_aero, True, kap))
    M = 4.0 / 3.0 * pi * rho_liq * r_wet ** 3 * A
    return x, z, M, A, Ns, kap


# ---------------------------------------------------------------------------
# per-droplet local condensation (cell-indexed thermodynamics)
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _cond_grow_parallel(M, A, Ns, kappa, cidx, supersat_c, G_c, r0_c, afac_c,
                        dt, switch_kappa, switch_kelvin, switch_solute, dM_i, phase):
    """Per-droplet condensational growth (the dominant cost). Each droplet is
    INDEPENDENT, so this runs in parallel (prange) across cores; M[i] and the
    per-droplet mass change dM_i[i] are written by exactly one iteration (no race).
    The growth solver is a pure function of its inputs, so the result is identical
    regardless of thread scheduling — bit-for-bit the same as the serial version."""
    bfac_default = (vanthoff_aero * rho_aero * molecular_weight_water
                    / (rho_liq * molecular_weight_aero))
    for i in prange(M.shape[0]):
        if Ns[i] < 1.0e-200 or A[i] <= 0.0 or M[i] <= 0.0 or phase[i] == 1:
            dM_i[i] = 0.0                              # skip ice: it grows by deposition, not
            continue                                  # liquid Koehler (wrong density/l_v)
        c = cidx[i]
        afactor = afac_c[c]
        bfactor = kappa[i] if switch_kappa else bfac_default
        if not switch_kelvin:
            afactor = 0.0
        if not switch_solute:
            bfactor = 0.0
        r_liq = (M[i] / (A[i] * 4.0 / 3.0 * pi * rho_liq)) ** 0.33333333333
        r_N = (Ns[i] / (A[i] * 4.0 / 3.0 * pi * rho_aero)) ** 0.33333333333
        M_old = M[i]
        r_liq = radius_liquid_euler(r_liq, dt, r0_c[c], G_c[c], supersat_c[c],
                                    1.0, afactor, bfactor, r_N, 0.0, 0.0)
        M[i] = A[i] * 4.0 / 3.0 * pi * rho_liq * r_liq ** 3.0
        dM_i[i] = M[i] - M_old


@njit(cache=True)
def _scatter_dM(dM_i, cidx, dM_cell):
    """Serial scatter of per-droplet mass changes into per-cell totals, in ascending
    droplet order — identical accumulation order (hence bit-identical sum) to the
    original fused loop."""
    for i in range(dM_i.shape[0]):
        dM_cell[cidx[i]] += dM_i[i]


def _cond_local(M, A, Ns, kappa, cidx, supersat_c, G_c, r0_c, afac_c,
                dt, switch_kappa, switch_kelvin, switch_solute, dM_cell, phase=None):
    """Grow each droplet using its cell's thermodynamics; accumulate the per-cell
    condensed mass into dM_cell (flat). Mutates M in place. The expensive per-droplet
    growth is parallelised; the cell scatter is serial so dM_cell is bit-identical to
    the original serial implementation."""
    dM_i = np.empty(M.shape[0])
    if phase is None:                                 # warm-only callers: all liquid
        phase = np.zeros(M.shape[0], dtype=np.int8)
    _cond_grow_parallel(M, A, Ns, kappa, cidx, supersat_c, G_c, r0_c, afac_c,
                        dt, switch_kappa, switch_kelvin, switch_solute, dM_i, phase)
    _scatter_dM(dM_i, cidx, dM_cell)


@njit(parallel=True, cache=True)
def _cond_grow_parallel_sd(M, A, Ns, kappa, cidx, supersat_sd, G_c, r0_c, afac_c,
                           dt, switch_kappa, switch_kelvin, switch_solute, dM_i, phase):
    """LEM variant of _cond_grow_parallel: each droplet grows against its OWN
    supersaturation supersat_sd[i] (cell value + the Linear-Eddy-Model perturbation)
    rather than the cell mean. G/r0/afac stay per-cell (weak T' dependence). Identical
    to _cond_grow_parallel when supersat_sd[i] == supersat_c[cidx[i]] for all i."""
    bfac_default = (vanthoff_aero * rho_aero * molecular_weight_water
                    / (rho_liq * molecular_weight_aero))
    for i in prange(M.shape[0]):
        if Ns[i] < 1.0e-200 or A[i] <= 0.0 or M[i] <= 0.0 or phase[i] == 1:
            dM_i[i] = 0.0
            continue
        c = cidx[i]
        afactor = afac_c[c]
        bfactor = kappa[i] if switch_kappa else bfac_default
        if not switch_kelvin:
            afactor = 0.0
        if not switch_solute:
            bfactor = 0.0
        r_liq = (M[i] / (A[i] * 4.0 / 3.0 * pi * rho_liq)) ** 0.33333333333
        r_N = (Ns[i] / (A[i] * 4.0 / 3.0 * pi * rho_aero)) ** 0.33333333333
        M_old = M[i]
        r_liq = radius_liquid_euler(r_liq, dt, r0_c[c], G_c[c], supersat_sd[i],
                                    1.0, afactor, bfactor, r_N, 0.0, 0.0)
        M[i] = A[i] * 4.0 / 3.0 * pi * rho_liq * r_liq ** 3.0
        dM_i[i] = M[i] - M_old


def _cond_local_sd(M, A, Ns, kappa, cidx, supersat_sd, G_c, r0_c, afac_c,
                   dt, switch_kappa, switch_kelvin, switch_solute, dM_cell, phase):
    """As _cond_local but with a PER-SUPER-DROPLET supersaturation (LEM). Returns the
    per-droplet condensed mass dM_i (the LEM perturbation feedback needs it) in addition
    to scattering the per-cell total into dM_cell. Mutates M in place."""
    dM_i = np.empty(M.shape[0])
    _cond_grow_parallel_sd(M, A, Ns, kappa, cidx, supersat_sd, G_c, r0_c, afac_c,
                           dt, switch_kappa, switch_kelvin, switch_solute, dM_i, phase)
    _scatter_dM(dM_i, cidx, dM_cell)
    return dM_i


def _cell_index(flow, x, z):
    ix = np.clip((x / flow.dx).astype(np.int64), 0, flow.Nx - 1)
    iz = np.clip((z / flow.dz).astype(np.int64), 0, flow.Nz - 1)
    return ix, iz, ix * flow.Nz + iz          # flat C-order index


@njit(parallel=True, cache=True)
def _fall_speeds(M, A, rho_air, p_env, T_air, phase):
    """Per-droplet terminal fall speed — added to the flow velocity so big drops
    sediment out (rain falls). Liquid uses Beard (1976); ice uses the Locatelli-Hobbs
    (1974) power-law for unrimed dendrites / snow (v = 0.69*(2r)^0.41 m/s).
    Each droplet is independent (v[i] written once, no reduction, no RNG) so the
    loop runs in parallel — bit-identical to the serial version when phase is all-zero.
    """
    n = M.shape[0]
    v = np.zeros(n)
    for i in prange(n):
        if A[i] <= 0.0 or M[i] <= 0.0:
            continue
        if phase[i] == 1:                          # ice: Locatelli-Hobbs snow speed
            r = (M[i] / (A[i] * 4.0 / 3.0 * pi * rho_ice)) ** (1.0 / 3.0)
            v[i] = 0.69 * (2.0e3 * r) ** 0.41      # D in mm (L&H units) -> realistic snow
        else:                                      # liquid: Beard (unchanged)
            r = (M[i] / (A[i] * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0)
            v[i] = ws_drops_beard(r, rho_air[i], rho_liq, p_env[i], T_air[i])
    return v


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_flow2d(nt=600, dt=2.0, Nx=64, Nz=64, X=2000.0, Z=2000.0, W0=2.0,
               pattern="cumulus", L_thermal=500.0, z_inv=1500.0, dtheta_inv=0.0,
               n_super=4000, N_modes=(100.0,), mu_um=(0.08,),
               sig=(1.6,), kappa=(0.6,), T0=288.0, P0=1.0e5, RH0=0.93,
               z_bl=600.0, RH_top=0.2, depth=1.0, collisions=False,
               switch_TICE=False, eps=0.0, sediment=True,
               switch_kappa_koehler=True, seed=0, collect_every=20):
    """Run the 2D cumulus. Returns a dict of fields + a list of snapshot frames.

    Moisture is stratified: a moist boundary layer (RH0) up to z_bl, drier air
    (RH_top) aloft. `pattern` is 'cumulus', 'single_eddy', or 'thermal' (a
    localized updraft fed by far-field convergence, capped at z_inv with a warm
    inversion dtheta_inv aloft)."""
    np.random.seed(seed)
    seed_numba_rng(seed)
    flow = Flow2D(X=X, Z=Z, Nx=Nx, Nz=Nz, W0=W0, pattern=pattern,
                  L_thermal=L_thermal, z_inv=z_inv)
    theta, qv, P_col, T_col, theta0, RH_col = _base_state(
        flow, T0, P0, RH0, z_bl, RH_top,
        z_inv=(z_inv if pattern == "thermal" else None), dtheta_inv=dtheta_inv)
    x, z, M, A, Ns, ka = _init_droplets(flow, n_super, N_modes, mu_um, sig,
                                        kappa, T_col, RH_col, depth, seed)

    Cx = flow.u * dt / flow.dx
    Cz = flow.w * dt / flow.dz
    kap_exp = r_a / cp
    V_cell = flow.dx * flow.dz * depth
    # constant reference air mass per cell (Boussinesq-like): keeps total water
    # exactly conserved, consistent with MPDATA conserving sum(q_v).
    rho0 = P0 / (r_a * T0)
    air_mass_cell = rho0 * V_cell
    frames = []

    def _thermo():
        """Per-cell T, supersat, and growth coefficients from (theta, qv)."""
        T = theta * (P_col[None, :] / p0) ** kap_exp           # (Nx,Nz)
        e_s = _esatw_v(T)
        e_a = qv * P_col[None, :] / (qv + r_a / rv)
        supersat = e_a / e_s - 1.0
        tcond = 7.94048e-5 * T + 0.00227011
        diff = 0.211e-4 * (T / 273.15) ** 1.94 * (101325.0 / P_col[None, :])
        G = 1.0 / (rho_liq * rv * T / (e_s * diff)
                   + (l_v / (rv * T) - 1.0) * rho_liq * l_v / (tcond * T))
        r0 = diff / 0.036 * np.sqrt(2.0 * np.pi / (rv * T)) / (
            1.0 + diff * l_v ** 2 * e_s / (tcond * rv ** 2 * T ** 3))
        afac = 2.0 * _sigma_v(T) / (rho_liq * rv * T)
        rho = P_col[None, :] / (r_a * T)
        return T, supersat, G, r0, afac, rho

    # sub-cycle condensation so its step stays <= 0.5 s regardless of the flow
    # dt: a coarse condensation step overshoots the supersaturation badly (the
    # 1D RH>>100% effect, amplified in 2D). Flow advection keeps the larger dt.
    n_sub = max(1, int(np.ceil(dt / 0.5)))
    dt_sub = dt / n_sub

    for t in range(nt):
        # 1. advect droplets
        x, z = flow.advect(x, z, dt)
        if sediment:                               # drops fall at their terminal speed
            iz = np.clip((z / flow.dz).astype(np.int64), 0, flow.Nz - 1)
            rho_a = P_col[iz] / (r_a * T_col[iz])
            vt = _fall_speeds(M, A, rho_a, P_col[iz], T_col[iz],
                              np.zeros(M.shape[0], np.int8))
            z = np.clip(z - vt * dt, 0.0, flow.Z)
        # 2. advect scalars
        theta = mpdata_step(theta, Cx, Cz)
        qv = mpdata_step(qv, Cx, Cz)
        # 3. condensation (two-way coupled), sub-cycled for stability
        ix, iz, cidx = _cell_index(flow, x, z)
        for _ in range(n_sub):
            T, supersat, G, r0, afac, rho = _thermo()
            dM_cell = np.zeros(flow.Nx * flow.Nz)
            _cond_local(M, A, Ns, ka, cidx, supersat.ravel(), G.ravel(),
                        r0.ravel(), afac.ravel(), dt_sub,
                        switch_kappa_koehler, True, True, dM_cell)
            dq = (dM_cell / air_mass_cell).reshape(flow.Nx, flow.Nz)
            qv -= dq
            theta += (l_v / cp * dq) * (p0 / P_col[None, :]) ** kap_exp
        # 4. collision (optional, grid-local)
        if collisions:
            M, A, Ns, ka, x, z = _collide_cells(flow, M, A, Ns, ka, x, z, ix, iz,
                                                dt, rho, P_col, T, switch_TICE, eps,
                                                depth)
        # 5. diagnostics
        if (t + 1) % collect_every == 0 or t == nt - 1:
            frames.append(_snapshot(flow, x, z, M, A, theta, qv, supersat,
                                    air_mass_cell, t + 1))

    return dict(flow=flow, x=x, z=z, M=M, A=A, Ns=Ns, kappa=ka,
                theta=theta, qv=qv, frames=frames, P_col=P_col)


@njit(cache=True)
def _collide_all(M, A, Ns, kappa, order, starts, ends, n_cells, dt,
                 rho_cell, P_cell, T_cell, swE, swV, swT, eps, V_cell, phase, rimed):
    """Collide EVERY grid cell in one njit pass (no per-cell Python overhead).
    `order` holds the ORIGINAL droplet indices sorted by cell; [starts[c], ends[c])
    is cell c's slice of it. The kernel operates on the original arrays through this
    indirection — no gather/scatter copies (the copies were ~60% of the collide cost).
    Reuses the validated `_collision_kernel` per cell — same droplets, same order,
    same arithmetic as the sorted-copy version -> bit-identical. V_cell is the
    grid-cell volume (the correct collision volume for the cell's droplets)."""
    for c in range(n_cells):
        lo = starts[c]; hi = ends[c]; n = hi - lo
        if n < 2:
            continue
        if not (150.0 < T_cell[c] < 400.0 and rho_cell[c] > 1.0e-3):
            continue                      # thermodynamically degenerate cell
        idx = order[lo:hi].copy()
        for k in range(n - 1, 0, -1):              # Fisher-Yates shuffle (LSM)
            jj = np.random.randint(0, k + 1)
            tmp = idx[k]; idx[k] = idx[jj]; idx[jj] = tmp
        half = n // 2
        _collision_kernel(M, A, Ns, kappa, idx, half, n, dt,
                          rho_cell[c], P_cell[c], T_cell[c], swE, swV, swT, eps, V_cell,
                          phase, c, rimed)


@njit(parallel=True, cache=True)
def _collide_all_par(M, A, Ns, kappa, order, starts, ends, n_cells, dt,
                     rho_cell, P_cell, T_cell, swE, swV, swT, eps, V_cell, phase, rimed,
                     salt):
    """OPT-IN parallel variant of _collide_all: prange over cells. Grid cells are
    physically independent (each collides only its own droplets — order[starts[c]:ends[c]]
    are disjoint index sets; rimed[c] is per-cell), so the only serial coupling in
    _collide_all is the shared global RNG stream. Here each cell draws from its OWN
    counter-based splitmix64 stream seeded by (salt, cell) — a pure function of the
    seed/step/cell, so the result is DETERMINISTIC for a fixed seed and independent of
    thread count/schedule, but it is a DIFFERENT (statistically equivalent) random
    realization than the serial golden stream. Same collision physics
    (_collision_kernel with use_global_rng=False)."""
    GOLD = np.uint64(0x9E3779B97F4A7C15)
    for c in prange(n_cells):
        lo = starts[c]; hi = ends[c]; n = hi - lo
        if n < 2:
            continue
        if not (150.0 < T_cell[c] < 400.0 and rho_cell[c] > 1.0e-3):
            continue                      # thermodynamically degenerate cell
        state = (np.uint64(salt) ^ (np.uint64(c + 1) * GOLD))
        state, _ = _sm64_next(state)                  # decorrelate nearby cell seeds
        idx = order[lo:hi].copy()
        for k in range(n - 1, 0, -1):                 # Fisher-Yates shuffle (LSM)
            state, z = _sm64_next(state)
            jj = int(z % np.uint64(k + 1))
            tmp = idx[k]; idx[k] = idx[jj]; idx[jj] = tmp
        half = n // 2
        _collision_kernel(M, A, Ns, kappa, idx, half, n, dt,
                          rho_cell[c], P_cell[c], T_cell[c], swE, swV, swT, eps, V_cell,
                          phase, c, rimed, False, state)


@njit(cache=True)
def _counting_sort_by_cell(flat, Nc):
    """Stable counting sort of droplet indices by flat cell index (in [0, Nc)).
    Returns (order, starts, ends): `order` is identical to np.argsort(flat,
    kind='stable'); starts[c]/ends[c] are the [start, end) slice of cell c in the
    sorted order (identical to searchsorted left/right). O(n + Nc) vs O(n log n)."""
    n = flat.shape[0]
    off = np.zeros(Nc + 1, np.int64)
    for i in range(n):
        off[flat[i] + 1] += 1
    for c in range(Nc):
        off[c + 1] += off[c]
    starts = off[:Nc].copy()
    ends = off[1:Nc + 1].copy()
    order = np.empty(n, np.int64)
    pos = off[:Nc].copy()
    for i in range(n):                                 # ascending i -> stable
        c = flat[i]
        order[pos[c]] = i
        pos[c] += 1
    return order, starts, ends


def _collide_cells(flow, M, A, Ns, ka, x, z, ix, iz, dt, rho, P_col, T,
                   switch_TICE=False, eps=0.0, depth=1.0, tag=None, phase=None, inp=None,
                   rimed_out=None, charge=None, hab=None, parallel=False, salt=0):
    """Run validated LSM collision independently in each grid cell, then drop
    dissolved (A<=0) super-droplets globally (positions included). Optional
    per-droplet arrays (`tag`, `phase`, `inp`, `charge`) ride along by index and are
    returned, in that order, subset by the same survival mask when supplied."""
    Nc = flow.Nx * flow.Nz
    V_cell = flow.dx * flow.dz * depth            # correct per-cell collision volume
    flat = (ix * flow.Nz + iz).astype(np.int64)
    order, starts, ends = _counting_sort_by_cell(flat, Nc)
    # The kernels operate on the ORIGINAL arrays through the order[lo:hi] indirection —
    # no gather/scatter copies (they were ~60% of the collide cost). Same droplets, same
    # processing order, same arithmetic as the old sorted-copy version -> bit-identical.
    P_cell = np.tile(P_col, flow.Nx)              # c = i*Nz+j -> P_col[j]
    phase_k = (phase if phase is not None
               else np.zeros(M.shape[0], dtype=np.int8))      # all-liquid when no ice
    rimed = np.zeros(Nc)                          # per-cell frozen (rimed) liquid mass
    if parallel:                                  # opt-in: per-cell splitmix64 streams,
        _collide_all_par(M, A, Ns, ka, order, starts, ends, Nc, float(dt),  # prange cells
                         np.ascontiguousarray(rho.ravel()), P_cell,
                         np.ascontiguousarray(T.ravel()),
                         False, False, bool(switch_TICE), float(eps), float(V_cell),
                         phase_k, rimed, np.uint64(salt))
    else:                                         # default: serial global-RNG (golden)
        _collide_all(M, A, Ns, ka, order, starts, ends, Nc, float(dt),
                     np.ascontiguousarray(rho.ravel()), P_cell,
                     np.ascontiguousarray(T.ravel()),
                     False, False, bool(switch_TICE), float(eps), float(V_cell),
                     phase_k, rimed)
    if rimed_out is not None:
        rimed_out[:] = rimed
    keep = A > 0.0
    extras = [e[keep] for e in (tag, phase, inp) if e is not None]
    if charge is not None:
        # a dissolved super-droplet (A<=0) merged into a survivor; its CHARGE must follow
        # like its mass did. Redistribute each cell's dropped charge evenly over that
        # cell's survivors -> charge is conserved across collision merges (exactly, except
        # for the rare cell whose every super-droplet dissolved).
        Nc2 = flow.Nx * flow.Nz
        lost = np.zeros(Nc2)
        np.add.at(lost, flat[~keep], charge[~keep])
        keepc = flat[keep]
        nsurv = np.bincount(keepc, minlength=Nc2)
        redist = np.where(nsurv > 0, lost / np.maximum(nsurv, 1), 0.0)
        extras.append(charge[keep] + redist[keepc])
    if hab is not None:                                # ice-habit state (N,3) rides by row
        extras.append(hab[keep])
    return (M[keep], A[keep], Ns[keep], ka[keep], x[keep], z[keep], *extras)


def _snapshot(flow, x, z, M, A, theta, qv, supersat, air_mass_cell, step):
    """Grid the liquid water (g/kg) and record droplet radii for animation."""
    ix, iz, cidx = _cell_index(flow, x, z)
    liq = np.zeros(flow.Nx * flow.Nz)
    np.add.at(liq, cidx, M)
    qc = (liq.reshape(flow.Nx, flow.Nz) / air_mass_cell) * 1e3
    r_um = np.where(A > 0, (M / (A * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0), 0.0) * 1e6
    return dict(step=step, x=x.copy(), z=z.copy(), r_um=r_um, qc=qc,
                supersat=supersat.copy(), theta=theta.copy(), qv=qv.copy())
