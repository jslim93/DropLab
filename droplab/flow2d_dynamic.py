"""DYNAMIC 2D moist convection: the flow is no longer prescribed but DRIVEN by
buoyancy (vorticity-streamfunction Boussinesq). Condensation latent heat warms
theta -> buoyancy -> updraft -> more lifting/condensation: the cloud organizes
itself, rises by its own buoyancy, and is physically capped by the inversion
(a buoyant parcel decelerates in the stable layer).

Vorticity-streamfunction form (2D incompressible):
    omega = du/dz - dw/dx ,   lap(psi) = -omega ,   (u,w) = (-dpsi/dz, dpsi/dx)
    d omega/dt + u.grad omega = -d b/dx + nu lap(omega)   (source is -db/dx; see code below)
    b = g (theta'/theta0 + 0.608 q_v' - q_c)            (buoyancy; q_c loads down)

Reuses everything else: MPDATA scalar advection, the per-droplet condensation
coupling, grid-local collision, and the multi-panel visualization.
"""
import math
import numpy as np
from numba import njit, prange

from droplab.parameters import p0, r_a, cp, rv, l_v, l_f, l_s, g, rho_liq, rho_ice, rho_aero, pi
from droplab.ice_microphysics import (_bigg_freeze, _abifm_freeze, _homogeneous_freeze,
                                    _melt, _hallett_mossop, ABIFM_SPECIES, _ice_deposition,
                                    _scatter_dM_ice, _esati_v, g_ice)
from droplab.condensation import esatw
from droplab.flow2d import Flow2D
from droplab.poisson import (solve_poisson, laplacian_dirichlet,
                           solve_poisson_periodic_x, laplacian_periodic_x,
                           solve_poisson_anelastic, solve_poisson_anelastic_periodic_x)
from droplab.mpdata import (mpdata_step, upwind_step, mpdata_step_periodic_x,
                          upwind_step_periodic_x)
from droplab.collision_soa import seed_numba_rng
from droplab import electrification as _elec
from droplab import ice_habit as _hab
from droplab import lem_driver as _lem
from droplab.flow2d_driver import (_base_state, _init_droplets, _rh_profile,
                                 _cond_local, _cond_local_sd, _cell_index, _collide_cells,
                                 _esatw_v, _sigma_v, _fall_speeds)


def _faces_from_psi(psi, dx, dz):
    """Discrete-divergence-free face velocities from a cell-centered psi (psi=0
    on the walls), via corner averaging — same staggering as Flow2D."""
    Nx, Nz = psi.shape
    pc = np.zeros((Nx + 1, Nz + 1))
    pc[1:Nx, 1:Nz] = 0.25 * (psi[:-1, :-1] + psi[1:, :-1]
                             + psi[:-1, 1:] + psi[1:, 1:])
    u = -(pc[:, 1:] - pc[:, :-1]) / dz          # (Nx+1, Nz)
    w = (pc[1:, :] - pc[:-1, :]) / dx           # (Nx, Nz+1)
    return u, w


def _faces_from_psi_periodic(psi, dx, dz):
    """Divergence-free face velocities from a cell-centered psi, PERIODIC in x and
    psi=0 on the z lids. u on x-faces (Nx,Nz, periodic), w on z-faces (Nx,Nz+1)."""
    Nx, Nz = psi.shape
    pim1 = np.roll(psi, 1, axis=0)
    pc = np.zeros((Nx, Nz + 1))                  # corner psi; =0 at z lids
    pc[:, 1:Nz] = 0.25 * (pim1[:, :-1] + psi[:, :-1] + pim1[:, 1:] + psi[:, 1:])
    u = -(pc[:, 1:] - pc[:, :-1]) / dz           # (Nx, Nz)
    w = (np.roll(pc, -1, axis=0) - pc) / dx      # (Nx, Nz+1)
    return u, w


def _faces_from_psi_anelastic(psi, dx, dz, rho0_c, rho0_f):
    """Closed-box faces from the MASS streamfunction: rho0 u = -dpsi/dz, rho0 w = dpsi/dx.
    u lives at cell-center z (divide by rho0_c (Nz,)); w at z-faces (divide by rho0_f (Nz+1,)).
    Dividing by rho0_c/rho0_f makes the mass flux (rho0_c*u, rho0_f*w) the EXACT discrete curl
    of psi, so div(rho0 V)=0 to machine zero -- which the rho0-weighted scalar transport
    requires. (The anelastic Poisson operator uses the arithmetic mean of 1/rho0 at z-faces,
    which differs from 1/rho0_f by ~1%; that is an unavoidable single-face-value tradeoff and we
    keep exact mass conservation over an exact vorticity round-trip.)"""
    Nx, Nz = psi.shape
    pc = np.zeros((Nx + 1, Nz + 1))
    pc[1:Nx, 1:Nz] = 0.25 * (psi[:-1, :-1] + psi[1:, :-1]
                             + psi[:-1, 1:] + psi[1:, 1:])
    u = -(pc[:, 1:] - pc[:, :-1]) / dz / rho0_c[None, :]      # (Nx+1, Nz)
    w = (pc[1:, :] - pc[:-1, :]) / dx / rho0_f[None, :]       # (Nx, Nz+1)
    return u, w


def _faces_from_psi_anelastic_periodic(psi, dx, dz, rho0_c, rho0_f):
    """Periodic-x faces from the MASS streamfunction (rho0 u = -dpsi/dz, rho0 w = dpsi/dx).
    Mass flux = exact discrete curl of psi -> div(rho0 V)=0 (see _faces_from_psi_anelastic)."""
    Nx, Nz = psi.shape
    pim1 = np.roll(psi, 1, axis=0)
    pc = np.zeros((Nx, Nz + 1))
    pc[:, 1:Nz] = 0.25 * (pim1[:, :-1] + psi[:, :-1] + pim1[:, 1:] + psi[:, 1:])
    u = -(pc[:, 1:] - pc[:, :-1]) / dz / rho0_c[None, :]      # (Nx, Nz)
    w = (np.roll(pc, -1, axis=0) - pc) / dx / rho0_f[None, :]  # (Nx, Nz+1)
    return u, w


_ALL_RATES = ("cond", "evap", "dep", "sub", "freeze", "melt", "rime", "sip")


def _wanted_rates(diagnose):
    """Resolve the `diagnose` argument to the set of process-rate names to accumulate.
    "micro"/"full" -> all; a list/tuple -> its intersection with the rate names; else none."""
    if not diagnose:
        return set()
    if isinstance(diagnose, str):
        return set(_ALL_RATES) if diagnose in ("micro", "full") else {diagnose} & set(_ALL_RATES)
    return set(diagnose) & set(_ALL_RATES)


def _ddx(field, dx):
    d = np.zeros_like(field)
    d[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dx)
    d[0, :] = (field[1, :] - field[0, :]) / dx
    d[-1, :] = (field[-1, :] - field[-2, :]) / dx
    return d


def _ddx_periodic(field, dx):
    return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * dx)


@njit(cache=True)
def _bilin_periodic(f, xx, zz, dx, dz, Nx, Nz):
    """Bilinear interpolation of cell-centered field `f` at (xx, zz), PERIODIC in x,
    edge-clamped in z. Arithmetic matches the original numpy expression bit-for-bit."""
    xi = xx / dx - 0.5
    zj = zz / dz - 0.5
    i0 = int(math.floor(xi))
    j0 = int(math.floor(zj))
    fx = xi - i0
    fz = zj - j0
    i0m = i0 % Nx
    i1m = (i0 + 1) % Nx
    j0c = min(max(j0, 0), Nz - 1)
    j1c = min(max(j0 + 1, 0), Nz - 1)
    return ((1 - fx) * (1 - fz) * f[i0m, j0c] + fx * (1 - fz) * f[i1m, j0c]
            + (1 - fx) * fz * f[i0m, j1c] + fx * fz * f[i1m, j1c])


@njit(parallel=True, cache=True)
def _advect_periodic(x, z, uc, wc, X, Z, dx, dz, dt):
    """RK2 droplet advection on cell-centered velocities, PERIODIC in x (positions
    wrap mod X) and clamped in z. Per-droplet loop, run in PARALLEL across droplets
    (each writes its own xo[p]/zo[p], no reduction, no RNG) — bit-identical to the
    serial/vectorised version, ~no temporaries."""
    Nx, Nz = uc.shape
    n = x.shape[0]
    xo = np.empty(n)
    zo = np.empty(n)
    for p in prange(n):
        u1 = _bilin_periodic(uc, x[p], z[p], dx, dz, Nx, Nz)
        w1 = _bilin_periodic(wc, x[p], z[p], dx, dz, Nx, Nz)
        xm = (x[p] + 0.5 * dt * u1) % X
        zm = min(max(z[p] + 0.5 * dt * w1, 0.0), Z)
        u2 = _bilin_periodic(uc, xm, zm, dx, dz, Nx, Nz)
        w2 = _bilin_periodic(wc, xm, zm, dx, dz, Nx, Nz)
        xo[p] = (x[p] + dt * u2) % X
        zo[p] = min(max(z[p] + dt * w2, 0.0), Z)
    return xo, zo


def _laplacian_neumann(f, dx, dz):
    """Laplacian with zero-gradient (no-flux) walls — for scalar diffusion."""
    lap = np.zeros_like(f)
    lap[1:-1, :] += (f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / dx ** 2
    lap[0, :] += (f[1, :] - f[0, :]) / dx ** 2
    lap[-1, :] += (f[-2, :] - f[-1, :]) / dx ** 2
    lap[:, 1:-1] += (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / dz ** 2
    lap[:, 0] += (f[:, 1] - f[:, 0]) / dz ** 2
    lap[:, -1] += (f[:, -2] - f[:, -1]) / dz ** 2
    return lap


def _init_inp(n_super, inp_n_cm3, inp_r_um, inp_sigma, N_modes, seed):
    """Assign an immersed-INP surface area [m^2] to a random subset of super-droplets,
    so a base INP population is present from the start (ABIFM). The INP-bearing
    FRACTION of super-droplets = inp_n_cm3 / total aerosol number conc; each such
    super-droplet gets 4*pi*r_inp^2 with r_inp ~ lognormal(inp_r_um um, inp_sigma).
    Mirrors the SAM6-LCM frac_ice / n_ice / rm_ice / sigma_ice initialisation."""
    inp = np.zeros(n_super)
    n_aero = float(np.asarray(N_modes, float).sum())
    if inp_n_cm3 <= 0.0 or n_aero <= 0.0:
        return inp
    n_inp = int(round(min(1.0, inp_n_cm3 / n_aero) * n_super))
    if n_inp <= 0:
        return inp
    rng = np.random.default_rng(seed + 101)
    idx = rng.choice(n_super, size=n_inp, replace=False)
    r_inp = rng.lognormal(np.log(inp_r_um * 1e-6), np.log(inp_sigma), n_inp)
    inp[idx] = 4.0 * pi * r_inp ** 2
    return inp


def _inject_aerosol(M, A, Ns, ka, x, z, tag, phase, inp, X, depth, spec, seed, tag_id=1):
    """Add super-droplets representing injected aerosol. Two intervention modes:

    - CCN seeding (default): liquid CCN are appended and activate into cloud drops,
      so MCB raises the droplet number (Twomey brightening) and GCCN seeding promotes
      drizzle.
    - INP / glaciogenic seeding (`spec["phase"]="ice"`): ICE embryos are injected
      (phase=1) — an efficient ice-nucleating particle in a supercooled cloud freezes
      its host drop at once, so the faithful abstraction is a small ice crystal. They
      grow by deposition (WBF), pulling vapour from the air and evaporating the
      surrounding liquid, so a localised INP injection drives a spreading glaciation
      front — the mixed-phase analogue of MCB.

    `spec`: x_frac=(lo,hi), z_lo, z_hi, N_cm3 (added conc), r_um (dry radius),
    kappa, n_super (how many computational droplets to add); optional r_wet_um and
    phase ("liquid" default, or "ice" for INP seeding)."""
    n = int(spec["n_super"])
    rng = np.random.default_rng(seed)
    x0, x1 = spec["x_frac"][0] * X, spec["x_frac"][1] * X
    zlo, zhi = spec["z_lo"], spec["z_hi"]
    V_reg = (x1 - x0) * (zhi - zlo) * depth
    N_real = spec["N_cm3"] * 1e6 * V_reg                  # total injected aerosols
    A_new = max(round(N_real / n), 1.0)
    r_dry = spec["r_um"] * 1e-6                            # dry (solute / INP core) radius
    # initial WET radius: for small sea-salt (MCB) it starts near its dry size and
    # activates by condensation; a giant CCN (high solute) is always above its Kohler
    # critical point and grows to a drizzle embryo essentially at once, so it can be
    # initialised already activated via r_wet_um (>> r_dry) without waiting. For INP
    # seeding r_wet_um is the initial ice-crystal radius.
    r_wet = spec.get("r_wet_um", spec["r_um"]) * 1e-6
    is_ice = spec.get("phase") == "ice"                   # direct-ice (glaciogenic) injection
    rho_new = rho_ice if is_ice else rho_liq
    phase_val = np.int8(1) if is_ice else np.int8(0)
    # INP-bearing liquid seeding: injected drops carry an immersed INP area (they then
    # freeze via ABIFM) when spec["inp_r_um"] is given; direct-ice injection carries none.
    inp_new = 4.0 * pi * (spec["inp_r_um"] * 1e-6) ** 2 if ("inp_r_um" in spec and not is_ice) else 0.0
    Ns_new = 4.0 / 3.0 * pi * rho_aero * r_dry ** 3 * A_new
    M_new = 4.0 / 3.0 * pi * rho_new * r_wet ** 3 * A_new  # ice mass if INP, else liquid
    xa = rng.uniform(x0, x1, n)
    za = rng.uniform(zlo, zhi, n)
    return (np.concatenate([M, np.full(n, M_new)]),
            np.concatenate([A, np.full(n, A_new)]),
            np.concatenate([Ns, np.full(n, Ns_new)]),
            np.concatenate([ka, np.full(n, spec["kappa"])]),
            np.concatenate([x, xa]), np.concatenate([z, za]),
            np.concatenate([tag, np.full(n, tag_id, dtype=np.int64)]),
            np.concatenate([phase, np.full(n, phase_val, dtype=np.int8)]),
            np.concatenate([inp, np.full(n, inp_new)]))


def _ihmd_mix(M, A, cidx, ncell, M0_cell, ihmd, r_cloud=1.0e-6):
    """Inhomogeneous-mixing adjustment for entrainment evaporation (Lim & Hoffmann
    2023). In each cell that LOST cloud water this step (beta = M_after/M_before < 1,
    i.e. dry air was entrained and droplets evaporated), reduce the droplet NUMBER by
    beta**ihmd instead of leaving every droplet shrunk:

        ihmd = 0  homogeneous   — all droplets shrink, N kept   (the model default)
        ihmd = 1  inhomogeneous — beta*N droplets survive at their original size

    Liquid water is untouched (only multiplicity A changes), so q_c is identical; the
    signature is in the droplet spectrum — inhomogeneous mixing leaves FEWER, LARGER
    drops (bigger r_eff, lower albedo, more collision-prone), which then diverges
    downstream through collision and sedimentation. Mutates A in place."""
    M1 = np.zeros(ncell)
    np.add.at(M1, cidx, M)
    beta = np.ones(ncell)
    nz = M0_cell > 0.0
    beta[nz] = np.clip(M1[nz] / M0_cell[nz], 0.0, 1.0)
    r = np.where(A > 0.0, (M / (A * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0), 0.0)
    cloud = r > r_cloud
    A[cloud] = A[cloud] * beta[cidx[cloud]] ** ihmd
    return A


def _radiative_cooling(qc, rho0, dz, F0, F1, kappa):
    """Cloud-top longwave radiative cooling (Stevens et al. 2005, DYCOMS-II).

    Net LW flux F(z) = F0*exp(-kappa*LWP_above) + F1*exp(-kappa*LWP_below); the
    flux divergence concentrates STRONG cooling in a thin layer at cloud top.
    This top-down cooling is the Rayleigh-Benard driver that sustains stratocumulus.
    qc is the liquid mixing ratio (kg/kg); returns the theta tendency (K/s)."""
    lwp = rho0 * qc * dz                                   # per-cell LWP (kg/m^2)
    lwp_above = np.cumsum(lwp[:, ::-1], axis=1)[:, ::-1] - lwp
    lwp_below = np.cumsum(lwp, axis=1) - lwp
    F = F0 * np.exp(-kappa * lwp_above) + F1 * np.exp(-kappa * lwp_below)
    dFdz = np.empty_like(F)
    dFdz[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2.0 * dz)
    dFdz[:, 0] = (F[:, 1] - F[:, 0]) / dz
    dFdz[:, -1] = (F[:, -1] - F[:, -2]) / dz
    return -dFdz / (rho0 * cp)                             # dtheta/dt (K/s)


def _laplacian_neumann_px(f, dx, dz):
    """Scalar-diffusion Laplacian: PERIODIC in x, no-flux at the z lids."""
    lap = (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx ** 2
    lap[:, 1:-1] += (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / dz ** 2
    lap[:, 0] += (f[:, 1] - f[:, 0]) / dz ** 2
    lap[:, -1] += (f[:, -2] - f[:, -1]) / dz ** 2
    return lap


def run_flow2d_dynamic(nt=1500, dt=1.5, Nx=96, Nz=72, X=4800.0, Z=3000.0,
                       n_super=90000, N_modes=(60.0,), mu_um=(0.08,), sig=(2.0,),
                       kappa=(0.6,), T0=289.0, P0=1.0e5, RH0=0.90, z_bl=600.0,
                       RH_top=0.2, z_inv=1400.0, dtheta_inv=4.0, gamma_theta=0.004,
                       depth=1.0, nu=20.0, dtheta_bubble=0.5, bubble_z=400.0,
                       bubble_r=400.0, v_max=35.0, collisions=True,
                       switch_TICE=True, eps=0.01, switch_kappa_koehler=True,
                       seed=0, collect_every=20, sounding=None, forcing=None,
                       pert_amp=0.1, sediment=True, b_max=0.12, omega_max=0.03,
                       nu_scalar=2.0, periodic_x=True, rad_cool=None,
                       seeding=None, ihmd=0.0, surface_cool=0.0,
                       diurnal_period=None, wind_shear=0.0, on_frame=None, diagnose=None,
                       ice=False, freezing_mode="abifm", homogeneous=True, melt=True,
                       hallett_mossop=True, dynamics="boussinesq",
                       sponge_frac=0.0, sponge_tau=300.0,
                       a_bigg=0.66, B_bigg=100.0,
                       inp_n_cm3=0.0, inp_r_um=0.5, inp_sigma=1.4,
                       inp_species="default",
                       electrification=False, q_rev_T=263.15, E_breakdown=1.5e5,
                       charge_eff=0.1, q_sc_min=1.0e-5, flash_every=1,
                       flash_neutralize=0.7, flash_radius=2, flash_rearm=0.95,
                       sd_per_cell=None, sim_hours=None,
                       habit=False, collide_parallel=False,
                       lem=False, lem_eps=1.0e-3, lem_tau=900.0):
    """Buoyancy-driven 2D cumulus. A warm moist bubble triggers the thermal;
    everything after is self-organized. With `sounding` (e.g. droplab.soundings.BOMEX)
    the cloud is capped by a REAL inversion; otherwise the free troposphere is
    stably stratified (gamma_theta). Returns the same dict shape as run_flow2d
    (with per-frame w for the velocity panel).

    Convenience overrides (grid/dt-independent): `sd_per_cell` sets the super-droplet
    count to sd_per_cell*Nx*Nz (overrides n_super); `sim_hours` sets nt to cover that
    many hours of simulated time at the current dt (overrides nt). E.g. sd_per_cell=100,
    sim_hours=3 -> a well-resolved 3-hour run."""
    if sd_per_cell is not None:
        n_super = int(sd_per_cell * Nx * Nz)
    if sim_hours is not None:
        nt = int(round(sim_hours * 3600.0 / dt))
    # LEM is an OPTIONAL/experimental feature (warm condensation-spectrum broadening only).
    # Its per-SD state (eta_sd, w_sgs) is threaded through injection and sedimentation but NOT
    # through collisional merging, which prunes super-droplets -> the arrays would desync.
    # So LEM requires collisions=False for now; the full physics (collision/sedimentation/ice)
    # is the default with lem=False. This keeps LEM from limiting the model.
    if lem and (collisions or ice or sediment):
        raise ValueError(
            "lem=True is a warm-condensation-only broadening demo and is incompatible with "
            "collisions / ice / sedimentation: the per-super-droplet LEM supersaturation state "
            "is not threaded through collisional merging, ice phase change, or fallout removal, "
            "so combining them would give meaningless results. Run the LEM with "
            "collisions=False, ice=False, sediment=False, or leave lem=False (default) for the "
            "full model.")
    np.random.seed(seed)
    seed_numba_rng(seed)
    flow = Flow2D(X=X, Z=Z, Nx=Nx, Nz=Nz)          # geometry holder; u,w overwritten
    theta, qv, P_col, T_col, theta0, RH_col = _base_state(
        flow, T0, P0, RH0, z_bl, RH_top, z_inv=z_inv, dtheta_inv=dtheta_inv,
        gamma_theta=gamma_theta, sounding=sounding)
    theta_base = theta[0, :].copy()
    qv_base = qv[0, :].copy()
    x, z, M, A, Ns, ka = _init_droplets(flow, n_super, N_modes, mu_um, sig,
                                        kappa, T_col, RH_col, depth, seed)
    tag = np.zeros(M.shape[0], dtype=np.int64)      # 0 = background, >0 = seeded
    phase = np.zeros(M.shape[0], dtype=np.int8)     # 0 = liquid, 1 = ice (ice=True only)
    # immersed-INP surface area per super-droplet (m^2); 0 = no INP. A base population
    # of INP-bearing super-droplets is present from the start (ABIFM nucleation).
    inp = (_init_inp(M.shape[0], inp_n_cm3, inp_r_um, inp_sigma, N_modes, seed)
           if ice else np.zeros(M.shape[0]))
    c_abifm, m_abifm = ABIFM_SPECIES.get(inp_species, ABIFM_SPECIES["default"])
    # per-super-droplet charge [C] (None when off -> zero overhead, bit-identical).
    # Electrification needs riming graupel + ice, so it only does anything with ice on.
    charge = np.zeros(M.shape[0]) if electrification else None
    # ice-habit state per super-droplet: (N,3) = [a_axis, c_axis, rho_app]; None when off.
    # 0 = "no shape yet" -> seeded to a sphere the step an SD becomes ice. Needs ice=True.
    hab = np.zeros((M.shape[0], 3)) if (habit and ice) else None
    # Linear Eddy Model state: per-SD zero-mean (T',q') + virtual-line position + AR-1 SGS
    # velocity. None when off -> bit-identical. The line length scales with SDs-per-cell.
    if lem:
        # SAM-LCM LEM: each SD is one LEM box; the SDs in a grid cell form the vertical LEM
        # line (dz_sgs = dz_cell/n_SD). Each SD carries a prognostic supersaturation eta_sd
        # with ~tau memory; transport of that memory through the heterogeneous supersaturation
        # field broadens the spectrum. See droplab.lem_driver.
        _lem_rng = np.random.default_rng(seed + 4242)
        eta_sd, w_sgs = _lem.init_lem_state(M.shape[0])
    else:
        eta_sd = w_sgs = None
    _elec_rng = np.random.default_rng(seed + 777)   # local RNG: flash stochasticity only
    _flashes_acc = []                                # flash channels since last frame
    _flash_armed = True                              # discharge hysteresis (relaxation oscillator):
    #   fire when the field crosses breakdown while ARMED; then disarm until the peak
    #   field falls below flash_rearm*E_crit. A real flash collapses the local field and
    #   the storm must recharge before the next one -- without this, the un-drained 2-D
    #   dipole re-fires EVERY step and the sky strobes continuously (unphysical).
    charge_to_ground = 0.0                           # charge carried out by precipitation
    _efield_hist = []                                # (step, max|E|, n_flash) per flash check

    zc = (np.arange(Nz) + 0.5) * flow.dz
    rho0 = P0 / (r_a * T0)
    # large-scale + surface forcing (sustains a realistic gentle cumulus field)
    if forcing is not None:
        tls_col = np.interp(zc, forcing["z"], forcing["tls"])
        qls_col = np.interp(zc, forcing["z"], forcing["qls"])
        wls_col = np.interp(zc, forcing["z"], forcing["wls"])
        sfc_th = forcing["H"] / (rho0 * cp * flow.dz)        # K/s into lowest level
        sfc_qv = forcing["LE"] / (rho0 * l_v * flow.dz)      # kg/kg/s
    if forcing is not None or rad_cool is not None or surface_cool != 0.0:
        # trigger: small random theta perturbations in the boundary layer (a cloud
        # FIELD / Sc deck / fog — cells seed everywhere, not one central plume)
        rng = np.random.default_rng(seed)
        bl = zc < z_bl
        theta[:, bl] += pert_amp * rng.standard_normal((Nx, bl.sum()))
    else:
        # trigger: a single warm bubble low in the centre (one cumulus)
        xc = (np.arange(Nx) + 0.5) * flow.dx
        XX, ZZ = np.meshgrid(xc, zc, indexing="ij")
        theta = theta + dtheta_bubble * np.exp(
            -(((XX - X / 2) / bubble_r) ** 2 + ((ZZ - bubble_z) / bubble_r) ** 2))

    omega = np.zeros((Nx, Nz))
    kap = r_a / cp
    rho0 = P0 / (r_a * T0)                          # surface density (forcing/radiation)
    anelastic = (dynamics == "anelastic")
    if anelastic:
        # height-varying base density rho0(z): a cell aloft holds less air, so the same
        # condensed mass is a larger mixing-ratio increment -> stronger buoyancy feedback.
        rho0_c = P_col / (r_a * T_col)             # cell-center base density (Nz,)
        rho0_f = np.empty(Nz + 1)                  # base density at z-faces (for the w-face,
        rho0_f[1:Nz] = 0.5 * (rho0_c[:-1] + rho0_c[1:])   # keeps div(rho0 V)=0 exactly)
        rho0_f[0] = rho0_c[0]; rho0_f[Nz] = rho0_c[-1]
        beta = 1.0 / rho0_c                        # 1/rho0 for the anelastic Poisson
        air_mass_cell = np.tile(rho0_c, Nx) * flow.dx * flow.dz * depth   # flat (Nx*Nz,)
    else:
        air_mass_cell = rho0 * flow.dx * flow.dz * depth                  # scalar (Boussinesq)
    surf_precip = 0.0                              # accumulated surface precipitation (kg)
    n_sub = max(1, int(np.ceil(dt / 0.5)))
    dt_sub = dt / n_sub
    frames = []

    def _qc_grid():
        _, _, cidx = _cell_index(flow, x, z)
        liq = np.zeros(Nx * Nz)
        np.add.at(liq, cidx, M)
        return (liq / air_mass_cell).reshape(Nx, Nz)      # kg/kg

    def _thermo():
        T = theta * (P_col[None, :] / p0) ** kap
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
        return T, supersat, G, r0, afac

    # a prescribed mean horizontal wind U(z)=wind_shear*(z-Z/2) only makes sense with a
    # periodic x-boundary: in a closed box it advects scalars and droplets into the wall
    # and blows up. Shear therefore implies periodic x (this is what made wind_shear work
    # only in the purpose-built shear case, which set periodic_x=True explicitly).
    if wind_shear != 0.0:
        periodic_x = True

    # select closed-box vs periodic-x operators once
    if anelastic:
        # variable-coefficient (1/rho0) Poisson + mass-streamfunction faces. Boussinesq keeps
        # the exact spectral path below -> golden bit-identity is preserved.
        if periodic_x:
            _poisson = lambda rhs, dx, dz: solve_poisson_anelastic_periodic_x(rhs, dx, dz, beta)
            _faces = lambda psi, dx, dz: _faces_from_psi_anelastic_periodic(psi, dx, dz, rho0_c, rho0_f)
        else:
            _poisson = lambda rhs, dx, dz: solve_poisson_anelastic(rhs, dx, dz, beta)
            _faces = lambda psi, dx, dz: _faces_from_psi_anelastic(psi, dx, dz, rho0_c, rho0_f)
    else:
        _poisson = solve_poisson_periodic_x if periodic_x else solve_poisson
        _faces = _faces_from_psi_periodic if periodic_x else _faces_from_psi
    _upwind = upwind_step_periodic_x if periodic_x else upwind_step
    _ddx_op = _ddx_periodic if periodic_x else _ddx
    _lap_om = laplacian_periodic_x if periodic_x else laplacian_dirichlet
    _mpdata = mpdata_step_periodic_x if periodic_x else mpdata_step

    # mean horizontal wind U(z) = wind_shear*(z - Z/2): a LINEAR vertical shear that
    # differentially advects the flow, scalars and droplets. Updrafts then tilt
    # downshear and convection organises into bands / cloud streets (and, with a cold
    # pool, squall-line-like propagation). Centred so the domain-mean wind is ~0 (the
    # field stays in the periodic window). Linear shear has d2U/dz2 = 0, so advection
    # by U(z) is the COMPLETE effect — no vortex-stretching term to add. Use with
    # periodic_x (a mean wind would sweep the field out of a closed domain).
    U_z = wind_shear * (zc - 0.5 * flow.Z)            # (Nz,)

    # Rayleigh sponge: damp perturbations in the top sponge_frac of the domain so that
    # convectively-generated gravity waves are ABSORBED near the lid instead of reflecting
    # off it (a rigid-lid artifact that otherwise leaves spurious oscillating cloud aloft).
    # The damping rate ramps smoothly from 0 at the sponge base to 1/sponge_tau at the top
    # (Klemp-Lilly form). Off by default (sponge_frac=0) -> golden bit-identical.
    damp_z = np.zeros(Nz)
    if sponge_frac > 0.0:
        z_s = (1.0 - sponge_frac) * flow.Z
        in_sponge = zc > z_s
        damp_z[in_sponge] = (1.0 / sponge_tau) * np.sin(
            0.5 * pi * (zc[in_sponge] - z_s) / (flow.Z - z_s)) ** 2

    # process-rate diagnostics (diagnose=): accumulate per-cell process amounts over each
    # output interval; emitted as rates in frame["rates"]. Empty by default -> no cost.
    _diag = _wanted_rates(diagnose)
    _acc = {k: np.zeros(Nx * Nz) for k in _diag}
    _acc_dt = 0.0

    for t in range(nt):
        # 0. aerosol seeding — inject CCN super-droplets at the scheduled step(s).
        # This is the climate-intervention lever: small sea-salt for marine cloud
        # brightening (Twomey), or giant CCN for precipitation cloud-seeding.
        if seeding is not None:
            specs = seeding if isinstance(seeding, (list, tuple)) else [seeding]
            for k, spec in enumerate(specs):
                if t == int(spec["t_inject"] / dt):
                    _n0 = M.shape[0]
                    M, A, Ns, ka, x, z, tag, phase, inp = _inject_aerosol(
                        M, A, Ns, ka, x, z, tag, phase, inp, flow.X, depth, spec,
                        seed + 7919 + t, tag_id=k + 1)
                    if charge is not None:           # injected SDs start uncharged
                        charge = np.concatenate([charge, np.zeros(M.shape[0] - _n0)])
                    if hab is not None:              # injected aerosol carry no ice shape
                        hab = np.concatenate([hab, np.zeros((M.shape[0] - _n0, 3))])
                    if lem:                          # injected SDs lazy-init to cell mean (NaN)
                        _ne = M.shape[0] - _n0
                        eta_sd = np.concatenate([eta_sd, np.full(_ne, np.nan)])
                        w_sgs = np.concatenate([w_sgs, np.zeros(_ne)])

        # 1. solve streamfunction from vorticity, build velocities (CFL-clipped)
        psi = _poisson(-omega, flow.dx, flow.dz)
        flow.u, flow.w = _faces(psi, flow.dx, flow.dz)
        np.clip(flow.u, -v_max, v_max, out=flow.u)
        np.clip(flow.w, -v_max, v_max, out=flow.w)
        Cx = flow.u * dt / flow.dx
        if wind_shear != 0.0:                        # add the mean wind to scalar+vorticity advection
            Cx = Cx + U_z[None, :] * dt / flow.dx
            np.clip(Cx, -0.9, 0.9, out=Cx)           # keep the TOTAL Courant CFL-stable: the
            #   mean wind U(z) grows with height and can blow up advection in deep domains
            #   (e.g. congestus, Z=7 km). No-op for the tuned shallow shear case (|Cx|<0.2).
        Cz = flow.w * dt / flow.dz
        if periodic_x:                              # cell-centred velocities for droplets
            uc = 0.5 * (flow.u + np.roll(flow.u, -1, axis=0))
        else:
            uc = 0.5 * (flow.u[:-1, :] + flow.u[1:, :])
        if wind_shear != 0.0:                        # droplets ride the mean wind too
            uc = uc + U_z[None, :]
        wc = 0.5 * (flow.w[:, :-1] + flow.w[:, 1:])

        # 2. buoyancy and vorticity tendency. Anelastic references the LOCAL base theta0(z)
        # (correct over a deep layer); Boussinesq uses the single surface theta0 (unchanged).
        qc_grid = _qc_grid()
        theta_ref = theta_base[None, :] if anelastic else theta0
        b = g * ((theta - theta_base[None, :]) / theta_ref
                 + 0.608 * (qv - qv_base[None, :]) - qc_grid)
        # buoyancy limiter (sub-grid entrainment closure): real cloud buoyancy is
        # a few K; without it the undiluted theta' runs away (grid-scale moist
        # instability -> blow-up). Capping b keeps the convection realistic & stable.
        np.clip(b, -b_max, b_max, out=b)
        omega = _upwind(omega, Cx, Cz)
        # vorticity source is -db/dx for omega = du/dz - dw/dx (with lap(psi)=-omega):
        # a warm bubble (b>0 centre) then drives a CENTRAL updraft. (A + sign here
        # reverses the circulation -> spurious central downdraft, split cloud.)
        omega = omega + dt * (-_ddx_op(b, flow.dx) + nu * _lap_om(omega, flow.dx, flow.dz))
        # hard vorticity backstop: Poisson is linear, so bounding omega bounds the
        # velocities -> bounds lifting -> no condensation runaway (the moist
        # instability cannot accumulate omega unboundedly).
        np.clip(omega, -omega_max, omega_max, out=omega)
        if sponge_frac > 0.0:                          # absorb gravity waves near the lid
            omega -= dt * damp_z[None, :] * omega

        # 3. advect droplets and scalars (+ small eddy diffusion for stability)
        if periodic_x:
            x, z = _advect_periodic(x, z, uc, wc, flow.X, flow.Z, flow.dx, flow.dz, dt)
        else:
            x, z = flow.advect(x, z, dt)
        if lem:                                    # subgrid vertical transport (SAM micro_sgs_uvw):
            w_sgs, _dz_sgs = _lem.sgs_velocity_step(w_sgs, lem_eps, flow.dz, dt, _lem_rng)
            z = np.clip(z + _dz_sgs, 0.0, flow.Z)  # carries each SD's eta memory between cells
        if sediment:                               # drops fall at their terminal speed
            iz = np.clip((z / flow.dz).astype(np.int64), 0, flow.Nz - 1)
            rho_a = P_col[iz] / (r_a * T_col[iz])
            vt = _fall_speeds(M, A, rho_a, P_col[iz], T_col[iz], phase)
            if hab is not None:                    # habit ice falls per its Böhm aspect-ratio
                ic = np.flatnonzero((phase == 1) & (A > 0.0) & (M > 0.0) & (hab[:, 0] > 0.0))
                if ic.size:
                    vt[ic] = _hab.boehm_fallspeed(M[ic] / A[ic], hab[ic, 0], hab[ic, 1],
                                                  hab[ic, 2], T_col[iz[ic]], rho_a[ic])
            z = z - vt * dt
            keep = z > 0.0                          # drops reaching the ground precipitate OUT
            if not keep.all():                     # (else they pile up -> q_c/buoyancy spike -> blow-up)
                surf_precip += float(M[~keep].sum())
                M, A, Ns, ka = M[keep], A[keep], Ns[keep], ka[keep]
                x, z, tag, phase, inp = x[keep], z[keep], tag[keep], phase[keep], inp[keep]
                if charge is not None:               # charge leaves with the precipitation
                    charge_to_ground += float(charge[~keep].sum())
                    charge = charge[keep]
                if hab is not None:
                    hab = hab[keep]
                if lem:                              # LEM state follows its super-droplet
                    eta_sd, w_sgs = eta_sd[keep], w_sgs[keep]
        if anelastic:
            # anelastic scalar transport conserves rho0*scalar: d(rho0 s)/dt + div(rho0 s V)=0
            # -> Ds/Dt=0. Plain flux-form div(sV) would instead give Ds/Dt=-s div(V), spuriously
            # cooling rising air (div V = w/H > 0) and killing convective growth. rho0=rho0(z)
            # is constant in time, so advect rho0*s and divide back out.
            theta = _mpdata(theta * rho0_c[None, :], Cx, Cz) / rho0_c[None, :]
            qv = _mpdata(qv * rho0_c[None, :], Cx, Cz) / rho0_c[None, :]
        else:
            theta = _mpdata(theta, Cx, Cz)
            qv = _mpdata(qv, Cx, Cz)
        # scalar diffusion is DECOUPLED from the vorticity viscosity nu: with the
        # omega/buoyancy limiters providing stability, the cloud (q_c/theta) field
        # can keep MPDATA's low diffusion (small nu_scalar) and stay crisp — the
        # large nu was what smeared it. nu still damps vorticity.
        if nu_scalar > 0.0:
            kdiff = nu_scalar * dt
            _lap_s = _laplacian_neumann_px if periodic_x else _laplacian_neumann
            theta = theta + kdiff * _lap_s(theta, flow.dx, flow.dz)
            qv = np.maximum(qv + kdiff * _lap_s(qv, flow.dx, flow.dz), 0.0)
        if sponge_frac > 0.0:                          # relax theta perturbation to base aloft
            # absorb the gravity wave's thermal (theta) signature, but DO NOT relax qv: the
            # base-state qv aloft can be ice-supersaturated, so forcing qv back up would be a
            # perpetual deposition source feeding spurious, constantly-precipitating
            # stratospheric ice. Damping omega + theta absorbs the wave; qv is left alone.
            theta -= dt * damp_z[None, :] * (theta - theta_base[None, :])

        # cloud-top radiative cooling (the stratocumulus / Rayleigh-Benard driver). LWP uses
        # the height-varying base density rho0_c(z) in anelastic mode (scalar surface rho0 in
        # Boussinesq) so water paths aloft aren't overweighted by surface density.
        if rad_cool is not None:
            rho_lw = rho0_c[None, :] if anelastic else rho0
            theta = theta + dt * _radiative_cooling(
                _qc_grid(), rho_lw, flow.dz, rad_cool["F0"], rad_cool["F1"], rad_cool["kappa"])

        # large-scale + surface forcing
        if forcing is not None:
            dthdz = np.zeros_like(theta); dthdz[:, :-1] = (theta[:, 1:] - theta[:, :-1]) / flow.dz
            dqdz = np.zeros_like(qv); dqdz[:, :-1] = (qv[:, 1:] - qv[:, :-1]) / flow.dz
            theta += dt * (tls_col[None, :] - wls_col[None, :] * dthdz)
            qv += dt * (qls_col[None, :] - wls_col[None, :] * dqdz)
            # DIURNAL surface heating: scale the surface sensible+latent fluxes by the
            # solar cycle (the large-scale tendencies stay constant). solar=0 at
            # sunrise, peaks at quarter-period, back to 0 at sunset, then night (no
            # heating) -> the boundary layer deepens and continental cumulus pop up in
            # the afternoon, then collapse as the ground stops heating in the evening.
            solar = 1.0
            if diurnal_period is not None:
                solar = max(0.0, math.sin(2.0 * pi * (t * dt) / diurnal_period))
            theta[:, 0] += sfc_th * solar * dt
            qv[:, 0] += sfc_qv * solar * dt
            qv = np.maximum(qv, 0.0)

        # surface radiative cooling (the RADIATION-FOG driver): the ground radiates
        # to the clear night sky and chills the near-surface air; in a near-saturated
        # stable boundary layer it reaches dew point and fog condenses AT the surface
        # and grows upward — the mirror image of stratocumulus cloud-top cooling.
        # Applied over the lowest ~40 m (the layer the ground cools), strongest at the
        # ground and tapering with height.
        if surface_cool != 0.0:
            n_cool = max(1, int(40.0 / flow.dz))
            taper = np.linspace(1.0, 0.0, n_cool + 1)[:n_cool]   # 1 at ground -> 0 aloft
            theta[:, :n_cool] += surface_cool * dt * taper[None, :]

        # 3b. immersion freezing: supercooled liquid super-droplets may turn to ice.
        # ABIFM (default) freezes INP-bearing drops at a water-activity-based, species-
        # and INP-surface-area-dependent rate; Bigg is the simple one-knob alternative.
        # Freezing releases latent heat of fusion into theta.
        ix, iz, cidx = _cell_index(flow, x, z)
        if ice:
            T_cell = theta * (P_col[None, :] / p0) ** kap          # (Nx,Nz)
            frozen_mass = np.zeros(Nx * Nz)
            if freezing_mode == "abifm":
                esw = _esatw_v(T_cell).ravel(); esi = _esati_v(T_cell).ravel()
                _abifm_freeze(M, A, phase, inp, cidx, esw, esi, T_cell.ravel(),
                              dt, c_abifm, m_abifm, frozen_mass)
            else:                                                  # "bigg"
                _bigg_freeze(M, A, phase, cidx, T_cell.ravel()[cidx], dt,
                             a_bigg, B_bigg, frozen_mass)
            if homogeneous:                                        # extra pathway: deep cold
                _homogeneous_freeze(M, A, phase, cidx, T_cell.ravel(), dt, frozen_mass)
            if "freeze" in _acc: _acc["freeze"] += frozen_mass
            dq_f = (frozen_mass / air_mass_cell).reshape(Nx, Nz)
            theta += (l_f / cp * dq_f) * (p0 / P_col[None, :]) ** kap

            # melting: ice that has fallen/mixed into air warmer than 0 C reverts to
            # liquid, absorbing l_f (a heat sink -- the opposite sign of freezing).
            if melt:
                melted_mass = np.zeros(Nx * Nz)
                _melt(M, A, phase, cidx, T_cell.ravel(), melted_mass)
                if "melt" in _acc: _acc["melt"] += melted_mass
                dq_m = (melted_mass / air_mass_cell).reshape(Nx, Nz)
                theta -= (l_f / cp * dq_m) * (p0 / P_col[None, :]) ** kap

            if hab is not None:                    # ice-habit: clear melted shape, seed new ice
                _hab.reset_melted_shape(hab, phase)
                _hab.init_ice_shape(hab, M, A, phase)

        # 4. condensation (two-way), sub-cycled
        if ihmd > 0.0:                                  # cloud liquid per cell before
            M0_cell = np.zeros(Nx * Nz)
            np.add.at(M0_cell, cidx, M)
        if lem:
            # SAM-LCM mixing ONCE per step: nudge each SD's prognostic supersaturation toward
            # its cell mean (tau memory) and homogenise within the box (substep-stable
            # diffusion + triplet rearrangement). The SD's LEM anomaly (eta_sd - cell mean) is
            # then held while the resolved supersaturation depletes across the substeps.
            _Tss, _ss0, _, _, _ = _thermo()
            _lem.nudge_and_mix(eta_sd, w_sgs, cidx, _ss0.ravel(), _Tss.ravel(), Nx * Nz,
                               flow.dz, lem_eps, lem_tau, dt, _lem_rng)
            _eta_anom = eta_sd - _ss0.ravel()[cidx]     # memory-carrying supersat anomaly
        for _ in range(n_sub):
            T, supersat, G, r0, afac = _thermo()
            dM_cell = np.zeros(Nx * Nz)
            if lem:                                     # each SD grows against cell s + its anomaly
                s_sd = supersat.ravel()[cidx] + _eta_anom
                _cond_local_sd(M, A, Ns, ka, cidx, s_sd, G.ravel(),
                               r0.ravel(), afac.ravel(), dt_sub,
                               switch_kappa_koehler, True, True, dM_cell, phase)
            else:
                _cond_local(M, A, Ns, ka, cidx, supersat.ravel(), G.ravel(),
                            r0.ravel(), afac.ravel(), dt_sub,
                            switch_kappa_koehler, True, True, dM_cell, phase)
            if "cond" in _acc: _acc["cond"] += np.maximum(dM_cell, 0.0)
            if "evap" in _acc: _acc["evap"] += -np.minimum(dM_cell, 0.0)
            dq = (dM_cell / air_mass_cell).reshape(Nx, Nz)
            qv -= dq
            theta += (l_v / cp * dq) * (p0 / P_col[None, :]) ** kap
            if ice:
                # ice deposition against over-ice supersaturation, drawing from the
                # SAME (now liquid-updated) cell qv -> WBF emerges: at water
                # saturation S_ice>0 so ice grows; the vapour it removes pushes the
                # liquid below saturation so drops evaporate next substep.
                e_si = _esati_v(T)
                e_a = qv * P_col[None, :] / (qv + r_a / rv)
                S_ice = (e_a / e_si - 1.0).ravel()
                # G_ice per cell (vectorised; same formula as ice_microphysics.g_ice,
                # r dr/dt = G_ice*S_ice convention with rho_ice/l_s/e_si)
                Kc = 7.94048e-5 * T + 0.00227011
                Dc = 0.211e-4 * (T / 273.15) ** 1.94 * (101325.0 / P_col[None, :])
                Fd = rv * T / (e_si * Dc)
                Fk = (l_s / (rv * T) - 1.0) * l_s / (Kc * T)
                G_ice_c = (1.0 / (rho_ice * (Fd + Fk))).ravel()
                dM_cell_i = np.zeros(Nx * Nz)
                if hab is not None:                    # shape-resolving capacitance growth
                    P_flat = np.tile(P_col, Nx)
                    rho_air_flat = (P_col[None, :] / (r_a * T)).ravel()
                    eswi_flat = (_esatw_v(T) / e_si).ravel()
                    dM_ice = _hab.deposit_habit(M, A, phase, cidx, hab, T.ravel(), P_flat,
                                                S_ice, rho_air_flat, eswi_flat, dt_sub)
                else:                                  # spherical r^2-law
                    dM_ice = np.zeros(M.shape[0])
                    _ice_deposition(M, A, phase, cidx, S_ice, G_ice_c, dt_sub, dM_ice)
                _scatter_dM_ice(dM_ice, cidx, dM_cell_i)
                if "dep" in _acc: _acc["dep"] += np.maximum(dM_cell_i, 0.0)
                if "sub" in _acc: _acc["sub"] += -np.minimum(dM_cell_i, 0.0)
                dq_i = (dM_cell_i / air_mass_cell).reshape(Nx, Nz)
                qv -= dq_i
                theta += (l_s / cp * dq_i) * (p0 / P_col[None, :]) ** kap
        if lem:                                        # carry the absolute prognostic forward
            _, _ssf, _, _, _ = _thermo()               # (anomaly preserved over the depletion)
            eta_sd = _ssf.ravel()[cidx] + _eta_anom
        # entrainment-mixing closure: split this step's evaporation between
        # homogeneous (all drops shrink) and inhomogeneous (fewer drops survive)
        if ihmd > 0.0:
            A = _ihmd_mix(M, A, cidx, Nx * Nz, M0_cell, ihmd)

        # 5. collision
        if collisions:
            rho = P_col[None, :] / (r_a * (theta * (P_col[None, :] / p0) ** kap))
            T = theta * (P_col[None, :] / p0) ** kap
            rimed_out = np.zeros(Nx * Nz) if ice else None
            _res = _collide_cells(
                flow, M, A, Ns, ka, x, z, ix, iz, dt, rho, P_col, T,
                switch_TICE, eps, depth, tag=tag, phase=phase, inp=inp,
                rimed_out=rimed_out, charge=charge, hab=hab,
                # opt-in parallel collision: per-cell splitmix64 streams salted by
                # (seed, step) so a fixed seed reproduces exactly, at any thread count
                parallel=collide_parallel, salt=(seed * 1000003 + t) & 0x7FFFFFFFFFFFFFFF)
            M, A, Ns, ka, x, z, tag, phase, inp = _res[:9]   # tag/phase/inp always present
            _i = 9
            if charge is not None:
                charge = _res[_i]; _i += 1
            if hab is not None:
                hab = _res[_i]; _i += 1
                _hab.init_ice_shape(hab, M, A, phase)        # seed shape for newly-rimed ice
            if ice and rimed_out is not None:        # riming/freezing-on-contact releases l_f
                if "rime" in _acc: _acc["rime"] += rimed_out
                dq_rime = (rimed_out / air_mass_cell).reshape(Nx, Nz)
                theta += (l_f / cp * dq_rime) * (p0 / P_col[None, :]) ** kap
                # Hallett-Mossop: rime accreted at -3..-8 C throws off secondary ice
                # splinters. Acts on the rimed mass (post-collision arrays are pruned, so
                # recompute the cell index); merges splinter NUMBER onto existing ice SDs
                # (no new super-droplets) -> mass-conserving, count-fixed. No l_f here: the
                # splinter mass is borrowed from existing ice, not freshly frozen.
                if hallett_mossop:
                    _, _, cidx_hm = _cell_index(flow, x, z)
                    T_hm = (theta * (P_col[None, :] / p0) ** kap).ravel()
                    splinters = np.zeros(Nx * Nz)
                    _hallett_mossop(M, A, phase, cidx_hm, rimed_out, T_hm, splinters)
                    if "sip" in _acc: _acc["sip"] += splinters

                # toy electrification: separate charge onto riming graupel/crystal ice,
                # then diagnose the field and fire a discharge if it breaks down. Pure
                # diagnostic -> no feedback on theta/qv/dynamics.
                if charge is not None:
                    _, _, cidx_e = _cell_index(flow, x, z)
                    T_e = (theta * (P_col[None, :] / p0) ** kap)
                    sc = np.zeros(Nx * Nz)            # supercooled-liquid mass per cell
                    np.add.at(sc, cidx_e, np.where((phase == 0) & (T_e.ravel()[cidx_e] < 273.15),
                                                   M, 0.0))
                    qsc = sc / air_mass_cell
                    _elec.deposit_charge(charge, M, A, phase, cidx_e, Nx * Nz,
                                         T_e.ravel(), qsc, q_sc_min, q_rev_T, dt,
                                         flow.dx * flow.dz * depth, Nx, Nz, flow.dz,
                                         charge_eff=charge_eff)
                    if (t + 1) % flash_every == 0:
                        rho_q = _elec.charge_density(charge, cidx_e, Nx, Nz,
                                                     flow.dx * flow.dz * depth)
                        phi = _elec.solve_potential(rho_q, flow.dx, flow.dz, periodic_x)
                        _, _, Emag = _elec.efield(phi, flow.dx, flow.dz, periodic_x)
                        # hysteresis: how close is the peak field to LOCAL breakdown?
                        _ratio = float((Emag / _elec.breakdown_field(rho, rho0,
                                                                     E_breakdown)).max())
                        fl = None
                        if _flash_armed and _ratio > 1.0:
                            # MULTI-STROKE discharge: a real flash keeps stroking (3-5
                            # return strokes) until the driving field collapses below the
                            # sustaining level -- one stroke's channel volume rarely drains
                            # the whole reservoir. Stroke until the peak field is below
                            # flash_rearm * E_crit (or a safety cap of 6 strokes).
                            for _stroke in range(6):
                                s_fl = _elec.flash(charge, cidx_e, Nx, Nz, phi, Emag,
                                                   rho, rho0, flow.dx, flow.dz, _elec_rng,
                                                   E_breakdown=E_breakdown,
                                                   flash_neutralize=flash_neutralize,
                                                   flash_radius=flash_radius)
                                if s_fl is None:
                                    break
                                fl = s_fl if fl is None else fl
                                fl["strokes"] = fl.get("strokes", 0) + 1
                                fl["q_neutralized"] += (s_fl["q_neutralized"]
                                                        if s_fl is not fl else 0.0)
                                rho_q = _elec.charge_density(charge, cidx_e, Nx, Nz,
                                                             flow.dx * flow.dz * depth)
                                phi = _elec.solve_potential(rho_q, flow.dx, flow.dz,
                                                            periodic_x)
                                _, _, Emag = _elec.efield(phi, flow.dx, flow.dz, periodic_x)
                                _ratio = float((Emag / _elec.breakdown_field(
                                    rho, rho0, E_breakdown)).max())
                                if _ratio < flash_rearm:
                                    break              # field collapsed -> flash over
                            if fl is not None:
                                _flash_armed = False   # storm must recharge before the next
                        elif not _flash_armed and _ratio < flash_rearm:
                            _flash_armed = True        # recharged enough to arm again
                        _efield_hist.append((t + 1, float(Emag.max()), 1 if fl is not None else 0))
                        if fl is not None:
                            _flashes_acc.append(dict(step=t + 1, **fl))

        if _diag:
            _acc_dt += dt                              # elapsed time over the output interval

        # 6. diagnostics
        if (t + 1) % collect_every == 0 or t == nt - 1:
            _, _, cidx = _cell_index(flow, x, z)
            r_um = np.where(A > 0,
                            (M / (A * 4.0 / 3.0 * pi *
                                  np.where(phase == 1, rho_ice, rho_liq))) ** (1.0 / 3.0),
                            0.0) * 1e6
            liq = np.zeros(Nx * Nz)
            np.add.at(liq, cidx, np.where(phase == 0, M, 0.0))
            qc = (liq / air_mass_cell).reshape(Nx, Nz) * 1e3
            wc = 0.5 * (flow.w[:, :-1] + flow.w[:, 1:])
            _, ss, *_ = _thermo()
            frame = dict(step=t + 1, x=x.copy(), z=z.copy(), r_um=r_um,
                         A=A.copy(), tag=tag.copy(), qc=qc, supersat=ss,
                         theta=theta.copy(), qv=qv.copy(),
                         u=uc.copy(), w=wc.copy(),       # cell-centred wind (incl. mean) for quiver
                         surf_precip=surf_precip)
            if ice:                                # split liquid/ice only when relevant
                ice_m = np.zeros(Nx * Nz)
                np.add.at(ice_m, cidx, np.where(phase == 1, M, 0.0))
                frame["q_liquid"] = qc
                frame["q_ice"] = (ice_m / air_mass_cell).reshape(Nx, Nz) * 1e3
                frame["phase"] = phase.copy()
            if lem:                                    # per-SD prognostic supersaturation
                frame["eta_sd"] = eta_sd.copy()
                _, _ssd, _, _, _ = _thermo()
                frame["eta_anom"] = (eta_sd - _ssd.ravel()[cidx]).copy()
            if hab is not None:                        # ice habit: axes + aspect ratio phi
                frame["a_axis"] = hab[:, 0].copy()
                frame["c_axis"] = hab[:, 1].copy()
                frame["rho_app"] = hab[:, 2].copy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    frame["phi"] = np.where(hab[:, 0] > 0, hab[:, 1] / hab[:, 0], 0.0)
            if _diag:                                  # process rates over the interval
                vol_cm3 = flow.dx * flow.dz * depth * 1e6
                interval = max(_acc_dt, dt)
                rates = {}
                for k, acc in _acc.items():
                    if k == "sip":                     # number rate -> 1/cm3/s
                        rates[k] = (acc / vol_cm3 / interval).reshape(Nx, Nz)
                    else:                              # mass rate -> g/kg/s
                        rates[k] = (acc / air_mass_cell * 1e3 / interval).reshape(Nx, Nz)
                    acc[:] = 0.0
                frame["rates"] = rates
                _acc_dt = 0.0
            if charge is not None:                     # electrification diagnostics
                frame["charge"] = charge.copy()
                cq = np.zeros(Nx * Nz); np.add.at(cq, cidx, charge)
                frame["charge_density"] = (cq / (flow.dx * flow.dz * depth)).reshape(Nx, Nz)
                frame["flashes"] = _flashes_acc        # flash channels since last frame
                _flashes_acc = []
            frames.append(frame)
            # optional live hook: lets a front-end render frames AS they are computed
            # (no-op unless a callback is passed, so the numerics are unchanged).
            if on_frame is not None:
                on_frame(t + 1, nt, frames[-1], flow)

    return dict(flow=flow, x=x, z=z, M=M, A=A, tag=tag, phase=phase, inp=inp,
                theta=theta, qv=qv, omega=omega, frames=frames, P_col=P_col, T_col=T_col,
                surf_precip=surf_precip, dt=dt, depth=depth, ice=ice,
                anelastic=anelastic, air_mass_cell=air_mass_cell, sounding=sounding,
                charge=charge, charge_to_ground=charge_to_ground,
                efield_history=np.array(_efield_hist) if _efield_hist else np.empty((0, 3)),
                hab=hab)
