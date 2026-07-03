"""Linear Eddy Model (LEM) subgrid turbulent mixing of super-droplets, ported faithfully
from the Fortran parcel model the reference Fortran parcel model (particle_model/source/mod_LEM.f90) (Krueger
1993; the LCM lineage Hoffmann et al.). The LEM represents the unresolved turbulent mixing
INSIDE a grid cell as a 1-D line of fluid elements undergoing stochastic triplet-map
rearrangement events plus molecular/eddy diffusion. Each super-droplet carries its own
local (T, q) environment, so subgrid supersaturation fluctuations BROADEN the droplet size
distribution -- the physical effect the LEM exists to capture.

Turbulence is PRESCRIBED by two parameters (the box-model approach, no SGS-TKE closure
needed): L_turb [m] (integral length scale) and eps [m^2/s^3] (dissipation rate). From
Krueger (1993) everything else follows:

    eta   = 6 * dz_lem                                  # LEM cutoff (>= physical Kolmogorov)
    D     = 0.1 * L_turb^(4/3) * eps^(1/3)              # turbulent diffusivity
    D_eta = max(D * (eta/L_turb)^(4/3), mu/rho)         # molecular diffusivity (eddy cutoff)
    lam   = 54/5 * D / L_turb^3 * (L_turb/eta)^(5/3)    # rearrangement-event rate per length

DropLab adaptation (documented simplification): the box model separates the air (T,q) field
from the droplets; here each SUPER-DROPLET carries its (T,q) and a 1-D LEM position, so the
triplet map permutes the super-droplets' order and diffusion mixes neighbours -- i.e. each
SD is its own small air+droplet parcel. The cell's mean (T,q) couples to DropLab's resolved
fields by forcing + relaxation. OPT-IN (lem=False -> bit-identical; golden gate).
"""
import numpy as np

from droplab.parameters import muelq, g, cp


def lem_coeffs(L_turb, eps, dz_lem, rho_air):
    """Krueger (1993) LEM coefficients from the prescribed turbulence (L_turb, eps).
    Returns (eta, D, D_eta, lam, tke). mod_LEM.f90 LEM_init L83-87."""
    eta = 6.0 * dz_lem
    D = 0.1 * L_turb ** (4.0 / 3.0) * eps ** (1.0 / 3.0)
    D_eta = max(D * (eta / L_turb) ** (4.0 / 3.0), muelq / rho_air)
    lam = 54.0 / 5.0 * D / L_turb ** 3 * (L_turb / eta) ** (5.0 / 3.0)
    tke = (D / (L_turb * 0.1)) ** 2
    return eta, D, D_eta, lam, tke


def eddy_length(eta, L_turb, z_lem, rng, z_length_max=20.0):
    """Draw an eddy segment length from the -5/3 PDF (Krueger 1993 Eq. 2.3), inverse-CDF
    with rejection to [eta, min(z_lem, z_length_max)]. mod_LEM.f90 L155-168."""
    hi = min(z_lem, z_length_max)
    if L_turb <= eta * 1.0001 or hi < eta:
        return eta
    for _ in range(1000):
        x = rng.random()
        zl = (eta ** (-5.0 / 3.0) - x * (eta ** (-5.0 / 3.0) - L_turb ** (-5.0 / 3.0))) ** (-3.0 / 5.0)
        if eta <= zl <= hi:
            return zl
    return eta


def triplet_indices(n_start, n_length, n_dom):
    """The triplet-map source index m for each target n in the segment (cyclic). Compresses
    the segment 3x and lays down 3 copies, the middle reversed. mod_LEM.f90 L185-224.
    Returns (targets, sources) 0-based into the n_dom-periodic line."""
    n1 = n_start + int(round(n_length / 3.0)) - 1
    n2 = n_start + int(round(2.0 * n_length / 3.0)) - 1
    tgt = np.arange(n_start, n_start + n_length)
    src = np.empty(n_length, dtype=np.int64)
    for i, n in enumerate(tgt):
        if n <= n1:
            m = 3 * (n - n_start) + n_start
        elif n <= n2:
            m = 2 * (n_length - 1) - 3 * (n - n_start) + n_start
        else:
            m = 3 * (n - n_start) - 2 * (n_length - 1) + n_start
        src[i] = m
    disp = tgt - src                          # un-modded displacement n-m (sums to 0)
    return tgt % n_dom, src % n_dom, disp


def triple_map(T, q, dz_lem, L_turb, eta, lam, dt, rng, supersat_fluct=True):
    """One triplet-map rearrangement ATTEMPT on the 1-D LEM line (the per-cell super-droplet
    (T,q) arrays, ordered by LEM position). Accepts with probability dt*lam*z_lem; on accept,
    permutes (T,q) by the triplet map and (if supersat_fluct) applies the adiabatic
    displacement heating -dz*g/cp. Mutates T, q in place; returns the per-element LEM
    displacement [in cells] (0 if no event). mod_LEM.f90 turb_rearrangement_LEM."""
    n_dom = T.shape[0]
    disp = np.zeros(n_dom)
    if n_dom <= 6:
        return disp
    z_lem = n_dom * dz_lem
    if rng.random() >= dt * lam * z_lem or eta > 20.0:
        return disp
    zl = eddy_length(eta, L_turb, z_lem, rng)
    n_length = int(np.floor(zl / dz_lem))
    n_length = min(max(int(round(n_length / 3.0)) * 3, 6), int(np.floor(n_dom / 3.0)) * 3)
    if n_length < 6:
        return disp
    n_start = int(min(max(rng.random(), 1e-8), 1 - 1e-8) * n_dom)   # 0-based start
    tgt, src, raw = triplet_indices(n_start, n_length, n_dom)
    Tn = T.copy(); qn = q.copy()
    Tn[tgt] = T[src]; qn[tgt] = q[src]
    # raw = target-source (un-modded, sums to 0): the adiabatic-fluctuation heating and
    # (in the driver) the distance the droplets are carried by the eddy
    if supersat_fluct:
        Tn[tgt] = Tn[tgt] - raw * dz_lem * g / cp
    T[:] = Tn; q[:] = qn
    disp[tgt] = raw * dz_lem
    return disp


def diffuse(T, q, D_eta, dz_lem, dt):
    """Explicit FTCS molecular/eddy diffusion of (T,q) on the periodic 1-D LEM line.
    mod_LEM.f90 mol_diffusion_LEM (cyclic branch). Mutates T, q. Conserves the sum."""
    Tn = T + dt * D_eta * (np.roll(T, -1) - 2.0 * T + np.roll(T, 1)) / dz_lem ** 2
    qn = q + dt * D_eta * (np.roll(q, -1) - 2.0 * q + np.roll(q, 1)) / dz_lem ** 2
    T[:] = Tn; q[:] = qn


def sgs_velocity(w_sgs, D_eta, eta, dt, rng):
    """AR-1 (Langevin) subgrid vertical velocity per super-droplet. mod_LEM.f90
    sgs_velocities_LEM L385-389. Returns the updated w_sgs (m/s)."""
    tke_eta = (D_eta / (eta * 0.1)) ** 2
    sigma_w = np.sqrt(tke_eta)
    tau_L = D_eta / max(tke_eta, 1e-30)
    RL = np.exp(-dt / tau_L)
    return RL * w_sgs + np.sqrt(1.0 - RL ** 2) * sigma_w * rng.standard_normal(w_sgs.shape)
