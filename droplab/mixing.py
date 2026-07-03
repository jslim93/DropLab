"""Entrainment mixing for droplab (warm cloud).

Mixing runs BEFORE condensation each timestep. The Inhomogeneous Mixing Degree
(IHMD, Lim & Hoffmann 2023) controls how entrainment-driven evaporation is split
between homogeneous (all droplets shrink, number conserved) and inhomogeneous
(a subset evaporates, survivors keep size) limits, satisfying exactly:

    N_c / N_{c,0} = (q_c / q_{c,0}) ** IHMD

Closed form per step with entrained fraction frac in [0,1):
    M <- M * (1 - frac)            # total super-droplet liquid mass
    A <- A * (1 - frac) ** IHMD    # multiplicity (droplet number)
"""
import numpy as np

from droplab.parameters import p0, r_a, cp, rv, l_v


def redistribute_droplets(particles_list, ihmd, frac):
    """Apply IHMD redistribution to cloud super-droplets in place.

    Returns the total evaporated liquid mass (sum of M lost), which the caller
    returns to the vapor field.
    """
    if frac <= 0.0:
        return 0.0
    keep_mass = 1.0 - frac
    keep_num = keep_mass ** ihmd
    evaporated = 0.0
    for p in particles_list:
        if p.A <= 0 or p.M <= 0:
            continue
        m_old = p.M
        p.M = p.M * keep_mass
        p.A = float(round(p.A * keep_num))  # remove whole droplets (integer multiplicity)
        evaporated += m_old - p.M
    return evaporated


def _interp(z, z_env, profile):
    return float(np.interp(z, z_env, profile))


def entrained_fraction(lambda_ent, w, dt, duration):
    """Total fraction of the parcel replaced by environmental air over the entrainment
    window. Each step relaxes toward the environment by frac_step = lambda*w*dt, so after
    N = duration/dt steps the cumulative fraction is 1 - (1 - frac_step)**N. This turns the
    per-metre rate lambda into the intuitive 'how much actually mixes in' number."""
    frac_step = min(max(lambda_ent * w * dt, 0.0), 0.999)
    n_steps = max(int(round(duration / dt)), 0) if dt > 0 else 0
    return 1.0 - (1.0 - frac_step) ** n_steps


class ParameterizedMixing:
    """Parameterized homogeneous/inhomogeneous entrainment mixing.

    The environment is defined ANALYTICALLY (no precomputed profile array): a
    potential-temperature lapse and a fixed relative humidity. At the parcel's height z,

        theta_env(z) = theta_init + lapse_rate * clip(z, z_init, z_top)
        T_env       = theta_env * (P/p0)^(Rd/cp)                     (P = parcel pressure)
        q_env       = rh_env * q_sat(T_env, P)

    lambda_ent  : fractional entrainment rate [1/m]; entrained fraction = lambda*w*dt.
    ihmd        : Inhomogeneous Mixing Degree in [0,1] (0 homogeneous, 1 inhomogeneous).
    theta_init  : environmental potential temperature at z_init [K].
    lapse_rate  : d(theta_env)/dz [K/m] (>0 stable, 0 neutral, <0 unstable).
    rh_env      : relative humidity of the entrained environmental air in [0,1].
    z_init/z_top: the height range over which the lapse applies (clamped outside).
    """

    def __init__(self, lambda_ent, ihmd, theta_init, lapse_rate, rh_env,
                 z_init=0.0, z_top=3000.0):
        from droplab.condensation import esatw   # local import: robust under %autoreload
        self._esatw = esatw
        self.lambda_ent = lambda_ent
        self.ihmd = ihmd
        self.theta_init = theta_init
        self.lapse_rate = lapse_rate
        self.rh_env = rh_env
        self.z_init = z_init
        self.z_top = z_top

    def apply(self, particles_list, T, q, P, z, dt, w, air_mass):
        frac = self.lambda_ent * w * dt
        if frac <= 0.0:
            return particles_list, T, q
        frac = min(frac, 0.999)
        # 1. Bulk entrainment: relax T, q toward the environment computed at this height.
        z_c = min(max(z, self.z_init), self.z_top)
        theta_env = self.theta_init + self.lapse_rate * z_c
        T_env = theta_env * (P / p0) ** (r_a / cp)
        es = self._esatw(T_env)
        q_env = self.rh_env * (r_a / rv) * es / (P - es)
        T = T + frac * (T_env - T)
        q = q + frac * (q_env - q)
        # 2. IHMD redistribution of cloud liquid; evaporated water -> vapor with
        #    latent cooling.
        evaporated = redistribute_droplets(particles_list, self.ihmd, frac)
        dq = evaporated / air_mass
        q = q + dq
        T = T - l_v * dq / cp
        return particles_list, T, q


class LEMMixing:
    """Linear Eddy Model mixing backend (same apply() signature as
    ParameterizedMixing). Not implemented this cycle — a 1D triplet-map domain
    with per-droplet supersaturation perturbation is Phase 3b.
    """

    def apply(self, particles_list, T, q, P, z, dt, w, air_mass):
        raise NotImplementedError(
            "LEMMixing is Phase 3b (1D triplet-map LEM). Use ParameterizedMixing for now."
        )
