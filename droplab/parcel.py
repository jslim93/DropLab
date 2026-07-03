import numpy as np
from droplab.parameters import *
from droplab.micro_particle import *
from droplab.condensation import *
from droplab.entrainment import *


def parcel_rho(P_parcel, T_parcel):
    from droplab.condensation import esatw
    p_env = P_parcel
    T_env = T_parcel
    theta_env = T_parcel * ( p0 / p_env )**( r_a / cp )
    e_s = esatw(T_parcel)
    
    rho_parcel = p_env / ( r_a * T_parcel ) #  Air density
    V_parcel   = PARCEL_AIR_MASS / rho_parcel # volume of the parcel for PARCEL_AIR_MASS of air
    air_mass_parcel = V_parcel * rho_parcel
    
    return(rho_parcel, V_parcel, air_mass_parcel) # (Assumed) air mass of parcel

def ascend_parcel(z_parcel, T_parcel,P_parcel,w_parcel,dt, time, max_z,theta_profiles,time_half_wave_parcel=1200.0, ascending_mode='linear', t_start_oscillation=800):
    # Computes values for the ascending parcel. Three ascending mode options are provided.
    # Users can change the half wavelength of the oscillation (time_half_wave_parcel (s)) and the oscillation start time (t_start_oscillation (s), only relevant for the 'in_cloud_oscillation' case)
    if ascending_mode=='linear':
        # Linear ascending
        if z_parcel < max_z: 
            dz = w_parcel * dt
            z_parcel   = z_parcel + dz
            T_parcel   = T_parcel - dz * g / cp
        #change environmental pressure
            theta_env  = get_interp1d_var(z_parcel,z_env,theta_profiles)
            T_env      = theta_env * (P_parcel / p0) ** (r_a / cp)
            P_parcel   = P_parcel - P_parcel * g * dz / ( r_a * T_env )
    elif ascending_mode=='sine':
        # Sinusoidal oscillation
        w_oscillate = w_parcel * np.pi / 2.0 * np.sin(np.pi * time / time_half_wave_parcel)
        dz = w_oscillate  * dt
        z_parcel = z_parcel + dz
        T_parcel = T_parcel - dz * g / cp
            
        #change environmental pressure
        theta_env  = get_interp1d_var(z_parcel,z_env,theta_profiles)
        T_env      = theta_env * (P_parcel / p0) ** (r_a / cp)
        P_parcel   = P_parcel - P_parcel * g * dz / ( r_a * T_env )

    elif ascending_mode=='in_cloud_oscillation':
        # The particle rises first linearly. After oscillation start time it starts to oscillate.
        phase = np.arccos(2/np.pi)
        if time < t_start_oscillation:
            dz = w_parcel * dt
            z_parcel   = z_parcel + dz
            T_parcel   = T_parcel - dz * g / cp
        else:
            w_oscillate = w_parcel * np.pi / 2.0 * np.cos(np.pi * (time-t_start_oscillation) / time_half_wave_parcel + phase)
            dz = w_oscillate  * dt
            z_parcel = z_parcel + dz
            T_parcel = T_parcel - dz * g / cp

        #change environmental pressure
        theta_env  = get_interp1d_var(z_parcel,z_env,theta_profiles)
        T_env      = theta_env * (P_parcel / p0) ** (r_a / cp)
        P_parcel   = P_parcel - P_parcel * g * dz / ( r_a * T_env )

    return z_parcel, T_parcel, P_parcel

#Functions to make environmental profiles for three different stability conditions
def create_env_profiles(T_init, qv_init, z_init, p_env, stability_condition, rh_env=0.2):
    """Environmental profiles the parcel entrains toward.

    Temperature: a potential-temperature lapse set by ``stability_condition``.
    Moisture: the environment is DRY air at a FIXED relative humidity ``rh_env``
    (default 0.2 = 20 %), so q_v,env(z) = rh_env * q_sat(T_env(z), P(z)). This makes the
    entrained air an explicit "20 % RH" choice rather than an opaque linear q_v profile.
    ``qv_init`` is accepted for signature compatibility but no longer used.
    """
    from droplab.condensation import esatw   # local import: robust under notebook %autoreload
    z_env = np.arange(z_init, 3001, 10) # vertical levels up to 3000m
    if stability_condition == 'Stable':
        lapse_rates = 5 / 1000 # +5 K/km (theta increases -> statically stable)
    elif stability_condition == 'Neutral':
        lapse_rates = 0           # 0 K/km
    elif stability_condition == 'Unstable':
        lapse_rates = -6.5 / 1000    # -6.5 K/km (theta decreases -> unstable)
    else:
        raise ValueError(f"Unknown stability condition: {stability_condition}")

    theta_init = T_init * ( p0 / p_env )**( r_a / cp )
    theta_profiles = theta_init + lapse_rates * z_env

    # Hydrostatic pressure P(z) from p_env at z_init, integrating d(exner)/dz = -g/(cp*theta)
    kap = r_a / cp
    dz = 10.0
    inv_theta = 1.0 / theta_profiles
    exner = (p_env / p0) ** kap - g / cp * (np.cumsum(inv_theta) * dz - 0.5 * inv_theta * dz)
    P_env = p0 * exner ** (1.0 / kap)
    T_env_profile = theta_profiles * exner
    # q_v at fixed RH: q_sat = (Rd/Rv) e_s / (P - e_s), then scale by rh_env
    es = np.array([esatw(t) for t in T_env_profile])
    qsat = (r_a / rv) * es / (P_env - es)
    qv_profiles = np.maximum(rh_env * qsat, 0.0)

    fig, ax1 = plt.subplots(figsize=(4, 6))
    ax1.plot(theta_profiles, z_env, c="r", lw=3, label=r"$ \Theta $ (K)")
    ax1.set_xlabel(r"$ \Theta $ (K)", color='r')
    ax1.tick_params(axis='x', colors='r') 
    
    # Create a second axis that shares the same y-axis
    ax2 = ax1.twiny()
    ax2.plot(qv_profiles*1e3, z_env, c="k", ls="--", lw=3, label=r"$q_{\mathrm{v}}$ (g/kg)")
    ax2.set_xlabel(r"$q_{\mathrm{v}}$ (g/kg)")
    ax2.tick_params(axis='x', colors='k')  
    
    plt.ylabel("z (m)")
    # Add a legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    plt.title(stability_condition + " condition")
    plt.show()

    
    return qv_profiles, theta_profiles, z_env
