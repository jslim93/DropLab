import math
import numpy as np
from droplab.parameters import rho_liq
from droplab.timestep_soa import dsd_spectrum


def test_dsd_spectrum_shape_and_number():
    A = np.full(100, 1.0e6)
    r = 20e-6
    M = A * 4.0 / 3.0 * math.pi * rho_liq * r ** 3
    air_mass = 1.0e6
    rho_parcel = 1.2  # kg/m^3, typical near-surface air density
    centers, num = dsd_spectrum(M, A, air_mass, rho_parcel, n_bins=40)
    assert centers.shape == (40,) and num.shape == (40,)
    assert np.all(num >= 0)
    # number density is per VOLUME (air_mass/rho_parcel), not per fixed dry-air mass
    V_parcel = air_mass / rho_parcel
    assert np.isclose(num.sum(), A.sum() / V_parcel / 1e6, rtol=1e-9)
    assert 5e-6 < centers[num.argmax()] < 60e-6


def test_dsd_spectrum_scales_with_parcel_density():
    """Same real droplet population, thinner air (higher altitude) -> the parcel
    occupies MORE volume for the same dry-air mass, so the cm^-3 concentration
    must be LOWER. A regression guard for the volume- vs mass-normalization bug."""
    A = np.full(100, 1.0e6)
    r = 20e-6
    M = A * 4.0 / 3.0 * math.pi * rho_liq * r ** 3
    air_mass = 1.0e6
    _, num_sfc = dsd_spectrum(M, A, air_mass, rho_parcel=1.2, n_bins=40)
    _, num_aloft = dsd_spectrum(M, A, air_mass, rho_parcel=0.6, n_bins=40)
    assert num_aloft.sum() < num_sfc.sum()
    assert np.isclose(num_aloft.sum() / num_sfc.sum(), 0.6 / 1.2, rtol=1e-9)
