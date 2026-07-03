"""Tests for the ACI diagnostics (droplab/aci.py).

Fast tests use pure functions + synthetic populations + a SHORT model run for the
Twomey directions (snapshot quantities, available pre-drizzle). The precipitation
susceptibility sign needs drizzle-development length, so its >0 check is an
opt-in slow test (set ACI_SLOW=1) — also exercised by examples/aci_diagnostics.py.
"""
import os
import types

import numpy as np
import pytest

from droplab.aci import (cloud_droplet_number, rain_water_path, aci_susceptibility,
                       cloud_radiative_effect, erfaci, make_runner, lwp_susceptibility,
                       erfaci_decomposition, cloud_albedo_direct, cloud_albedo_diffuse,
                       diffusion_brightening)
from droplab.parameters import rho_liq, pi


def _flow_stub(Nx=2, dx=100.0, dz=50.0):
    return types.SimpleNamespace(Nx=Nx, dx=dx, dz=dz)


def _drop(r_um, A):
    """Super-droplet mass M for multiplicity A of single droplets of radius r_um."""
    r = r_um * 1e-6
    return A * 4.0 / 3.0 * pi * rho_liq * r ** 3


# --- pure-function tests (no model run) ---

def test_cre_monotonic_and_cooling():
    cre = cloud_radiative_effect(np.array([0.1, 0.3, 0.6]))
    assert np.all(cre < 0)                       # clouds cool in SW
    assert cre[0] > cre[1] > cre[2]              # more albedo -> more negative


def test_erfaci_negative_when_polluted_brighter():
    assert erfaci(albedo_pi=0.30, albedo_pd=0.55) < 0     # PD brighter -> cooling
    assert erfaci(albedo_pi=0.55, albedo_pd=0.30) > 0     # opposite sign sanity


def test_cloud_droplet_number_excludes_haze_and_is_positive():
    flow = _flow_stub()
    # haze (0.3 um), cloud (8 um), rain (60 um) — only cloud+rain are "activated"
    A = np.array([1.0e8, 1.0e8, 1.0e6])
    M = np.array([_drop(0.3, A[0]), _drop(8.0, A[1]), _drop(60.0, A[2])])
    x = np.array([10.0, 10.0, 10.0]); z = np.array([400.0, 450.0, 500.0])
    out = cloud_droplet_number(M, A, x, z, flow)
    assert out["nd_mean"] > 0.0
    # exclude the haze drop -> activated count = A[cloud]+A[rain], not A[haze]
    assert out["nd_mean"] < (A.sum() / 1e6)     # less than if haze were counted


def test_rain_water_path_only_counts_rain():
    flow = _flow_stub()
    A = np.array([1.0e8, 1.0e6])
    M = np.array([_drop(8.0, A[0]), _drop(60.0, A[1])])     # cloud drop + rain drop
    x = np.array([10.0, 10.0]); z = np.array([400.0, 400.0])
    rwp_all = rain_water_path(M, A, x, z, flow)["rwp_mean"]
    rwp_cloudonly = rain_water_path(M[:1], A[:1], x[:1], z[:1], flow)["rwp_mean"]
    assert rwp_all > 0.0                          # the 60 um drop counts
    assert rwp_cloudonly == 0.0                   # the 8 um drop does not


def test_susceptibility_fit_recovers_known_slopes():
    # synthetic runner: N_d~Na^0.8, r_eff~Na^-0.3, albedo up, precip~Na^-1.5, LWP~Na^0.5
    def runner(N):
        return (N ** 0.8, 1e-5 * N ** -0.3, 0.2 + 0.1 * np.log10(N),
                1e-3 * N ** -1.5, 0.05 * N ** 0.5)
    res = aci_susceptibility([50, 100, 200, 400], runner)
    assert abs(res["ACI_N"] - 0.8) < 1e-6
    assert abs(res["ACI_r"] - 0.3) < 1e-6
    assert abs(res["S_pop"] - 1.5) < 1e-6
    assert abs(res["ACI_L"] - 0.625) < 1e-6      # dlnLWP/dlnNd = 0.5/0.8


def _out(radii_um, A_each, col_x=10.0):
    """Synthetic run output (1 column) for the ERFaci decomposition."""
    flow = _flow_stub(Nx=1, dx=100.0, dz=50.0)
    M = np.array([_drop(r, a) for r, a in zip(radii_um, A_each)])
    A = np.array(A_each, float)
    x = np.full(len(A), col_x); z = np.full(len(A), 400.0)
    return dict(M=M, A=A, x=x, z=z, flow=flow)


def test_erfaci_decomposition_additive_and_isolates_terms():
    # Case 1: same LWP, smaller polluted drops -> pure Twomey, ZERO adjustment
    Mpi = _drop(15.0, 1.0e8)                      # one column of 15 um drops
    A_pd = Mpi / _drop(8.0, 1.0)                  # 8 um drops with the SAME total mass
    d = erfaci_decomposition(_out([15.0], [1.0e8]), _out([8.0], [A_pd]))
    assert abs(d["ERFaci_total"] - (d["RFaci_Twomey"] + d["ERFaci_adjustment"])) < 1e-9
    assert abs(d["ERFaci_adjustment"]) < 1e-9    # LWP unchanged -> no adjustment
    assert d["RFaci_Twomey"] < 0                  # smaller drops -> brighter -> cooling

    # Case 2: same reff, more polluted water (higher LWP) -> pure adjustment, ZERO Twomey
    d2 = erfaci_decomposition(_out([10.0], [1.0e8]), _out([10.0], [2.0e8]))
    assert abs(d2["RFaci_Twomey"]) < 1e-9        # reff unchanged -> no Twomey
    assert d2["ERFaci_adjustment"] < 0           # more LWP -> brighter -> cooling


def test_cloud_albedo_angle_relations():
    tau = np.array([2.0, 10.0, 40.0])
    overhead = cloud_albedo_direct(tau, 1.0)
    lowsun = cloud_albedo_direct(tau, 0.3)
    diffuse = cloud_albedo_diffuse(tau)
    assert np.all(lowsun > overhead)              # low sun -> longer path -> brighter
    assert np.all(diffuse > overhead)             # diffuse brighter than overhead-direct
    assert np.all(diffuse < lowsun)               # but darker than very low sun
    assert cloud_albedo_diffuse(np.array([0.0]))[0] == 0.0
    assert cloud_albedo_direct(np.array([1e4]), 0.5) > 0.99   # thick cloud -> ~1


def test_diffusion_brightening_sign_flips_with_sun():
    out = _out([8.0], [1.0e8])                    # one cloudy column (tau set by the drops)
    hi = diffusion_brightening(out, delta_f_diff=0.2, mu0=0.95)
    lo = diffusion_brightening(out, delta_f_diff=0.2, mu0=0.30)
    assert hi["d_albedo"] > 0 and hi["d_CRE"] < 0   # high sun: bonus brightening -> cooling
    assert lo["d_albedo"] < 0                        # low sun: reverses (diffuse darkens)


def test_lwp_susceptibility_sign_per_branch():
    Nd = np.array([5.0, 10.0, 20.0, 40.0])
    s_up, _ = lwp_susceptibility(Nd, Nd ** 0.5)    # ascending (suppression) branch
    s_dn, _ = lwp_susceptibility(Nd, Nd ** -0.4)   # descending (entrainment-drying) branch
    assert s_up > 0 and s_dn < 0


# --- short model run: Twomey directions (snapshot, pre-drizzle OK) ---

def test_twomey_directions_short_run():
    from droplab.soundings import DYCOMS, DYCOMS_RADIATION
    cfg = dict(Nx=28, Nz=28, X=2800.0, Z=1500.0, nt=120, dt=1.5, n_super=8000,
               sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
               collisions=True, switch_TICE=True, eps=0.01, sediment=True,
               nu=6.0, nu_scalar=1.5, pert_amp=0.1, seed=0)
    res = aci_susceptibility([50.0, 200.0], make_runner(depth=1.0, **cfg))
    assert 0.0 < res["ACI_N"] < 1.6              # activation efficiency (low-Nd inflates)
    assert 0.0 < res["ACI_r"] < 0.7              # Twomey radius susceptibility (~1/3)
    assert res["albedo"][-1] > res["albedo"][0]  # more aerosol -> brighter


# --- slow: precipitation suppression sign (validated fact; needs drizzle length) ---

@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("ACI_SLOW"), reason="set ACI_SLOW=1 (full-length, ~1 min)")
def test_precip_suppression_full_length():
    from droplab.soundings import DYCOMS, DYCOMS_RADIATION
    cfg = dict(Nx=40, Nz=40, X=4000.0, Z=1500.0, nt=1200, dt=1.5, n_super=18000,
               sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
               collisions=True, switch_TICE=True, eps=0.01, sediment=True,
               nu=6.0, nu_scalar=1.5, pert_amp=0.1, seed=0)
    res = aci_susceptibility([50.0, 100.0, 200.0, 400.0], make_runner(depth=1.0, **cfg))
    assert res["S_pop"] > 0.0                     # aerosol suppresses warm rain
