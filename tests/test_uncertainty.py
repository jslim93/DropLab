"""Tests for the stochastic-microphysics demonstrator (droplab/uncertainty.py)."""
import numpy as np

from droplab.uncertainty import rain_onset, seed_ensemble, spread_vs_resolution

# small, fast config near the rain threshold
CFG = dict(RH=0.98, w=1.0, N_raw=(60.0,), mu_um=(0.08,), sig=(1.6,),
           kappa=0.6, collisions=True, dt=1.0)


def test_rain_onset_detects_threshold_crossing():
    out = {40: dict(qr=0.0, z=100.0), 80: dict(qr=0.02, z=200.0), 120: dict(qr=0.1, z=300.0)}
    step, z = rain_onset(out, qr_thr=0.01)
    assert step == 80 and z == 200.0
    s2, z2 = rain_onset({40: dict(qr=0.0, z=100.0)}, qr_thr=0.01)
    assert np.isnan(s2) and np.isnan(z2)


def test_seeds_give_different_outcomes():
    ens = seed_ensemble(n_members=6, n_ptcl=500, nt=1200, **CFG)   # nt long enough to rain
    # the whole point: identical setup, different seeds -> non-zero spread
    assert ens["onset_z_std"] > 0.0 or np.nanstd(ens["final_qr"]) > 0.0
    assert 0.0 <= ens["rain_fraction"] <= 1.0


def test_spread_shrinks_with_resolution():
    res = spread_vs_resolution([200, 5000], n_members=6, nt=1200, **CFG)
    # numerical sampling component converges: more super-droplets -> less spread
    assert res["onset_z_std"][-1] <= res["onset_z_std"][0]
