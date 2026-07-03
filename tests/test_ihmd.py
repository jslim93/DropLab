"""Inhomogeneous-mixing (IHMD) toggle gates.

IHMD splits entrainment evaporation between homogeneous (all droplets shrink, number
kept) and inhomogeneous (fewer droplets survive at their original size) mixing:
  ihmd = 0  must be a no-op (the model's default homogeneous behaviour, unchanged);
  ihmd = 1  must leave FEWER, LARGER droplets (lower N, bigger r_eff) at ~the same
            liquid water — the inhomogeneous-mixing signature.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")

from droplab.flow2d import Flow2D
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.climate_diag import column_optics
from droplab.soundings import DYCOMS, DYCOMS_RADIATION

CFG = dict(dt=1.0, Nx=48, Nz=32, X=2400.0, Z=1200.0, n_super=24000,
           sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True, pert_amp=0.1,
           nu=6, nu_scalar=1.5, collisions=True, switch_TICE=True, eps=0.01,
           sediment=True, collect_every=100000, seed=3, N_modes=(250.,))


def test_ihmd_zero_is_noop():
    """ihmd=0 must reproduce the default run bit-for-bit (no droplets removed)."""
    a = run_flow2d_dynamic(nt=400, **CFG)
    b = run_flow2d_dynamic(nt=400, ihmd=0.0, **CFG)
    assert a["A"].size == b["A"].size, "ihmd=0 changed the droplet count"
    assert np.array_equal(a["A"], b["A"]), "ihmd=0 changed multiplicities (not a no-op)"


def test_inhomogeneous_mixing_fewer_larger_drops():
    """ihmd=1 (inhomogeneous) must leave fewer total droplets and a larger effective
    radius than ihmd=0 (homogeneous), while liquid water is roughly preserved."""
    flow = Flow2D(X=CFG["X"], Z=CFG["Z"], Nx=CFG["Nx"], Nz=CFG["Nz"])
    homo = run_flow2d_dynamic(nt=700, ihmd=0.0, **CFG)
    inho = run_flow2d_dynamic(nt=700, ihmd=1.0, **CFG)
    oh = column_optics(homo["M"], homo["A"], homo["x"], homo["z"], flow)
    oi = column_optics(inho["M"], inho["A"], inho["x"], inho["z"], flow)
    assert inho["A"].sum() < homo["A"].sum(), "inhomogeneous mixing did not reduce droplet number"
    assert oi["reff_mean"] > oh["reff_mean"], "inhomogeneous mixing did not grow r_eff"
    assert abs(oi["lwp_mean"] - oh["lwp_mean"]) / oh["lwp_mean"] < 0.15, "LWP changed too much"
