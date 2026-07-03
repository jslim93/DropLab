"""Gates for the buoyancy-driven (dynamic) 2D cumulus.

Guards the two hard-won fixes: the vorticity buoyancy-source SIGN (a warm bubble
must drive a CENTRAL updraft, not a central downdraft) and the stabilizers (no
blow-up). Also checks a cloud forms and total water is conserved.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")

from droplab.parameters import r_a
from droplab.flow2d_dynamic import run_flow2d_dynamic

CFG = dict(Nx=32, Nz=32, X=2000.0, Z=2000.0, n_super=16000, dt=0.5,
           RH0=0.97, z_bl=500.0, z_inv=1500.0, dtheta_inv=6.0,
           dtheta_bubble=0.5, bubble_z=400.0, bubble_r=300.0)


def test_buoyancy_drives_central_updraft():
    """A warm bubble must produce a CENTRAL updraft (w>0 in the middle). The
    wrong vorticity-source sign reverses this into a central downdraft."""
    out = run_flow2d_dynamic(nt=600, collisions=False, collect_every=10000, **CFG)
    w = out["frames"][-1]["w"]
    Nx = w.shape[0]
    w_center = w[Nx // 2 - 2:Nx // 2 + 2, :].max()
    w_edge = w[:3, :].max()
    assert w_center > 0.3, f"no central updraft (w_center={w_center:.2f}) — vorticity sign bug?"
    assert w_center > w_edge, "updraft not centered (sign/circulation wrong)"


def test_stable_no_blowup():
    """The buoyancy/vorticity limiters keep the moist instability from blowing up."""
    out = run_flow2d_dynamic(nt=800, collisions=True, switch_TICE=True, eps=0.05,
                             collect_every=10000, **CFG)
    for f in out["frames"]:
        assert np.isfinite(f["qc"]).all() and np.isfinite(f["w"]).all(), "blew up (nan)"
    assert np.abs(out["frames"][-1]["w"]).max() < 60.0, "velocity unbounded"


def test_cloud_forms():
    # buoyancy-driven ascent reaches saturation and condenses (well above the
    # ~1e-6 g/kg haze background); the small test box keeps the cloud thin.
    out = run_flow2d_dynamic(nt=700, collisions=False, collect_every=10000, **CFG)
    assert out["frames"][-1]["qc"].max() > 0.02, "no cloud formed"


def test_radiative_cooling_peaks_at_cloud_top():
    """Cloud-top LW cooling (the Sc driver) must concentrate the cooling at the
    cloud top, not in mid-cloud."""
    from droplab.flow2d_dynamic import _radiative_cooling
    Nx, Nz, dz = 8, 48, 25.0
    qc = np.zeros((Nx, Nz)); qc[:, 16:33] = 0.5e-3        # a cloud layer
    dth = _radiative_cooling(qc, 1.2, dz, 70.0, 22.0, 85.0)
    assert dth[0, 32] < -1e-4, "no cloud-top cooling"
    assert dth[0, 24] > dth[0, 32], "cooling not concentrated at the cloud top"


def test_periodic_x_stable_and_conserves():
    """The periodic-x domain (for the Sc deck / MCB) runs stably and conserves
    water (vapour + liquid + surface precip)."""
    T0, P0, depth = 289.0, 1.0e5, 1.0
    from droplab.flow2d import Flow2D
    from droplab.flow2d_driver import _base_state
    f = Flow2D(Nx=CFG["Nx"], Nz=CFG["Nz"], X=CFG["X"], Z=CFG["Z"])
    amc = (P0 / (r_a * T0)) * f.dx * f.dz * depth
    _, qv0, *_ = _base_state(f, T0, P0, CFG["RH0"], CFG["z_bl"], 0.2,
                             z_inv=CFG["z_inv"], dtheta_inv=CFG["dtheta_inv"],
                             gamma_theta=0.004)
    out = run_flow2d_dynamic(nt=400, collisions=True, periodic_x=True,
                             collect_every=10000, **CFG)
    for fr in out["frames"]:
        assert np.isfinite(fr["qc"]).all() and np.isfinite(fr["w"]).all(), "periodic blew up"
    W_fin = out["qv"].sum() + out["M"].sum() / amc + out["surf_precip"] / amc
    assert abs(W_fin - qv0.sum()) / qv0.sum() < 0.005, "water not conserved (periodic)"


def test_aerosol_injection_adds_ccn():
    """Aerosol seeding (the climate-intervention lever) must inject the requested
    super-droplets at the scheduled step and the requested aerosol NUMBER, without
    destabilising the run. This is the mechanism behind MCB and cloud-seeding."""
    spec = dict(t_inject=50.0, x_frac=(0.4, 0.6), z_lo=50.0, z_hi=400.0,
                N_cm3=300.0, r_um=0.1, kappa=1.2, n_super=2000)
    base = run_flow2d_dynamic(nt=200, collisions=True, periodic_x=True,
                              collect_every=10000, **CFG)
    seeded = run_flow2d_dynamic(nt=200, collisions=True, periodic_x=True,
                                collect_every=10000, seeding=spec, **CFG)
    # exactly n_super more computational droplets survive (none fall out in 200 steps)
    assert seeded["A"].size - base["A"].size == 2000, "injected super-droplet count wrong"
    # injected aerosol NUMBER matches N_cm3 * region volume (depth=1)
    depth = 1.0
    V_reg = (0.6 - 0.4) * CFG["X"] * (400.0 - 50.0) * depth
    N_expected = 300.0 * 1e6 * V_reg
    assert abs((seeded["A"].sum() - base["A"].sum()) - N_expected) / N_expected < 0.02, \
        "injected aerosol number does not match request"
    assert np.isfinite(seeded["qv"]).all() and np.isfinite(seeded["M"]).all(), "seeding blew up"


def test_seeded_droplets_are_tagged_and_tracked():
    """Injected droplets carry a tag (>0) so they can be drawn distinctly, and the
    tag rides along through sedimentation and collision (Lagrangian identity)."""
    spec = dict(t_inject=50.0, x_frac=(0.4, 0.6), z_lo=50.0, z_hi=400.0,
                N_cm3=300.0, r_um=0.1, kappa=1.2, n_super=2000)
    out = run_flow2d_dynamic(nt=300, collisions=True, periodic_x=True,
                             collect_every=50, seeding=spec, **CFG)
    assert "tag" in out and out["tag"].shape == out["A"].shape, "tag not aligned with droplets"
    n_seeded = int((out["tag"] > 0).sum())
    assert 0 < n_seeded <= 2000, f"seeded tag count off ({n_seeded})"
    # the tag survives in the captured frames too (for the side-by-side animation)
    last = out["frames"][-1]
    assert "tag" in last and last["tag"].shape == last["A"].shape
    assert (last["tag"] > 0).any(), "no tagged droplets survived to the last frame"


def test_total_water_conserved():
    """Vapour + liquid + surface precipitation is conserved (forcing off). The
    bubble perturbs theta only, and the initial liquid is negligible haze, so the
    final total water must match the initial vapour to within that haze."""
    T0, P0, depth = 289.0, 1.0e5, 1.0
    from droplab.flow2d import Flow2D
    from droplab.flow2d_driver import _base_state
    f = Flow2D(Nx=CFG["Nx"], Nz=CFG["Nz"], X=CFG["X"], Z=CFG["Z"])
    amc = (P0 / (r_a * T0)) * f.dx * f.dz * depth
    _, qv0, *_ = _base_state(f, T0, P0, CFG["RH0"], CFG["z_bl"], 0.2,
                             z_inv=CFG["z_inv"], dtheta_inv=CFG["dtheta_inv"],
                             gamma_theta=0.004)
    W0 = qv0.sum()                                  # initial vapour (haze liquid ~ 0)
    out = run_flow2d_dynamic(nt=400, collisions=True, collect_every=10000, **CFG)
    W_fin = out["qv"].sum() + out["M"].sum() / amc + out["surf_precip"] / amc
    assert abs(W_fin - W0) / W0 < 0.005, f"water not conserved: {W_fin:.4f} vs {W0:.4f}"
