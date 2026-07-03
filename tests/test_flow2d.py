"""Flow field gates for the 2D kinematic framework.

The prescribed cumulus flow must be DISCRETELY non-divergent (face velocities
derived from finite differences of the stream function), have no flux through
the rigid top/bottom lids (w=0 there), and show a central updraft with flanking
downdrafts. Droplet-advection velocity interpolation must be sane.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")

from droplab.flow2d import Flow2D


def test_discrete_divergence_is_zero():
    """Face velocities from psi differences => per-cell divergence ~ machine 0.
    This is what lets MPDATA conserve scalar mass exactly."""
    f = Flow2D(X=2000.0, Z=2000.0, Nx=64, Nz=64, W0=2.0)
    div = f.divergence()                     # shape (Nx, Nz)
    assert div.shape == (64, 64)
    assert np.max(np.abs(div)) < 1e-10, f"max |div| = {np.max(np.abs(div))}"


def test_no_flux_through_lids():
    """w = 0 at z=0 and z=Z (rigid lids): psi ~ sin(pi z/Z) vanishes there."""
    f = Flow2D(Nx=32, Nz=32)
    assert np.allclose(f.w[:, 0], 0.0, atol=1e-12), "w != 0 at bottom lid"
    assert np.allclose(f.w[:, -1], 0.0, atol=1e-12), "w != 0 at top lid"


def test_central_updraft_flanking_downdraft():
    """Cumulus geometry: rising air in the middle, sinking on the flanks."""
    f = Flow2D(X=2000.0, Z=2000.0, Nx=64, Nz=64, W0=2.0)
    uc, wc = f.cell_velocities()             # (Nx, Nz) cell-centered
    jmid = f.Nz // 2                          # mid-height row
    w_center = wc[f.Nx // 2, jmid]
    w_edge = wc[0, jmid]
    assert w_center > 0.0, "no updraft at the core"
    assert w_edge < 0.0, "no downdraft on the flank"


def test_interpolation_matches_core_updraft():
    """Bilinear velocity sampling at the domain core returns a strong updraft
    and near-zero horizontal velocity (by symmetry)."""
    f = Flow2D(X=2000.0, Z=2000.0, Nx=64, Nz=64, W0=2.0)
    u, w = f.interpolate(np.array([1000.0]), np.array([1000.0]))
    assert w[0] > 1.0, f"core updraft too weak: {w[0]}"
    assert abs(u[0]) < 0.2, f"core horizontal velocity not ~0: {u[0]}"


def test_single_eddy_is_one_rotating_cell():
    """The circulation pattern: updraft over the left half, downdraft over the
    right half (one rotating cell), still divergence-free with no-flux lids."""
    f = Flow2D(X=2000.0, Z=2000.0, Nx=64, Nz=64, W0=2.0, pattern="single_eddy")
    assert np.max(np.abs(f.divergence())) < 1e-10
    assert np.allclose(f.w[:, 0], 0.0, atol=1e-12)
    uc, wc = f.cell_velocities()
    jmid = f.Nz // 2
    assert wc[f.Nx // 4, jmid] > 0.0, "no updraft on the left half"
    assert wc[3 * f.Nx // 4, jmid] < 0.0, "no downdraft on the right half"


def test_droplets_stay_in_box():
    f = Flow2D(X=2000.0, Z=2000.0, Nx=64, Nz=64, W0=2.0)
    rng = np.random.default_rng(0)
    x = rng.uniform(100, 1900, 500)
    z = rng.uniform(100, 1900, 500)
    for _ in range(400):
        x, z = f.advect(x, z, dt=5.0)
    assert x.min() >= 0 and x.max() <= f.X
    assert z.min() >= 0 and z.max() <= f.Z


def test_streamfunction_conserved_along_trajectory():
    """Droplets follow streamlines: psi is a material invariant, so it stays
    ~constant along each trajectory (bounded only by RK2 integration error)."""
    f = Flow2D(X=2000.0, Z=2000.0, Nx=64, Nz=64, W0=2.0)
    rng = np.random.default_rng(1)
    x = rng.uniform(400, 1600, 200)
    z = rng.uniform(400, 1600, 200)
    psi0 = f.psi_at(x, z)
    for _ in range(300):
        x, z = f.advect(x, z, dt=5.0)
    psi1 = f.psi_at(x, z)
    psi_scale = f.psi.max() - f.psi.min()
    drift = np.max(np.abs(psi1 - psi0)) / psi_scale
    assert drift < 0.08, f"streamline drift too large: {drift:.3f}"
