"""Entrainment-mixing closure (entrain_mode const/auto): the subgrid stage-2 that
homogenizes engulfed environment filaments. See flow2d_dynamic step 3c."""
import numpy as np

from droplab.flow2d_dynamic import run_flow2d_dynamic

# small, fast bubble config: cloud forms within ~150 steps; closed water budget
# (no forcing, no sedimentation, no collisions) so conservation is testable exactly.
FAST = dict(nt=250, dt=1.5, Nx=32, Nz=24, X=1600.0, Z=1200.0, n_super=6000,
            collisions=False, sediment=False, seed=1, collect_every=250,
            smagorinsky=False)


def _total_water(out):
    """grid vapor + all super-droplet condensate, one number per frame [kg]"""
    amc = out["air_mass_cell"]
    f = out["frames"][-1]
    qv_mass = float((f["qv"].ravel() * (amc if np.ndim(amc) else amc)).sum())
    # frames carry qc in g/kg on the same cells -> condensate mass via mixing ratio
    qc_mass = float((f["qc"].ravel() * 1e-3 * (amc if np.ndim(amc) else amc)).sum())
    return qv_mass + qc_mass


def test_entrainment_reduces_cloud_water():
    off = run_flow2d_dynamic(**FAST, entrain_mode="off")
    on = run_flow2d_dynamic(**FAST, entrain_mode="const", entrain_eps=2.0e-3)
    qc_off = off["frames"][-1]["qc"].sum()
    qc_on = on["frames"][-1]["qc"].sum()
    assert qc_off > 0.0, "baseline bubble made no cloud -- test config broken"
    assert qc_on < 0.85 * qc_off, f"entrainment did not thin the cloud ({qc_on:.3f} vs {qc_off:.3f})"


def test_entrainment_preserves_droplet_number():
    """dilution shrinks droplet mass but must not remove super-droplets."""
    off = run_flow2d_dynamic(**FAST, entrain_mode="off")
    on = run_flow2d_dynamic(**FAST, entrain_mode="const", entrain_eps=2.0e-3)
    assert on["frames"][-1]["A"].shape == off["frames"][-1]["A"].shape


def test_entrainment_conserves_total_water():
    """closed box (no forcing/sediment): the conservative detrainment must not leak.
    Compare the on-run's water drift against the off-run's numerical drift."""
    off = run_flow2d_dynamic(**FAST, entrain_mode="off")
    on = run_flow2d_dynamic(**FAST, entrain_mode="auto", entrain_ce=2.0e-3)
    w_off, w_on = _total_water(off), _total_water(on)
    assert abs(w_on - w_off) / w_off < 5.0e-3, \
        f"entrainment changed the total-water budget: off={w_off:.6e} on={w_on:.6e}"


def test_auto_mode_runs_all_regimes_smoke():
    for kw in (dict(entrain_mode="auto"),
               dict(entrain_mode="auto", ice=True, freezing_mode="bigg")):
        out = run_flow2d_dynamic(**FAST, T0=270.0 if kw.get("ice") else 289.0, **kw)
        f = out["frames"][-1]
        assert np.isfinite(f["qc"]).all() and np.isfinite(f["theta"]).all()
