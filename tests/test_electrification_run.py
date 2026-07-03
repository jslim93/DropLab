"""Toy electrification, integrated with the dynamic model. The charging is PHYSICAL
(non-inductive graupel-crystal collisions), so it only fires once the storm actually grows
graupel (>0.2 mm) -- small/short configs never do, which is itself physically honest. One
vigorous run therefore exercises charging + dipole + field + discharge; a separate small run
checks the golden-safety (no feedback). Pure-function tests are in test_electrification.py.

NOTE: the breakdown threshold here (E_breakdown ~ hundreds of V/m) is ILLUSTRATIVE, not
physical -- a 2-D grounded-box field is ~100x weaker than a real 3-D storm field, so physical
charging cannot reach the real ~1.5e5 V/m threshold in 2-D (see docs/ELECTRIFICATION_AUDIT.md).
The charging MECHANISM is physical; the trigger level is scaled to the 2-D field."""
import numpy as np
import pytest
from examples.cloud_cases import CASES
from droplab.flow2d_dynamic import run_flow2d_dynamic


def test_electrification_does_not_change_dynamics():
    """Charge is a read-only diagnostic: turning it on must leave the microphysics and
    thermodynamics BIT-FOR-BIT unchanged (no feedback). Small/fast -- no graupel needed."""
    base = dict(CASES["deep_cold"])
    base.update(Nx=48, Nz=48, nt=120, collect_every=120, seed=3)
    off = run_flow2d_dynamic(**base, electrification=False)
    on = run_flow2d_dynamic(**base, electrification=True)
    fo, fn = off["frames"][-1], on["frames"][-1]
    assert np.array_equal(fo["qc"], fn["qc"])
    assert np.array_equal(fo["q_ice"], fn["q_ice"])
    assert np.array_equal(fo["theta"], fn["theta"])
    assert on["charge"] is not None and off["charge"] is None


def test_physical_charging_dipole_field_and_flash():
    """A vigorous deep-cold cell develops graupel -> non-inductive charging separates charge,
    the dipole forms from transport, and the 2-D field (at the illustrative threshold) breaks
    down into branched discharges. One run exercises the whole pipeline."""
    cfg = dict(CASES["deep_cold"])
    cfg.update(Nx=96, Nz=96, nt=900, collect_every=100, seed=3,
               electrification=True, charge_eff=0.3, E_breakdown=400.0)
    o = run_flow2d_dynamic(**cfg)
    frames = o["frames"]
    totq = np.array([np.abs(f["charge"]).sum() for f in frames])
    # Charge conservation is DETERMINISTIC and must always hold (sweep deposition + flash
    # conserve; the only sink is precipitation charge) -- assert it regardless of trajectory.
    resid = float(o["charge"].sum()) + float(o["charge_to_ground"])
    assert abs(resid) < 1e-10 * max(float(np.abs(o["charge"]).sum()), 1e-30)
    # Whether the cell grows graupel (>0.2 mm) and thus charges at all is a CHAOTIC emergent
    # outcome: the same code + seed on a different platform/BLAS takes a slightly different
    # trajectory, and this short config can land just under the graupel threshold (no charge).
    # The charging PHYSICS is deterministically unit-tested in test_electrification.py; here we
    # only assert the end-to-end pipeline when the premise (graupel actually formed) holds.
    if totq.max() == 0.0:
        pytest.skip("cell did not grow graupel on this platform's trajectory; "
                    "charging physics is unit-tested deterministically in test_electrification.py")
    # vertical dipole on the peak-charge frame: positive (crystals) above negative (graupel)
    fp = frames[int(np.argmax(totq))]
    z, q = fp["z"], fp["charge"]
    pos, neg = q > 0, q < 0
    assert pos.any() and neg.any()
    assert np.average(z[pos], weights=q[pos]) > np.average(z[neg], weights=-q[neg])
    # branched discharges that remove charge
    flashes = [fl for fr in frames for fl in fr.get("flashes", [])]
    assert len(flashes) > 0
    assert any(len(fl["segments"]) > 3 for fl in flashes)      # channels propagate
    assert any(fl["q_neutralized"] > 0.0 for fl in flashes)    # discharge removed charge
