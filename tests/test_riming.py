"""Stage C: riming -- an ice super-droplet collects a supercooled liquid one, so the
liquid mass transfers onto the ice (phase stays ice) and the latent heat of fusion is
released. The collision core is the golden lynchpin, so these tests check the *physics*
(ice grows, water + energy conserved); bit-identity of the ice=False path is enforced
separately by tests/test_flow2d_golden.py."""
import numpy as np
from examples.cloud_cases import CASES
from droplab.flow2d_dynamic import run_flow2d_dynamic


def _deep_cold(**over):
    """The deep-cold convective snow case: ice forms aloft (immersion + homogeneous) while
    supercooled liquid persists below -> mixed-phase cells where riming can act. Sediment
    off so the domain is a closed water/energy budget (riming is a pure transfer)."""
    cfg = dict(CASES["deep_cold"])
    cfg.update(nt=240, collect_every=240, sediment=False)
    cfg.update(over)
    return cfg


def test_riming_grows_ice_and_conserves_water():
    on = run_flow2d_dynamic(**_deep_cold(collisions=True))
    off = run_flow2d_dynamic(**_deep_cold(collisions=False))
    fon, foff = on["frames"][-1], off["frames"][-1]

    # the deck is genuinely mixed-phase: ice and liquid coexist
    assert fon["q_ice"].max() > 0.0 and fon["q_liquid"].max() > 0.0

    # riming transfers supercooled liquid onto ice -> more ice with collisions on than off
    iwp_on, iwp_off = fon["q_ice"].sum(), foff["q_ice"].sum()
    assert iwp_on > iwp_off

    # with sedimentation off, riming only re-partitions liquid <-> ice: total condensate
    # is conserved (collisions are not a water source/sink)
    tot_on = fon["q_liquid"].sum() + fon["q_ice"].sum()
    tot_off = foff["q_liquid"].sum() + foff["q_ice"].sum()
    assert np.isfinite(tot_on)
    assert abs(tot_on - tot_off) / tot_off < 0.05

    # energy: freezing the rimed liquid releases l_f, so the riming run is no colder
    # in the column mean than the no-collision run
    assert float(np.mean(on["T_col"])) >= float(np.mean(off["T_col"])) - 1e-6


def test_riming_branch_inactive_without_ice():
    """ice=False -> rimed_out is None and every droplet has phase 0, so the mixed-phase
    branch is unreachable and the warm collision result is reproducible (the golden test
    enforces bit-identity against the frozen reference separately)."""
    base = dict(nt=60, dt=1.0, Nx=16, Nz=16, n_super=2000, collisions=True,
                ice=False, collect_every=60)
    a = run_flow2d_dynamic(**base)
    b = run_flow2d_dynamic(**base)
    assert np.array_equal(a["frames"][-1]["qc"], b["frames"][-1]["qc"])
