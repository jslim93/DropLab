"""Gates for the student climate-intervention control panel.

Checks the testable core (`simulate`) returns sensible diagnostics for the headline
knobs, and that the interactive panel constructs without a notebook front-end.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")

from droplab.climate_widget import simulate, climate_panel, figure


_SMALL = dict(nt=400, Nx=40, Nz=28, X=2000.0, Z=1200.0, n_super=12000)


def test_simulate_returns_sensible_summary():
    res, s = simulate(background_N=200.0, **_SMALL)
    assert set(s) == {"reff_um", "albedo", "precip_kg", "droplet_number"}
    assert 1.0 < s["reff_um"] < 60.0, f"r_eff out of range ({s['reff_um']})"
    assert 0.0 <= s["albedo"] <= 1.0
    assert s["droplet_number"] > 0.0
    # the panel's figure renders from this without error
    figure(res, s, title="test")


def test_seeding_increases_droplet_number():
    """Turning on MCB seeding injects CCN, so total droplet number must rise."""
    _, base = simulate(background_N=200.0, seed_on=False, **_SMALL)
    _, seed = simulate(background_N=200.0, seed_on=True, seed_kind="MCB sea-salt",
                       seed_N=300.0, seed_r=0.1, **_SMALL)
    assert seed["droplet_number"] > base["droplet_number"], "seeding did not add droplets"


def test_panel_builds():
    """The ipywidgets panel constructs headlessly (no display needed)."""
    climate_panel()      # builds and calls display(); must not raise
