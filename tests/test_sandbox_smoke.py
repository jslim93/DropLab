"""Smoke tests for the DropLab sandbox app (app/).

These run WITHOUT a Streamlit server: the page files guard their render call on
the Streamlit runtime context, and all real logic lives in importable functions
(app.ui.*). We exercise the cached run wrappers with tiny configs, assert the
regime-dependent frame keys appear per toggle, verify the hard physics couplings
are enforced, confirm the parked LEM/Smagorinsky knobs are surfaced nowhere, and
that the page modules import cleanly. The whole file runs in well under a minute.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

from app.ui import presets, cache, plots, controls, theme, modes  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --- ui package imports cleanly -------------------------------------------- #
def test_ui_modules_import():
    for m in (presets, cache, plots, controls, theme, modes):
        assert m is not None


# --- page files import without executing the Streamlit script -------------- #
@pytest.mark.parametrize("rel", [
    "app/Home.py", "app/pages/1_Parcel.py", "app/pages/2_TwoD.py",
    "app/pages/3_Climate.py", "app/pages/4_Lecture.py",
])
def test_page_imports(rel):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(f"page_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)        # guarded: render only runs under Streamlit
    assert mod is not None


# --- no parked research knobs anywhere in the app -------------------------- #
def test_no_lem_or_smagorinsky():
    import re
    # word-boundary match so 'problem'/'implement'/'element' don't false-positive
    pat = re.compile(r"\b(lem|lem_eps|lem_tau|lem_depletion|smag|smagorinsky|smag_cs)\b",
                     re.IGNORECASE)
    hits = [f"{p}:{i}" for p in (ROOT / "app").rglob("*.py")
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if pat.search(line)]
    assert not hits, f"parked knobs surfaced: {hits}"


def test_twod_streams_live_then_serves_from_cache():
    # a unique config so it isn't already cached by another test
    args = ("dycoms", "tiny", 37, 2.0, True, False, False, False, "abifm",
            True, True, True, (123.0,), (0.08,), (2.0,), (0.6,), False,
            "MCB sea-salt", 200.0, 0.1, None, 0.0, None, None, None, 400.0, 0.3)
    n = {"c": 0}
    assert not cache.twod_is_cached(*args)
    r1 = cache.run_twod(*args, on_frame=lambda *a: n.__setitem__("c", n["c"] + 1))
    assert n["c"] > 0                      # frames streamed live on the first run
    assert cache.twod_is_cached(*args)
    n["c"] = 0
    r2 = cache.run_twod(*args, on_frame=lambda *a: n.__setitem__("c", n["c"] + 1))
    assert n["c"] == 0 and r2 is r1        # second run served from cache, no recompute


# --- 2-D regime-aware frame keys per toggle -------------------------------- #
def _twod(scenario, **t):
    # keyword args so adding/reordering run_twod params can't silently misalign
    return cache.run_twod(
        scenario=scenario, resolution="tiny", nt=t.get("nt", 40),
        dt=t.get("dt", 2.0),
        collisions=t.get("collisions", True), ice=t.get("ice", False),
        habit=t.get("habit", False),
        electrification=t.get("electrification", False), freezing_mode="abifm",
        homogeneous=True, melt=True, hallett_mossop=True,
        N_modes=(150.0,), mu_um=(0.08,), sig=(2.0,), kappa=(0.6,),
        seed_on=t.get("seed_on", False), seed_kind="MCB sea-salt",
        seed_N=200.0, seed_r=0.1, inject_min=t.get("inject_min", 0.0),
        wind_shear=0.0, dtheta_bubble=None, inp_n_cm3=None, inp_r_um=None,
        E_breakdown=400.0, charge_eff=0.3)


def test_twod_warm_has_base_keys_only():
    r = _twod("bomex", ice=False)
    assert not r["unstable"]
    f = r["frames"][0]
    for k in ("step", "x", "z", "r_um", "A", "qc", "supersat", "theta", "qv",
              "u", "w", "surf_precip"):
        assert k in f, k
    assert "q_ice" not in f and "phase" not in f
    assert "phi" not in f
    assert "charge" not in f


def test_twod_ice_adds_phase_keys():
    r = _twod("deep_cold", ice=True)
    f = r["frames"][0]
    assert "q_ice" in f and "phase" in f and "q_liquid" in f
    assert "phi" not in f          # habit off
    views = [v[0] for v in plots.regime_views(r)]
    assert "phase" in views and "bergeron" in views


def test_twod_habit_adds_phi():
    r = _twod("deep_cold", ice=True, habit=True)
    f = r["frames"][0]
    assert "phi" in f and "a_axis" in f and "c_axis" in f
    assert "habit" in [v[0] for v in plots.regime_views(r)]


def test_twod_electrification_adds_charge_and_flashes():
    r = _twod("deep_convection", ice=True, electrification=True)
    f = r["frames"][0]
    assert "charge" in f and "charge_density" in f and "flashes" in f
    assert "electric" in [v[0] for v in plots.regime_views(r)]


# --- the hard physics couplings (enforced in build_twod_config) ------------ #
def _cfg(scenario, **t):
    return cache.build_twod_config(
        scenario, "tiny",
        collisions=t.get("collisions", True), ice=t.get("ice", False),
        habit=t.get("habit", False),
        electrification=t.get("electrification", False), freezing_mode="abifm",
        homogeneous=True, melt=True, hallett_mossop=True,
        N_modes=(150.0,), mu_um=(0.08,), sig=(2.0,), kappa=(0.6,),
        seed_on=t.get("seed_on", False), seed_kind=t.get("seed_kind", "MCB sea-salt"),
        seed_N=t.get("seed_N", 200.0), seed_r=t.get("seed_r", 0.1),
        inject_min=t.get("inject_min", None),
        wind_shear=t.get("wind_shear", 0.0), dtheta_bubble=None,
        inp_n_cm3=None, inp_r_um=None, E_breakdown=400.0, charge_eff=0.3,
        nt=t.get("nt"), dt=t.get("dt"))


def test_coupling_habit_requires_ice():
    cfg = _cfg("deep_cold", ice=False, habit=True)
    assert cfg["ice"] is True and cfg["habit"] is True


def test_coupling_electrification_requires_ice():
    cfg = _cfg("deep_convection", ice=False, electrification=True)
    assert cfg["ice"] is True and cfg["electrification"] is True


def test_coupling_electrification_gated_to_scenario():
    # a warm scenario can never turn on electrification
    cfg = _cfg("bomex", ice=True, electrification=True)
    assert cfg["electrification"] is False


def test_coupling_deep_convection_forces_anelastic():
    cfg = _cfg("deep_convection", ice=True)
    assert cfg["dynamics"] == "anelastic"


def test_coupling_wind_shear_forces_periodic_x():
    cfg = _cfg("idealized", wind_shear=2.5e-3)
    assert cfg["wind_shear"] > 0 and cfg["periodic_x"] is True


def test_warm_scenario_cannot_run_ice():
    # Item 6: a stale disabled ice/habit True must NOT leak onto a warm scenario.
    cfg = _cfg("bomex", ice=True, habit=True)
    assert cfg["ice"] is False and cfg["habit"] is False


def test_seeding_inject_minute_override():
    # Item 4: inject_min sets the spec's t_inject (seconds).
    cfg = _cfg("deep_cold", ice=True, seed_on=True, inject_min=3.0, nt=100, dt=2.0)
    assert cfg["seeding"]["t_inject"] == pytest.approx(3.0 * 60.0)


def test_seed_defaults_differ_by_strategy():
    # Item 3: MCB (many tiny) vs GCCN (few giant) must have distinct defaults.
    mcb = controls.SEED_DEFAULTS["MCB sea-salt"]
    gccn = controls.SEED_DEFAULTS["GCCN (precip)"]
    assert mcb["N"] > gccn["N"]      # many vs few
    assert mcb["r"] < gccn["r"]      # tiny vs giant


def test_ice_capability_rule():
    # every scenario declares ice_capable, and the rule is correct:
    # shallow-warm scenarios cannot enable ice; congestus + cold/deep can
    for v in presets.SCENARIOS.values():
        assert isinstance(v.get("ice_capable"), bool)
    for warm in ("bomex", "dycoms", "fog", "diurnal", "idealized"):
        assert presets.SCENARIOS[warm]["ice_capable"] is False
        assert warm not in presets.ICE_CAPABLE
    for cold in ("arctic", "deep_cold", "deep_convection"):
        assert presets.SCENARIOS[cold]["ice_capable"] is True
        assert cold in presets.ICE_CAPABLE


# --- parcel + climate wrappers --------------------------------------------- #
def test_parcel_run_and_figures():
    out, M, A = cache.run_parcel(
        0, 600, 120, 1.0, 293.2, 1013e2, 0.92, 1.0, "linear",
        (100.0, 20.0), (0.08, 0.4), (1.6, 2.0), 1.0, True, False, 0.0, 0.0, 0.0)
    assert len(out) > 0 and M.size == A.size
    last = out[sorted(out)[-1]]
    assert {"NC", "NR", "rv", "qc", "qr", "RH"} <= set(last)
    fig = plots.parcel_timeseries([("m", out, M, A)], 1.0)
    assert fig is not None


def test_climate_twin_metrics_and_synced_animation():
    # climate collects >1 frame; ts carries the MCB metrics (N_d, albedo, CRE);
    # the synced scene+graphs animation (with dotted control) builds.
    seeded = cache.run_climate(200.0, 0.0, True, "MCB sea-salt", 200.0, 0.1,
                               0.5, 120, 32, 24, 6000)
    base = cache.run_climate(200.0, 0.0, False, "MCB sea-salt", 200.0, 0.1,
                             0.5, 120, 32, 24, 6000)
    for k in ("png", "frames", "meta", "ts", "n_frames", "albedo", "albedo_mean"):
        assert k in seeded and k in base
    assert seeded["n_frames"] > 1
    for k in ("t", "nc", "albedo", "cre"):       # MCB metrics, NOT q_c
        assert k in seeded["ts"] and len(seeded["ts"][k]) == seeded["n_frames"]
    g = plots.climate_scene_series_gif(seeded, ctrl_ts=base["ts"])
    assert isinstance(g, (bytes, bytearray)) and len(g) > 1000
    assert isinstance(cache.climate_forcing(
        seeded["albedo_mean"] - base["albedo_mean"]), float)


# --- a render produces image bytes / figures ------------------------------- #
def test_twod_render_outputs():
    r = _twod("deep_cold", ice=True, habit=True)
    assert len(plots.scene_image(r)) > 1000
    assert len(plots.phase_image(r)) > 1000
    assert len(plots.habit_image(r)) > 1000
    assert plots.bergeron_figure(r) is not None


def test_twod_timeseries_regime_aware():
    # Item 7: warm run → 3 base panels; ice run adds the liquid/ice panel.
    warm = _twod("bomex")
    assert plots.twod_timeseries(warm) is not None
    iced = _twod("deep_cold", ice=True)
    fig = plots.twod_timeseries(iced)
    titles = {a.text for a in fig.layout.annotations}
    assert any("Liquid vs ice" in s for s in titles)
    assert "qc_max" in iced and iced["qc_max"] > 0


def test_scene_and_series_gif_synced_and_memoized():
    # combined scene + growing time-series animation (shown after caching)
    r = _twod("deep_cold", ice=True)
    g = plots.scene_and_series_gif(r, show_field=True, wind="off")
    assert isinstance(g, (bytes, bytearray)) and len(g) > 0
    assert plots.scene_and_series_gif(r, show_field=True, wind="off") is g  # memoized


def test_parcel_aerosol_presets_within_widget_bounds():
    # the Parcel "Edit modes" number_inputs must ACCEPT every preset's defaults
    # (a preset value above a widget max crashes the page — e.g. default σ=3.3, κ=1.6).
    for _name, p in presets.AEROSOL_PRESETS.items():
        for N in p["N_raw"]:
            assert 0.0 <= float(N) <= 1e5
        for mu in p["mu_um"]:
            assert 0.001 <= float(mu) <= 5.0
        for s in p["sig"]:
            assert 1.05 <= float(s) <= 4.0
        ks = p["kappa"] if isinstance(p["kappa"], (list, tuple)) else [p["kappa"]]
        for k in ks:
            assert 0.01 <= float(k) <= 2.0


def test_climate_is_cached_matches_run_args():
    # the render path calls climate_is_cached(*clim_args) with NO dt — its key must
    # match what run_climate(*clim_args) stores (both default dt=1.0).
    args = (170.0, 0.0, False, "MCB sea-salt", 200.0, 0.1, 0.5, 120, 32, 24, 6000)
    assert not cache.climate_is_cached(*args)
    cache.run_climate(*args)
    assert cache.climate_is_cached(*args)


def test_demo_climate_args_match_render_defaults():
    # the warmed MCB-demo key must equal what the demo button actually runs —
    # this drifted once (warm nt=1000/inject 25% vs app Standard=1800/20%).
    args = cache.demo_climate_args(seed_on=True)
    nt = presets.CLIMATE_RUN_STEPS[presets.CLIMATE_RUN_DEFAULT]
    assert args[7] == nt
    assert args[6] == round(presets.CLIMATE_INJECT_FRAC * (nt / 60.0), 1)


def test_showcase_lessons_registry():
    # Lecture lessons (mixed-phase I1-I4 + showcase H1/S1/D1/L1): registry
    # integrity. The lesson runs themselves are quick-grid (~10-60 s each) and
    # physics-validated separately — keep the default smoke suite fast.
    from app.ui import lessons
    assert len(lessons.MIXED_LESSONS) == 4
    assert len(lessons.SHOWCASE_LESSONS) == 4
    assert len(lessons.ALL_LESSONS) == 8
    for name, spec in lessons.ALL_LESSONS.items():
        assert callable(spec["render"]) and spec["misconception"]
    # curriculum spine order: mixed-phase I1→I4, then habit→snow→deep→lightning
    keys = [k.split(" ")[0] for k in lessons.ALL_LESSONS]
    assert keys == ["I1", "I2", "I3", "I4", "H1", "S1", "D1", "L1"]


def test_climate_timeseries_figure():
    out = cache.run_climate(200.0, 0.0, False, "MCB sea-salt", 200.0, 0.1,
                            0.5, 120, 32, 24, 6000)
    assert plots.climate_timeseries(out["ts"]) is not None
    # dotted control overlay variant
    assert plots.climate_timeseries(out["ts"], ctrl=out["ts"]) is not None
