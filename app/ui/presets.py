"""Curated configuration data for the sandbox — aerosol presets, the grouped 2-D
scenario catalogue, quick-look sizing, and the five one-click demos.

This module is plain data + small pure helpers (no Streamlit, no physics). The
2-D scenarios are thin wrappers over the validated ``examples.cloud_cases.CASES``
configs: a scenario names a CASES entry plus UI metadata (which group it lives
in, which microphysics toggles make sense for it). Quick-look sizing shrinks the
heavy CASES grids so the first click returns in well under ~30 s while keeping
the qualitative phenomenon; "full resolution" uses the original CASES config.
"""
from __future__ import annotations

import copy

from examples.cloud_cases import CASES

# --- Parcel aerosol presets (per-mode N cm^-3, mean radius um, geom sigma; scalar kappa) ---
AEROSOL_PRESETS = {
    "default":     dict(N_raw=(118., 11., .72), mu_um=(.019, .056, .46),
                        sig=(3.3, 1.6, 2.2), kappa=1.6),
    "maritime":    dict(N_raw=(100., 20.), mu_um=(.08, .4),
                        sig=(1.6, 2.0), kappa=1.0),
    "continental": dict(N_raw=(3200., 2900.), mu_um=(.012, .04),
                        sig=(1.7, 2.0), kappa=0.3),
    "arctic":      dict(N_raw=(15., 5.), mu_um=(.05, .2),
                        sig=(1.6, 2.0), kappa=0.5),
}
GCCN_MODE = dict(N=0.01, r=2.0, sig=1.5, kappa=1.2)

# --- 2-D scenario catalogue ------------------------------------------------- #
# group order also drives the picker grouping.
GROUPS = ["Warm", "Cold & mixed-phase", "Deep convection", "Idealized"]

# A scenario = a CASES key + UI metadata.
#   ice_default        : sensible default state of the ice toggle for this scenario
#   ice_capable        : whether the cloud top can reach 0 °C, so ice does anything.
#                        False on shallow-warm clouds (the ice toggle is disabled
#                        there — toggling it would do nothing, which reads as
#                        broken). Congestus IS ice_capable: its tower reaches
#                        ~7 km / well below freezing, so ice glaciates it.
#   allow_electrify    : electrification only makes sense in deep/cold mixed-phase
#                        (needs riming graupel) — gate it to those
#   default_min        : default simulated duration (minutes) — long enough to
#                        actually FORM a cloud at the quick grid (physics-tuned)
#   dt_default         : default time step (s) for this scenario's stability
#   blurb              : one-line "what is this environment"
SCENARIOS = {
    # --- Warm ---
    "bomex":     dict(group="Warm", case="bomex", label="BOMEX shallow cumulus",
                      blurb="Trade-wind fair-weather cumulus field — small, non-raining puffs.",
                      ice_default=False, ice_capable=False, allow_electrify=False,
                      default_min=35, dt_default=2.0),
    "rico":      dict(group="Warm", case="rico", label="RICO warm-rain cumulus (drizzle)",
                      blurb="Trade cumulus that starts clear and warm-rains via "
                            "collision–coalescence — the clean warm-rain example.",
                      ice_default=False, ice_capable=False, allow_electrify=False,
                      default_min=35, dt_default=1.5),
    "dycoms":    dict(group="Warm", case="dycoms", label="DYCOMS marine stratocumulus",
                      blurb="A radiatively-driven reflective Sc sheet — the climate sunshade.",
                      ice_default=False, ice_capable=False, allow_electrify=False,
                      default_min=30, dt_default=1.0),
    "congestus": dict(group="Warm", case="congestus", label="Congestus deep cumulus",
                      blurb="A tall warm cumulus tower (moist sounding — starts with "
                            "scattered cloud); add ice to glaciate the upper cloud.",
                      ice_default=False, ice_capable=True, allow_electrify=False,
                      default_min=30, dt_default=2.0),
    "fog":       dict(group="Warm", case="fog", label="Radiation fog",
                      blurb="A surface cloud from nocturnal ground cooling (base near 0 m).",
                      ice_default=False, ice_capable=False, allow_electrify=False,
                      default_min=15, dt_default=1.0),
    "diurnal":   dict(group="Warm", case="diurnal", label="Diurnal cumulus cycle",
                      blurb="Continental cumulus building through a compressed afternoon.",
                      ice_default=False, ice_capable=False, allow_electrify=False,
                      default_min=60, dt_default=2.0),
    # --- Cold & mixed-phase ---
    "arctic":    dict(group="Cold & mixed-phase", case="arctic",
                      label="Arctic mixed-phase deck (MOSAiC)",
                      blurb="A supercooled-liquid deck slowly glaciating by Bergeron (WBF) — "
                            "give it time to overturn.",
                      ice_default=True, ice_capable=True, allow_electrify=False,
                      default_min=75, dt_default=1.0),
    "deep_cold": dict(group="Cold & mixed-phase", case="deep_cold",
                      label="Deep cold storm (snow)",
                      blurb="A sub-freezing convective storm that glaciates aloft and snows out.",
                      ice_default=True, ice_capable=True, allow_electrify=True,
                      default_min=45, dt_default=2.0),
    # --- Deep convection ---
    "deep_convection": dict(group="Deep convection", case="deep_convection",
                            label="Anelastic cumulonimbus",
                            blurb="A single ~10 km tower with a glaciating anvil — needs the anelastic core.",
                            ice_default=True, ice_capable=True, allow_electrify=True,
                            default_min=60, dt_default=4.0),
    # --- Idealized ---
    "idealized": dict(group="Idealized", case="shear", label="Idealized warm bubble",
                      blurb="One warm bubble in an idealized column — add wind shear (Dynamics) "
                            "to tilt it into bands.",
                      ice_default=False, ice_capable=False, allow_electrify=False,
                      default_min=20, dt_default=2.0),
}

# Derived sets (single source of truth = the per-scenario flags above).
# Scenarios physically deep/cold enough for electrification's riming path:
ELECTRIFY_SCENARIOS = {k for k, v in SCENARIOS.items() if v["allow_electrify"]}
# Scenarios whose cloud top can reach freezing, so the ice toggle does something:
ICE_CAPABLE = {k for k, v in SCENARIOS.items() if v["ice_capable"]}


# --- quick-look sizing ------------------------------------------------------ #
# Per-scenario GRID shrink for a fast first click while preserving the
# qualitative phenomenon. Duration (nt) is NOT set here — it comes from the
# "Simulated time" control (default_min / dt_default per scenario) so grid
# resolution and run length are decoupled.
QUICKLOOK = {
    "bomex":     dict(Nx=72, Nz=56, n_super=20000),
    "rico":      dict(Nx=72, Nz=64, n_super=24000),
    "dycoms":    dict(Nx=72, Nz=40, n_super=16000),
    "congestus": dict(Nx=64, Nz=64, n_super=18000),
    "fog":       dict(Nx=56, Nz=40, n_super=14000),
    "diurnal":   dict(Nx=64, Nz=48, n_super=16000),
    "arctic":    dict(Nx=64, Nz=44, n_super=18000),
    "deep_cold": dict(Nx=64, Nz=64, n_super=18000),
    "deep_convection": dict(Nx=84, Nz=80, X=12000.0, Z=12000.0,
                            n_super=20000, sponge_frac=0.28),
    "idealized": dict(Nx=80, Nz=48, n_super=16000),
}


def base_config(scenario: str) -> dict:
    """A deep copy of the validated CASES config for a scenario (full resolution)."""
    return copy.deepcopy(CASES[SCENARIOS[scenario]["case"]])


# A deliberately tiny sizing used only by the smoke tests — fast enough to run
# several toggle combinations per second while still producing every frame key.
TINY = dict(Nx=24, Nz=20, nt=40, n_super=2000)


def sized_config(scenario: str, resolution: str = "quick", nt=None,
                 dt=None) -> dict:
    """Return the run_flow2d_dynamic kwargs for a scenario at the chosen GRID
    resolution. ``quick`` applies the quick-look grid shrink; ``full`` uses CASES
    as-is; ``tiny`` is a minimal grid for tests. ``nt`` / ``dt`` (run length and
    time step) override the CASES values when given — grid and duration are
    decoupled. ``collect_every`` is set for a smooth ~30-frame animation.
    """
    cfg = base_config(scenario)
    if resolution == "quick":
        cfg.update(QUICKLOOK.get(scenario, {}))
    elif resolution == "tiny":
        cfg.update(TINY)
    if dt is not None:
        cfg["dt"] = float(dt)
    if nt is not None:
        cfg["nt"] = int(nt)
    cfg["collect_every"] = max(2, cfg["nt"] // 30)
    return cfg


def default_nt(scenario: str, resolution: str = "quick") -> int:
    """The nt implied by a scenario's default duration/dt (for sizing checks)."""
    m = SCENARIOS[scenario]
    return max(1, round(m["default_min"] * 60 / m["dt_default"]))


# --- climate-mode run-length constants (single source of truth) ------------- #
# Used by BOTH render_climate and cache.demo_climate_args so the warmed demo key
# can never drift from what the MCB demo button actually runs.
CLIMATE_RUN_STEPS = {"Quick": 1000, "Standard": 1800, "Long": 2800}
CLIMATE_RUN_DEFAULT = "Standard"
CLIMATE_INJECT_FRAC = 0.20        # default injection time as a fraction of run


# --- the five curated one-click demos --------------------------------------- #
# Each demo names a scenario, the toggles that make it sing, and a short pitch.
# All use the quick-look sizing above so they return fast. (The climate demo is
# handled by the Climate page; it is listed here for the home strip.)
DEMOS = [
    dict(key="lightning", title="⚡ Lightning cumulonimbus", page="2D",
         scenario="deep_convection",
         toggles=dict(ice=True, electrification=True),
         pitch="A 10 km anelastic tower glaciates, charges up, and throws a bolt."),
    dict(key="arctic", title="❄️ Arctic mixed-phase (WBF)", page="2D",
         scenario="arctic",
         toggles=dict(ice=True),
         pitch="A supercooled liquid deck with a few ice crystals growing by Bergeron."),
    dict(key="habit", title="✦ Ice-habit gallery", page="2D",
         scenario="deep_cold",
         toggles=dict(ice=True, habit=True),
         pitch="Crystal SHAPES predicted by temperature — plates ↔ columns."),
    dict(key="warmrain", title="🌧️ Warm-rain shower", page="2D",
         scenario="rico",
         toggles=dict(ice=False, collisions=True),
         pitch="Collision-coalescence broadens the spectrum into drizzle."),
    dict(key="mcb", title="🌊 Marine cloud brightening", page="Climate",
         scenario=None, toggles={},
         pitch="A baseline-vs-seeded twin: more, smaller drops brighten the deck."),
]
