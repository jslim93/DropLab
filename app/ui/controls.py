"""Reusable Streamlit control clusters.

These build sidebar widgets and return plain Python values. The 2-D microphysics
panel is where the hard physics couplings surface in the UI: invalid toggle
combinations are made impossible (a dependent toggle is disabled with a visible
reason) — the cache wrapper then re-enforces the same rules as the authoritative
backstop.
"""
from __future__ import annotations

import streamlit as st

from app.ui import presets


def aerosol_two_mode(key: str, default_N=200.0, default_mu=0.08,
                     default_kappa=0.6):
    """A 1–2 mode lognormal aerosol editor. Returns four tuples
    (N_modes, mu_um, sig, kappa). Mode 2 is an opt-in coarse/giant mode."""
    with st.expander("🌫️ Aerosol", expanded=False):
        N1 = st.slider("Mode 1 — number N (cm⁻³)", 10.0, 1000.0, default_N, 10.0,
                       key=f"{key}_N1")
        mu1 = st.slider("Mode 1 — dry radius (µm)", 0.02, 1.0, default_mu, 0.01,
                        key=f"{key}_mu1")
        sig1 = st.slider("Mode 1 — geom. σ", 1.2, 2.8, 2.0, 0.1, key=f"{key}_sig1")
        kappa1 = st.slider("Mode 1 — hygroscopicity κ", 0.05, 1.2, default_kappa,
                           0.05, key=f"{key}_k1",
                           help="κ≈1.2 sea salt, ≈0.6 sulfate, ≈0.1 organics/dust.")
        mode2 = st.checkbox("Add a 2nd (coarse / giant) mode", key=f"{key}_m2")
        N2 = st.slider("Mode 2 — N (cm⁻³)", 0.01, 200.0, 1.0, 0.01,
                       disabled=not mode2, key=f"{key}_N2")
        mu2 = st.slider("Mode 2 — dry radius (µm)", 0.05, 3.0, 1.5, 0.05,
                        disabled=not mode2, key=f"{key}_mu2")
        sig2 = st.slider("Mode 2 — geom. σ", 1.2, 2.8, 1.6, 0.1,
                         disabled=not mode2, key=f"{key}_sig2")
        kappa2 = st.slider("Mode 2 — κ", 0.05, 1.2, 1.2, 0.05,
                           disabled=not mode2, key=f"{key}_k2")
    if mode2:
        return (N1, N2), (mu1, mu2), (sig1, sig2), (kappa1, kappa2)
    return (N1,), (mu1,), (sig1,), (kappa1,)


# Strategy-specific seed defaults — MCB = many tiny; GCCN = a few giant.
SEED_KINDS = ["MCB sea-salt", "GCCN (precip)"]
_GCCN_N_OPTS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0]
SEED_DEFAULTS = {
    "MCB sea-salt":  dict(N=500.0, r=0.05),   # many tiny → brighten
    "GCCN (precip)": dict(N=0.01, r=2.0),     # a few giant → trigger rain
}


def seed_amount_size(key: str, kind: str, disabled: bool = False):
    """Strategy-aware seed amount/size widgets. Keyed by ``kind`` so flipping the
    strategy resets N/r to that strategy's defaults and ranges. Returns (N, r).

      * MCB sea-salt   — MANY TINY: N 500 cm⁻³ [50, 2000]; r 0.05 µm [0.02, 0.2]
      * GCCN (precip)  — FEW GIANT: N 0.01 cm⁻³ [1e-3, 1] (log); r 2.0 µm [0.5, 3]
    """
    d = SEED_DEFAULTS[kind]
    if kind == "MCB sea-salt":
        N = st.slider("Seed amount N (cm⁻³)", 50.0, 2000.0, d["N"], 10.0,
                      disabled=disabled, key=f"{key}_seedN_mcb",
                      help="MCB = many tiny sea-salt particles → brighten.")
        r = st.slider("Seed dry radius (µm)", 0.02, 0.2, d["r"], 0.01,
                      disabled=disabled, key=f"{key}_seedr_mcb")
    else:  # GCCN — N spans decades → a log select-slider
        N = st.select_slider("Seed amount N (cm⁻³)", options=_GCCN_N_OPTS,
                             value=d["N"], format_func=lambda v: f"{v:g}",
                             disabled=disabled, key=f"{key}_seedN_gccn",
                             help="GCCN = a few giant particles → trigger rain.")
        r = st.slider("Seed dry radius (µm)", 0.5, 3.0, d["r"], 0.1,
                      disabled=disabled, key=f"{key}_seedr_gccn")
    return float(N), float(r)


def seeding_panel(key: str, run_min: float):
    """Mid-run aerosol seeding controls (strategy-driven amount/size + timing).
    Returns (on, kind, N, r, inject_min)."""
    with st.expander("💉 Mid-run seeding"):
        on = st.checkbox("Inject aerosol part-way through", key=f"{key}_seed")
        kind = st.selectbox("Strategy", SEED_KINDS, disabled=not on,
                            key=f"{key}_seedkind",
                            help="MCB = many tiny particles (brighten). "
                                 "GCCN = a few giant particles (trigger rain).")
        N, r = seed_amount_size(key, kind, disabled=not on)
        inject_min = st.slider("Inject at (simulated min)", 0.0, float(run_min),
                               round(0.25 * run_min, 1),
                               step=max(0.5, round(run_min / 40, 1)),
                               disabled=not on, key=f"{key}_inject_{run_min:g}",
                               help="When the injection fires, in simulated time.")
    return on, kind, N, r, inject_min


def microphysics_panel(scenario: str, key: str, *, ice0=False, habit0=False,
                       electrify0=False):
    """The microphysics toggle cluster with the hard couplings enforced in the UI.

    Returns a dict of toggle values. Couplings:
      * ice needs a cold/deep-enough scenario → disabled on shallow-warm clouds
      * habit needs ice            → habit disabled unless ice is on
      * electrification needs ice  → and only on deep/cold mixed-phase scenarios
    """
    can_electrify = scenario in presets.ELECTRIFY_SCENARIOS
    ice_capable = presets.SCENARIOS[scenario].get("ice_capable", True)
    with st.expander("🧪 Microphysics", expanded=True):
        collisions = st.checkbox("Collision–coalescence (warm rain)", True,
                                 key=f"{key}_{scenario}_coll",
                                 help="Larger drops collect smaller ones → drizzle/rain.")
        ice = st.checkbox("Ice / mixed-phase", value=(ice0 and ice_capable),
                          disabled=not ice_capable, key=f"{key}_{scenario}_ice",
                          help="Freezing, depositional growth, melting and snow.")
        # a disabled checkbox retains its stored session value → force off on warm
        ice = bool(ice and ice_capable)
        if not ice_capable:
            st.caption("· Warm cloud — temperatures stay above 0 °C, no ice phase.")

        # coupling 1: habit ⇒ ice
        habit = st.checkbox("Ice habit (crystal shapes)",
                            value=(habit0 and ice), disabled=not ice,
                            key=f"{key}_{scenario}_habit",
                            help="Predicts plate↔column crystal shapes (needs ice on).")
        habit = bool(habit and ice)   # don't let a stale disabled value leak through
        if not ice:
            st.caption("· Ice habit needs **Ice / mixed-phase** on.")

        # coupling 2: electrification ⇒ ice AND a deep/cold scenario
        elec_disabled = (not ice) or (not can_electrify)
        electrification = st.checkbox(
            "Electrification + lightning",
            value=(electrify0 and ice and can_electrify),
            disabled=elec_disabled, key=f"{key}_{scenario}_elec",
            help="Charge separation by riming → a dielectric-breakdown bolt.")
        electrification = bool(electrification and ice and can_electrify)
        if not can_electrify:
            st.caption("· Lightning needs a **deep / cold mixed-phase** scenario "
                       "(deep cold storm or cumulonimbus).")
        elif not ice:
            st.caption("· Lightning needs **Ice / mixed-phase** on.")

        freezing_mode, homogeneous, melt, hm = "abifm", True, True, True
        if ice:
            with st.popover("Ice details"):
                freezing_mode = st.selectbox(
                    "Heterogeneous freezing mode", ["abifm", "bigg"], index=0,
                    key=f"{key}_fmode",
                    help="ABIFM (Knopf–Alpert) is the default; Bigg is the classic "
                         "volume/temperature parameterization.")
                homogeneous = st.checkbox("Homogeneous freezing (cold/cirrus)",
                                          value=True, key=f"{key}_homog")
                melt = st.checkbox("Melting below 0 °C", value=True, key=f"{key}_melt")
                hm = st.checkbox("Hallett–Mossop splintering", value=True,
                                 key=f"{key}_hm")
    return dict(collisions=collisions, ice=ice, habit=habit,
                electrification=electrification, freezing_mode=freezing_mode,
                homogeneous=homogeneous, melt=melt, hallett_mossop=hm)
