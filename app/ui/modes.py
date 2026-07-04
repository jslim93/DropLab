"""The four sandbox modes + the home console, as importable ``render_*``
functions. Page files are thin glue that call these; the smoke tests import this
module and exercise the underlying cached run wrappers directly.

No physics here — every number comes from the cached wrappers in ``app.ui.cache``.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from app.ui import theme, controls, plots, cache, presets


# ========================================================================= #
# HOME — the cloud-laboratory console
# ========================================================================= #
def render_home():
    theme.apply("Home", "⛅")
    theme.header(
        "Droplet-resolving open laboratory",
        "DropLab sandbox",
        "Build clouds from individual super-droplets. Four instruments share one "
        "lab: a microphysics microscope, a 2-D cloud you can watch, a planetary "
        "thermostat, and guided lessons.",
    )

    cards = [
        ("parcel", "01 · Microscope", "Parcel", "pages/1_Parcel.py",
         "Rise one air parcel and watch aerosol activate, grow, and rain — at "
         "the droplet level."),
        ("twod", "02 · Showcase", "2-D cloud", "pages/2_TwoD.py",
         "A 2-D dynamic cloud with ice, crystal habit, electrification and deep "
         "convection. The visuals live here."),
        ("climate", "03 · Thermostat", "Climate", "pages/3_Climate.py",
         "Marine cloud brightening as a baseline-vs-intervention twin — albedo "
         "and top-of-atmosphere forcing."),
        ("lecture", "04 · Guided", "Lecture", "pages/4_Lecture.py",
         "Linear, narrated lessons that step through fixed scenarios — "
         "instructor-friendly."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (mkey, eye, name, page, job) in zip(cols, cards):
        accent = theme.MODE_ACCENT[mkey]
        with col:
            st.markdown(
                f"<div class='dl-card'><div class='top' style='background:{accent}'></div>"
                f"<div class='eye'>{eye}</div><div class='name'>{name}</div>"
                f"<div class='job'>{job}</div></div>",
                unsafe_allow_html=True)
            if st.button(f"Open {name}", key=f"go_{mkey}", use_container_width=True):
                st.switch_page(page)

    st.markdown("###")
    theme.header("One click", "Curated demos",
                 "Each runs a fast quick-look that still shows the real "
                 "phenomenon. Open it pre-configured.", accent=theme.EMBER)
    dcols = st.columns(len(presets.DEMOS), gap="small")
    for col, demo in zip(dcols, presets.DEMOS):
        with col:
            st.markdown(f"**{demo['title']}**")
            st.caption(demo["pitch"])
            if st.button("Run demo", key=f"demo_{demo['key']}",
                         use_container_width=True):
                st.session_state["pending_demo"] = demo
                st.switch_page("pages/2_TwoD.py" if demo["page"] == "2D"
                               else "pages/3_Climate.py")

    st.divider()
    st.caption("DropLab · a pure consumer of the validated droplab physics engine "
               "— no new physics in the interface. Launch: "
               "`streamlit run app/Home.py`")


# ========================================================================= #
# PARCEL — the microphysics microscope (warm only)
# ========================================================================= #
def render_parcel():
    theme.apply("Parcel", "🔬")
    theme.header("01 · Microscope", "Parcel mode",
                 "A single rising air parcel: activation → condensation → "
                 "collision-coalescence → rain, at the droplet level. Warm-phase.",
                 accent=theme.MODE_ACCENT["parcel"])

    with st.sidebar:
        # Numerics FIRST and visible — resolution/duration are among the most important
        # choices (accuracy vs speed), not something to bury in an expander.
        st.subheader("Numerics")
        n_ptcl = int(st.number_input(
            "Super-droplets", 500, 20000, 2000, 500,
            help="How many simulated particles represent the population. More = "
                 "smoother spectra and less Monte-Carlo noise, but slower."))
        sim_min = st.slider("Simulated time (min)", 5, 60, 25, 5,
                            help="How long the parcel ascends.")
        dt = float(st.number_input("Time step dt (s)", 0.25, 5.0, 1.0, 0.25,
                                   help="Physics step. Smaller = more accurate "
                                        "condensation, slower run."))
        nt = max(1, int(round(sim_min * 60.0 / dt)))

        st.subheader("Parcel")
        T0 = st.slider("Initial T₀ (K)", 273.0, 303.0, 293.2, 0.1)
        P0 = st.slider("Initial P₀ (hPa)", 700.0, 1030.0, 1013.0, 1.0) * 100.0
        RH = st.slider("Initial RH (fraction)", 0.80, 1.0, 0.92, 0.01)
        w = st.slider("Updraft w (m/s)", 0.1, 5.0, 1.0, 0.1)
        ascending_mode = st.selectbox(
            "Ascending mode", ["linear", "sine", "in_cloud_oscillation"],
            help="linear: steady updraft. sine: the whole ascent oscillates (parcel "
                 "rises and sinks). in_cloud_oscillation: rises, then bobs up and "
                 "down inside the cloud — repeated activation/evaporation cycles.")

        st.subheader("Aerosol")
        preset = st.selectbox("Preset (quick-fill)", list(presets.AEROSOL_PRESETS),
                              help="maritime: few large CCN → fast rain. "
                                   "continental: many small CCN → suppressed rain. "
                                   "arctic: ultra-clean. Edit the modes below to "
                                   "tweak from the preset.")
        compare_preset = st.selectbox(
            "Compare with (overlay)", ["(none)"] + list(presets.AEROSOL_PRESETS),
            help="Overlay a second preset on the time series and profiles — e.g. "
                 "maritime vs continental shows aerosol suppressing rain directly.")
        p = presets.AEROSOL_PRESETS[preset]
        kappa_pre = float(p["kappa"]) if np.isscalar(p["kappa"]) else float(p["kappa"][0])
        use_edit = st.checkbox("✎ Edit aerosol modes", value=False,
                               key="parcel_use_edit",
                               help="Set N / radius / σ per mode directly, "
                                    "pre-filled from the preset.")
        if use_edit:
            with st.expander("Edit modes", expanded=True):
                nmodes = int(st.number_input(
                    "Number of modes", 1, 3, min(3, len(p["N_raw"])), 1,
                    key="parcel_nmodes"))
                N_list, mu_list, sig_list = [], [], []
                for i in range(nmodes):
                    dN = float(p["N_raw"][i]) if i < len(p["N_raw"]) else 50.0
                    dmu = float(p["mu_um"][i]) if i < len(p["mu_um"]) else 0.1
                    dsig = float(p["sig"][i]) if i < len(p["sig"]) else 1.8
                    st.markdown(f"**Mode {i + 1}**")
                    c1, c2, c3 = st.columns(3)
                    # ranges are generous — presets go up to N=3200, σ=3.3, κ=1.6
                    N_list.append(c1.number_input(
                        "N (cm⁻³)", 0.0, 1e5, dN, 10.0, key=f"parcel_N{i}"))
                    mu_list.append(c2.number_input(
                        "radius (µm)", 0.001, 5.0, dmu, 0.001, format="%.3f",
                        key=f"parcel_mu{i}"))
                    sig_list.append(c3.number_input(
                        "geom σ", 1.05, 4.0, dsig, 0.05, key=f"parcel_sig{i}"))
                kappa = st.number_input("κ (hygroscopicity, all modes)", 0.01, 2.0,
                                        kappa_pre, 0.05, key="parcel_kappa")
                N_raw, mu_um, sig = tuple(N_list), tuple(mu_list), tuple(sig_list)
        else:
            N_raw, mu_um, sig, kappa = p["N_raw"], p["mu_um"], p["sig"], p["kappa"]
        # GCCN = a few GIANT nuclei added as an extra mode; amount + size adjustable
        gccn = st.checkbox("Add GCCN (giant CCN)", False,
                           help="A few giant sea-salt nuclei that seed warm rain.")
        if gccn:
            g1, g2 = st.columns(2)
            gN = g1.number_input("GCCN amount N (cm⁻³)", 0.0, 10.0, 0.01, 0.01,
                                 format="%.3f", key="parcel_gccn_N",
                                 help="Giant nuclei are sparse — typically ≪1 cm⁻³.")
            gr = g2.number_input("GCCN radius (µm)", 0.5, 10.0, 2.0, 0.1,
                                 key="parcel_gccn_r")
            n_base = len(N_raw)  # before appending the coarse mode
            N_raw = tuple(N_raw) + (float(gN),)
            mu_um = tuple(mu_um) + (float(gr),)
            sig = tuple(sig) + (1.5,)
            kappa = ((kappa,) * n_base + (1.2,)
                     if np.isscalar(kappa) else tuple(kappa) + (1.2,))

        st.subheader("Physics")
        collisions = st.checkbox("Collision–coalescence", True,
                                 help="Required for rain to form.")
        sedi_removal = st.checkbox("Rain falls out (sedimentation removal)", True,
                                   help="Rain-sized drops sink at terminal velocity and "
                                        "EXIT the ~100 m parcel within seconds–minutes, "
                                        "accumulating as precipitation. Off = a closed "
                                        "box whose rain stays and sweeps up the whole "
                                        "population (unbuffered supersaturation).")
        switch_TICE = st.checkbox("Turbulent collision enhancement (TICE)", False,
                                  help="Wang–Ayala turbulent collision kernel.")
        eps = (st.number_input("ε (m²/s³)", 0.0, 0.5, 0.01, 0.01)
               if switch_TICE else 0.0)

        st.subheader("Entrainment")
        ent_on = st.checkbox("Entrain environmental air", False,
                             help="Mix in outside air during a chosen window of the "
                                  "ascent — dries the parcel and evaporates cloud water.")
        lambda_ent = st.slider("Strength λ (m⁻¹)", 1e-4, 2e-3, 5e-4, 1e-4,
                               format="%.4f", disabled=not ent_on,
                               help="Fraction of the parcel replaced per metre risen "
                                    "(λ·w·dt per step). Realistic cumulus ~0.0002–0.002.")
        rh_env = st.slider("Environment RH", 0.0, 1.0, 0.2, 0.05, disabled=not ent_on,
                           help="How DRY the entrained air is. 20% = very dry free "
                                "troposphere; higher RH entrains gentler air.")
        c1, c2 = st.columns(2)
        ent_start_min = c1.number_input("Starts at (min)", 0.0, float(sim_min),
                                        min(5.0, float(sim_min)), 1.0,
                                        disabled=not ent_on)
        ent_dur_min = c2.number_input("Lasts (min)", 1.0, float(sim_min),
                                      min(10.0, float(sim_min)), 1.0,
                                      disabled=not ent_on)
        ihmd = st.slider("IHMD (mixing degree)", 0.0, 1.0, 0.5, 0.1,
                         disabled=not ent_on,
                         help="0 homogeneous (every droplet shrinks, number kept); "
                              "1 inhomogeneous (whole droplets evaporate, survivors "
                              "keep their size).")
        if ent_on:
            from droplab.mixing import entrained_fraction
            _ef = entrained_fraction(lambda_ent, w, dt, ent_dur_min * 60.0)
            st.caption(f"≈ {_ef*100:.0f}% of the parcel replaced by {rh_env*100:.0f}%-RH "
                       f"air between min {ent_start_min:g} and "
                       f"{ent_start_min + ent_dur_min:g}.")

    _lam = lambda_ent if ent_on else 0.0
    with st.spinner("Running parcel ascent…"):
        out, M, A = cache.run_parcel(
            0, n_ptcl, nt, dt, T0, P0, RH, w, ascending_mode,
            tuple(N_raw), tuple(mu_um), tuple(sig),
            kappa if np.isscalar(kappa) else tuple(kappa),
            collisions, switch_TICE, eps, _lam, ihmd,
            rh_env=rh_env, ent_start=ent_start_min * 60.0,
            ent_duration=ent_dur_min * 60.0, sedi_removal=sedi_removal)

    last = out[sorted(out)[-1]]
    m = st.columns(5)
    m[0].metric("N_c (cm⁻³)", f"{last['NC']:.1f}",
                help="Cloud-droplet number concentration at the end of the ascent.")
    m[1].metric("N_r (cm⁻³)", f"{last['NR']:.3f}",
                help="Raindrop number concentration — nonzero means collisions "
                     "made rain.")
    m[2].metric("Mean radius (µm)", f"{last['rv']:.2f}")
    m[3].metric("LWC q_c+q_r (g/kg)", f"{last['qc'] + last['qr']:.3f}",
                help="Liquid water content: cloud + rain mixing ratio.")
    m[4].metric("Supersaturation (%)", f"{(last['RH'] - 1) * 100:+.3f}",
                help="Water vapour above saturation. Droplets normally BUFFER this "
                     "near ~0.1–1% by condensing the excess; it can only run higher "
                     "if the droplet population is gone.")
    # The closed-parcel rain-out endgame: once collection has eaten (almost) the whole
    # population, nothing condenses the excess vapour and S climbs unboundedly. Real
    # parcels drop their rain out and entrain fresh CCN — say so, or a student reads
    # the S spike as a model bug.
    if (last["NA"] + last["NC"]) < 1.0 and (last["RH"] - 1) * 100 > 1.5:
        st.info("**The cloud has rained out.** Collision–coalescence collected "
                "essentially the whole droplet (and aerosol) population "
                f"(N_a+N_c = {last['NA'] + last['NC']:.2f} cm⁻³), so nothing is left "
                "to condense the excess vapour and supersaturation is no longer "
                "buffered — that is why S climbs at the end. An idealized closed "
                "parcel keeps its rain and gets no fresh aerosol; a real one "
                "wouldn't. Try a shorter run, more aerosol, or entrainment.")

    runs = [(preset, out, M, A)]
    if compare_preset != "(none)" and compare_preset != preset:
        cp = presets.AEROSOL_PRESETS[compare_preset]
        with st.spinner(f"Running the {compare_preset} overlay…"):
            out2, M2, A2 = cache.run_parcel(
                0, n_ptcl, nt, dt, T0, P0, RH, w, ascending_mode,
                tuple(float(x) for x in cp["N_raw"]),
                tuple(float(x) for x in cp["mu_um"]),
                tuple(float(x) for x in cp["sig"]),
                float(cp["kappa"]) if np.isscalar(cp["kappa"]) else tuple(cp["kappa"]),
                collisions, switch_TICE, eps, _lam, ihmd,
                rh_env=rh_env, ent_start=ent_start_min * 60.0,
                ent_duration=ent_dur_min * 60.0, sedi_removal=sedi_removal)
        runs.append((compare_preset, out2, M2, A2))
    tabs = st.tabs(["Time series", "DSD", "Particle population", "Vertical profiles"])
    with tabs[0]:
        theme.whatami("Six-panel parcel evolution: humidity, vapour, height, "
                      "temperature, the three mixing ratios, and number "
                      "concentrations (log axis).")
        st.plotly_chart(plots.parcel_timeseries(runs, dt), use_container_width=True)
    with tabs[1]:
        theme.whatami("Droplet size distribution. Growth narrows the spectrum; "
                      "collisions broaden it toward rain sizes.")
        st.plotly_chart(plots.parcel_dsd_contour(out, dt), use_container_width=True)
    with tabs[2]:
        theme.whatami("Each marker is one super-droplet at its radius; size/colour "
                      "scale with its multiplicity A.")
        st.plotly_chart(plots.parcel_particles(M, A), use_container_width=True)
    with tabs[3]:
        theme.whatami("Vertical structure of the ascent.")
        st.plotly_chart(plots.parcel_profiles(runs), use_container_width=True)


# ========================================================================= #
# 2-D — the cloud you can watch (the showcase)
# ========================================================================= #
def _scenario_picker():
    """Grouped scenario selectbox. Returns the scenario key."""
    options, labels = [], {}
    for grp in presets.GROUPS:
        for key, meta in presets.SCENARIOS.items():
            if meta["group"] == grp and not meta.get("hidden"):
                options.append(key)
                labels[key] = f"{grp} · {meta['label']}"
    return st.selectbox("Scenario (environment)", options,
                        format_func=lambda k: labels[k], key="twod_scenario")


def render_twod():
    theme.apply("2-D", "⚡")

    # apply a pending one-click demo BEFORE widgets are created so their defaults
    # pick it up via session_state.
    demo = st.session_state.pop("pending_demo", None)
    if demo and demo.get("scenario"):
        st.session_state["twod_scenario"] = demo["scenario"]
        t = demo["toggles"]
        st.session_state["twod_ice"] = t.get("ice", False)
        st.session_state["twod_habit"] = t.get("habit", False)
        st.session_state["twod_elec"] = t.get("electrification", False)
        st.session_state["twod_coll"] = t.get("collisions", True)
        st.session_state["twod_autorun"] = True

    theme.header("02 · Showcase", "2-D cloud",
                 "Compose a cloud, choose its microphysics, and watch it evolve. "
                 "The view adapts to what you turned on — ice, crystal habit, "
                 "lightning.", accent=theme.MODE_ACCENT["twod"])

    with st.sidebar:
        st.subheader("Build your cloud")
        scenario = _scenario_picker()
        smeta = presets.SCENARIOS[scenario]
        st.caption(smeta["blurb"])

        micro = controls.microphysics_panel(
            scenario, "twod", ice0=smeta["ice_default"])
        # per-SCENARIO aerosol keys + the case's own validated N as the default —
        # a shared key let one scenario's slider silently contaminate another
        # (e.g. a low-N experiment made BOMEX/DYCOMS drizzle heavily).
        _bN = float(presets.base_config(scenario)["N_modes"][0])
        N_modes, mu_um, sig, kappa = controls.aerosol_two_mode(
            f"twod_{scenario}", default_N=_bN)

        # duration (decoupled from grid) — defaults follow the scenario, keyed by
        # scenario so switching gives fresh physics-tuned defaults.
        with st.expander("⏱️ Duration", expanded=True):
            # minimum 60 simulated minutes: shorter runs kept ending while the cloud
            # was still forming (clouds need their full life cycle to read).
            sim_min = st.slider("Simulated time (min)", 60,
                                max(120, int(2 * smeta["default_min"])),
                                int(smeta["default_min"]), 1,
                                key=f"twod_min_{scenario}",
                                help="How long to evolve the cloud. Defaults are "
                                     "tuned to actually form a cloud.")
            dt = st.number_input("dt (s)", 0.25, float(smeta["dt_default"] * 1.5),
                                 float(smeta["dt_default"]), 0.25,
                                 key=f"twod_dt_{scenario}",
                                 help="Time step. Lower = more stable but slower.")
        nt = max(1, round(sim_min * 60 / dt))

        seed_on, seed_kind, seed_N, seed_r, inject_min = controls.seeding_panel(
            "twod", run_min=sim_min)

        # dynamics / advanced (scenario-aware)
        wind_shear, dtheta_bubble = 0.0, None
        inp_n_cm3 = micro.get("inp_n_cm3")
        inp_r_um = micro.get("inp_r_um")
        E_breakdown, charge_eff = 400.0, 0.3
        _bubble = {"idealized", "congestus", "deep_cold", "deep_convection"}
        with st.expander("🌀 Dynamics & advanced"):
            # wind shear is a general option now (>0 auto-forces periodic walls).
            # The slider MAX is depth-aware: U(z)=shear*(z-Z/2) peaks at shear*Z/2, so
            # the same dU/dz that is mild on a 2-km domain is a ~40 m/s jet on a 12-km
            # deep domain (unstable even at reduced dt). Cap so U_max <= ~12 m/s.
            _Z = float(presets.sized_config(scenario, "quick")["Z"])
            _shear_max = round(min(6.0e-3, 24.0 / _Z), 4)
            wind_shear = st.slider("Wind shear dU/dz (s⁻¹)", 0.0, _shear_max, 0.0,
                                   5.0e-4, format="%.4f", key=f"twod_shear_{scenario}",
                                   help="Tilts the updraft into bands. Any value >0 "
                                        "forces periodic side walls (and, on deep "
                                        "domains, automatically reduces dt for CFL).")
            if scenario in _bubble:
                dtheta_bubble = st.slider(
                    "Warm-bubble strength Δθ (K)", 0.5, 5.0,
                    float(presets.base_config(scenario).get("dtheta_bubble", 2.5)),
                    0.1, key=f"twod_bub_{scenario}")
            if scenario == "deep_convection":
                st.caption("Anelastic core is forced for this scenario "
                           "(Boussinesq caps the tower ~2.6 km).")
            if micro["electrification"]:
                charge_eff = st.slider("Charge separation efficiency", 0.05, 0.6,
                                       0.3, 0.05)
                E_breakdown = st.slider("Breakdown field (illustrative)", 200.0,
                                        1500.0, 400.0, 50.0,
                                        help="Lower → flashes fire more readily in "
                                             "the idealized 2-D field.")

        st.subheader("Run & view")
        resolution = st.radio("Grid resolution", ["quick", "full"], horizontal=True,
                              key="twod_res",
                              help="quick: a coarse grid for a fast first click. "
                                   "full: the validated CASES grid (slow). Run "
                                   "length is set above, independently.")
        show_field = st.checkbox("Show q_c field tiles", True)
        wind = st.radio("Wind overlay", ["off", "streamlines", "arrows"],
                        horizontal=True)
        run = st.button("▶ Run cloud", type="primary", use_container_width=True)
        st.caption("quick ≈ ½–1 min live, full ≈ 2–5 min. Repeats of the same "
                   "settings are cached and instant.")

    if run or st.session_state.pop("twod_autorun", False):
        st.session_state["twod_active"] = True
    if not st.session_state.get("twod_active"):
        st.info("Pick a scenario and microphysics on the left, then press "
                "**▶ Run cloud** — or open a curated demo from Home.")
        # scenario teaser: tell the user what THIS pick will show before they commit
        st.markdown(f"#### {smeta['label']}")
        st.markdown(smeta["blurb"])
        return

    twod_args = (
        scenario, resolution, nt, dt, micro["collisions"], micro["ice"],
        micro["habit"], micro["electrification"], micro["freezing_mode"],
        micro["homogeneous"], micro["melt"], micro["hallett_mossop"],
        N_modes, mu_um, sig, kappa, seed_on, seed_kind, seed_N, seed_r,
        inject_min, wind_shear, dtheta_bubble, inp_n_cm3, inp_r_um,
        E_breakdown, charge_eff)

    if cache.twod_is_cached(*twod_args):
        result = cache.run_twod(*twod_args)          # already computed → instant
    else:
        # Watch it compute LIVE — frames stream in as the engine produces them;
        # afterwards the run is cached and replays as a loop on the next identical
        # config. Better than staring at a spinner for the long runs.
        st.caption(f"Simulating **{scenario}** ({resolution}) live — frames appear "
                   "as they are computed, then this run is cached and loops.")
        live = st.empty()
        bar = st.progress(0.0, text="starting…")
        qmax = [0.5]
        _draw_fails = [0]

        def _cb(step, total, frame, flow):
            try:
                qmax[0] = max(qmax[0], float(frame["qc"].max()))
                fig = plots.live_frame_fig(flow, frame, scenario, dt,
                                           show_field, wind, qmax[0])
                live.pyplot(fig, clear_figure=True)
                plots.close(fig)      # clear_figure clears but doesn't free it
            except Exception:
                # a draw hiccup must never abort the physics run — but if drawing
                # is SYSTEMATICALLY broken, say so instead of looking frozen
                _draw_fails[0] += 1
                if _draw_fails[0] == 3:
                    live.caption("Live preview unavailable — the simulation is "
                                 "still computing in the background.")
            try:
                bar.progress(min(1.0, step / max(1, total)),
                             text=f"step {step} / {total}")
            except Exception:
                pass

        result = cache.run_twod(*twod_args, on_frame=_cb)
        bar.empty()
        live.empty()

    if result.get("unstable"):
        st.error("⚠️ This configuration went **numerically unstable** (NaN). "
                 "Try the quick-look resolution, gentler dynamics, or fewer/larger "
                 "aerosols.")
        return

    _render_twod_result(result, show_field, wind)


def _render_twod_result(result, show_field, wind):
    met = result["metrics"]
    cols = st.columns(5)
    cols[0].metric("Peak cloud water", f"{result['qc_max']:.2f} g/kg",
                   help="Densest liquid anywhere in the scene.")
    cols[1].metric("Cloud fraction", f"{met['cloud_fraction']:.0%}",
                   help="Share of the domain that is cloudy.")
    cols[2].metric("Effective radius", f"{met['reff_um']:.1f} µm",
                   help="Optics-weighted mean droplet size — smaller drops make a "
                        "brighter cloud.")
    cols[3].metric("Cloud albedo", f"{met['albedo']:.2f}",
                   help="Fraction of sunlight reflected — higher = brighter/cooler.")
    if result["meta"].get("electrification"):
        cols[4].metric("Lightning flashes", f"{result['n_flashes']}",
                       help="Discharges fired this run (illustrative model).")
    else:
        cols[4].metric("Surface rain", f"{result['surf_precip']:.2e} kg",
                       help="Liquid that fell out of the domain.")

    views = plots.regime_views(result)
    tab_titles = [t for _, t, _ in views]
    tabs = st.tabs(tab_titles)
    for tab, (key, title, caption) in zip(tabs, views):
        with tab:
            theme.whatami(caption)
            if key == "scene":
                top = st.columns([4, 1])
                with top[1]:
                    animate = st.toggle("Animate", value=True, key="twod_anim",
                                        help="Loop from frame 0 with the graphs "
                                             "growing in sync with the cloud. Turn "
                                             "off to freeze: final scene + full "
                                             "interactive graphs.")
                if animate:
                    st.image(plots.scene_and_series_gif(result, show_field, wind),
                             use_container_width=True,
                             caption="Cloud scene + key variables building together, "
                                     "in sync (loops from frame 0).")
                else:
                    # frozen mode = a frame SCRUBBER: step through the run by hand
                    frames = result["frames"]
                    n_fr = len(frames)
                    dt_m = result["meta"]["dt"]
                    k = st.slider("Frame", 0, n_fr - 1, n_fr - 1, 1,
                                  key="twod_frame",
                                  help="Drag to move through the run frame by frame.")
                    t_k = frames[k]["step"] * dt_m
                    st.image(plots.scene_image(result, show_field, wind,
                                               frame_idx=k),
                             use_container_width=True,
                             caption=f"Frame {k + 1}/{n_fr} — t = {t_k / 60:.1f} min.")
                    st.markdown("**Key variables over time**")
                    fig_ts = plots.twod_timeseries(result)
                    fig_ts.add_vline(x=t_k, line=dict(color="#E8743B", width=1.5,
                                                      dash="dot"))
                    st.plotly_chart(fig_ts, use_container_width=True)
            elif key == "phase":
                st.image(plots.phase_image(result), use_container_width=True)
            elif key == "bergeron":
                st.plotly_chart(plots.bergeron_figure(result),
                                use_container_width=True)
            elif key == "habit":
                st.image(plots.habit_image(result), use_container_width=True)
            elif key == "electric":
                st.image(plots.electric_image(result), use_container_width=True)
                st.caption("Illustrative 2-D discharge (dielectric-breakdown "
                           "model) — qualitative, not a calibrated flash rate.")
                if result["n_flashes"] == 0:
                    st.caption("No bolt fired in this quick-look — the charge "
                               "dipole is still building. Lower the breakdown "
                               "field or run full resolution for a discharge.")


# ========================================================================= #
# CLIMATE — the planetary thermostat (MCB twin)
# ========================================================================= #
def render_climate():
    theme.apply("Climate", "🌊")
    st.session_state.pop("pending_demo", None)  # demo just routes here; no preset state
    theme.header("03 · Thermostat", "Climate intervention",
                 "A marine stratocumulus deck as an aerosol-mediated state. Seed "
                 "it and compare to an unseeded twin: more, smaller droplets "
                 "brighten the cloud (Twomey).", accent=theme.MODE_ACCENT["climate"])

    with st.sidebar:
        st.subheader("1 · Background deck")
        background = st.selectbox(
            "Cloud regime", ["DYCOMS stratocumulus", "BOMEX cumulus",
                             "Arctic mixed-phase"],
            help="Sc = the classic marine sunshade (MCB target). BOMEX = shallow "
                 "trade cumulus. Arctic = the MOSAiC supercooled deck, where "
                 "GLACIOGENIC INP seeding is the intervention.")
        background_N = st.slider("Background aerosol N₀ (cm⁻³)", 10.0, 500.0,
                                 200.0, 10.0,
                                 help="Clean marine ~20; polluted ~400.")
        st.subheader("2 · Entrainment mixing")
        ihmd = st.slider("Inhomogeneous mixing degree (IHMD)", 0.0, 1.0, 0.0, 0.1,
                         help="How dry-air mixing removes droplets. 0 = every droplet "
                              "shrinks a little (number kept); 1 = whole droplets "
                              "evaporate (number drops, survivors keep their size).")
        st.subheader("3 · Deliberate seeding")
        seed_on = st.checkbox("Seed the cloud", value=True)
        _kinds = (["Glaciogenic INP (ice)"] if background == "Arctic mixed-phase"
                  else [k for k in controls.SEED_KINDS if "INP" not in k])
        seed_kind = st.selectbox("Strategy", _kinds,
                                 disabled=not seed_on, key=f"clim_kind_{background}")
        seed_N, seed_r = controls.seed_amount_size("clim", seed_kind,
                                                   disabled=not seed_on)
        st.subheader("4 · Run length")
        run_choice = st.select_slider("How long to simulate",
                                      list(presets.CLIMATE_RUN_STEPS),
                                      value=presets.CLIMATE_RUN_DEFAULT,
                                      help="Long enough for the seeding effect to "
                                           "develop — not end right after injection.")
        nt = presets.CLIMATE_RUN_STEPS[run_choice]
        run_min = nt / 60.0   # dt = 1 s in the climate deck
        inject_min = st.slider("Inject at (simulated min)", 0.0, round(run_min, 1),
                               0.0,
                               step=max(0.5, round(run_min / 40, 1)),
                               disabled=not seed_on, key="clim_inject",
                               help="When the seeding fires — early, so the effect "
                                    "has time to develop before the run ends.")
        compare = st.checkbox("Compare to an unseeded twin (control)",
                              value=True, disabled=not seed_on,
                              help="Runs a second unseeded deck: TOA forcing plus a "
                                   "dotted control baseline on every graph.")
        run = st.button("▶ Run deck", type="primary", use_container_width=True)
        st.caption("First run computes live — roughly ½–1 min for the deck, doubled "
                   "with the control twin. Repeats are cached and instant.")

    Nx, Nz, n_super = 64, 40, 30000
    if not run:
        st.info("Set the controls on the left, then press **▶ Run deck**.")
        return

    # SEEDED deck first, streaming live, so the user sees frames immediately;
    # the quiet control twin runs afterwards (its baseline is only needed by the
    # comparison overlays, which all render later anyway).
    clim_args = (background_N, ihmd, seed_on, seed_kind, seed_N, seed_r,
                 inject_min, nt, Nx, Nz, n_super, 1.0, background)
    if cache.climate_is_cached(*clim_args):
        out = cache.run_climate(*clim_args)          # already computed → instant
    else:
        st.caption("Simulating the deck live — frames appear as computed, then it "
                   "is cached and loops.")
        live = st.empty()
        bar = st.progress(0.0, text="starting…")
        _draw_fails = [0]

        def _cb(step, total, frame, flow):
            try:
                fig = plots.live_frame_fig(flow, frame, "", 1.0, True, "off",
                                           max(0.3, float(frame["qc"].max())))
                live.pyplot(fig, clear_figure=True)
                plots.close(fig)      # clear_figure clears but doesn't free it
            except Exception:
                # protect the physics run from a draw hiccup — but if drawing is
                # SYSTEMATICALLY broken, say so instead of looking frozen.
                _draw_fails[0] += 1
                if _draw_fails[0] == 3:
                    live.caption("Live preview unavailable — the simulation is "
                                 "still computing in the background.")
            try:
                bar.progress(min(1.0, step / max(1, total)),
                             text=f"step {step} / {total}")
            except Exception:
                pass

        out = cache.run_climate(*clim_args, on_frame=_cb)
        bar.empty()
        live.empty()

    if out.get("unstable"):
        st.error("⚠️ This deck went **numerically unstable** (NaN). Try fewer "
                 "seed particles, a weaker background change, or the default "
                 "run length.")
        return

    # control (unseeded) twin AFTER the seeded run — the user has already seen the
    # live frames, so this wait is contextualized (and skipped when cached).
    twin = None
    if seed_on and compare:
        ctrl_args = (background_N, ihmd, False, seed_kind, seed_N, seed_r,
                     inject_min, nt, Nx, Nz, n_super, 1.0, background)
        if cache.climate_is_cached(*ctrl_args):
            twin = cache.run_climate(*ctrl_args)
        else:
            with st.spinner("Running the unseeded control twin (for the dotted "
                            "baseline and the forcing estimate)…"):
                twin = cache.run_climate(*ctrl_args)
        if twin.get("unstable"):
            st.warning("The unseeded control twin went unstable — showing the "
                       "seeded run without the comparison overlay.")
            twin = None
    ctrl_ts = twin["ts"] if twin is not None else None

    # FULL-WIDTH result (the combined scene+graphs image is wide — don't squeeze it
    # into a narrow column), with the headline metrics in a row underneath.
    theme.whatami("Stratocumulus deck (seeded droplets ringed magenta) with the "
                  "MCB metrics — N_d, albedo, CRE — building in sync. The dotted "
                  "line is the unseeded control.")
    animate = st.toggle("Animate", value=True, key="clim_anim",
                        help="Loop the deck with the graphs growing in sync. "
                             "Turn off to freeze: final deck + interactive graphs.")
    if animate:
        st.image(plots.climate_scene_series_gif(out, ctrl_ts=ctrl_ts),
                 use_container_width=True,
                 caption="Deck + MCB metrics building together (loops from frame 0).")
    else:
        st.image(out["png"], use_container_width=True, caption="Final deck (frozen).")
        st.plotly_chart(plots.climate_timeseries(out["ts"], ctrl=ctrl_ts),
                        use_container_width=True)

    st.subheader("What the cloud did")
    m = st.columns(4)
    m[0].metric("Effective radius", f"{out['reff_um']:.1f} µm", help="Smaller → brighter.")
    m[1].metric("Cloud albedo", f"{out['albedo']:.2f}",
                help="Fraction of sunlight the deck reflects — higher = brighter "
                     "= more cooling.")
    m[2].metric("Surface rain", f"{out['precip_kg']:.2e} kg")
    m[3].metric("Droplet number ΣA", f"{out['droplet_number']:.1e}")
    if twin is not None:
        d_albedo = out["albedo_mean"] - twin["albedo_mean"]
        dF = cache.climate_forcing(d_albedo)
        f = st.columns(4)
        f[0].metric("Estimated TOA forcing", f"{dF:+.1f} W/m²",
                    delta=f"Δalbedo = {d_albedo:+.3f}", delta_color="off",
                    help="Negative = cooling. An idealized model estimate, not a "
                         "real-world MCB forcing.")
        if dF < 0:
            st.success(f"This seeding **cooled** the column by about "
                       f"{abs(dF):.1f} W/m² of reflected sunlight.")


# ========================================================================= #
# LECTURE — guided, narrated lessons
# ========================================================================= #
_LESSONS = {
    "1 · Aerosol activation": dict(
        text="As the parcel rises it cools and supersaturation grows. Once it "
             "crosses each particle's critical value, that haze particle "
             "*activates* into a cloud droplet. Watch Nc climb with updraft/RH.",
        eqn=r"S_{crit} \propto \sqrt{A^3 / B}",
        notice="Raise the updraft and watch more aerosol activate into droplets.",
        run=dict(collisions=False, preset="default")),
    "2 · Condensational growth": dict(
        text="With collisions OFF, droplets grow only by vapour diffusion. Small "
             "droplets grow faster than large ones (dr/dt ∝ 1/r), so the spectrum "
             "**narrows** with height.",
        eqn=r"\frac{dr}{dt} = \frac{G\,S}{r}",
        notice="The DSD spectrum tightens as the parcel rises.",
        run=dict(collisions=False, preset="default")),
    "3 · Collision → rain": dict(
        text="With collisions ON (maritime aerosol), larger droplets fall faster "
             "and collect smaller ones. The distribution **broadens** and a rain "
             "mode (Nr, qr) develops.",
        eqn=r"K(r_1,r_2) = \pi (r_1+r_2)^2\,|v_1-v_2|\,E",
        notice="A second peak appears at large radii — that is rain.",
        run=dict(collisions=True, preset="maritime")),
    "4 · Maritime vs continental": dict(
        text="Same updraft, different aerosol: maritime air has *few large* CCN "
             "that rain quickly; continental air has *many small* CCN that make "
             "many tiny droplets and **suppress** rain at the same water content.",
        eqn=r"N_c \uparrow \Rightarrow \bar r \downarrow \Rightarrow \text{rain suppressed}",
        notice="Compare the two final spectra — narrow vs broad.",
        run=dict(collisions=True, preset="compare")),
    "5 · Entrainment mixing (IHMD)": dict(
        text="Dry environmental air is entrained and evaporates cloud water. "
             "*Homogeneous* mixing (IHMD=0) shrinks every droplet but keeps the "
             "number; *inhomogeneous* (IHMD=1) evaporates whole droplets but keeps "
             "the survivors' size.",
        eqn=r"N_c / N_{c,0} = \left(q_c / q_{c,0}\right)^{\mathrm{IHMD}}",
        notice="IHMD=0 keeps Nc while r shrinks; IHMD=1 drops Nc.",
        run=dict(collisions=False, preset="ihmd")),
}


def render_lecture():
    from app.ui import lessons as _showcase

    theme.apply("Lecture", "📘")
    theme.header("04 · Guided", "Lecture mode",
                 "Narrated, linear lessons. Pick one, read what to notice, then "
                 "step through the fixed scenario.", accent=theme.MODE_ACCENT["lecture"])

    # curriculum order (EDU_FRAMEWORK §4): warm parcel 5 → mixed-phase 4 → showcase 4
    # group the flat lesson list so students see the intended progression
    _warm = list(_LESSONS)
    _mixed = [k for k in _showcase.ALL_LESSONS if k.split(" ")[0].startswith("I")]
    _show = [k for k in _showcase.ALL_LESSONS if k not in _mixed]
    _group = {**{k: "Warm parcel" for k in _warm},
              **{k: "Mixed-phase" for k in _mixed},
              **{k: "Showcase" for k in _show}}
    _all = _warm + _mixed + _show
    lesson = st.sidebar.selectbox(
        "Lesson", _all, format_func=lambda k: f"{_group[k]} · {k}",
        help="Warm parcel 1-5 build the fundamentals; Mixed-phase I1-I4 add ice; "
             "Showcase lessons tour the headline 2-D phenomena.")
    if lesson in _showcase.ALL_LESSONS:
        _showcase.render_showcase(lesson)
        return
    L = _LESSONS[lesson]
    nt, dt, n_ptcl = 1500, 1.0, 1500
    T0, P0, RH, w = 293.2, 1013.0e2, 0.92, 1.0

    st.subheader(lesson)
    st.markdown(L["text"])
    st.latex(L["eqn"])
    st.info(f"**What to notice:** {L['notice']}")

    spec = L["run"]
    if spec["preset"] == "compare":
        with st.spinner("Running maritime and continental parcels…"):
            runs = []
            for nm in ("maritime", "continental"):
                p = presets.AEROSOL_PRESETS[nm]
                o, M, A = cache.run_parcel(0, n_ptcl, nt, dt, T0, P0, RH, w,
                                           "linear", p["N_raw"], p["mu_um"],
                                           p["sig"], p["kappa"], True, False, 0.0,
                                           0.0, 0.0)
                runs.append((nm, o, M, A))
    elif spec["preset"] == "ihmd":
        with st.spinner("Running entrainment-mixing sweep…"):
            p = presets.AEROSOL_PRESETS["default"]
            runs = []
            for val in (0.0, 0.5, 1.0):
                o, M, A = cache.run_parcel(0, n_ptcl, nt, dt, T0, P0, RH, w,
                                           "linear", p["N_raw"], p["mu_um"],
                                           p["sig"], p["kappa"], False, False, 0.0,
                                           5e-4, val)
                runs.append((f"IHMD={val:g}", o, M, A))
    else:
        with st.spinner("Running parcel ascent…"):
            p = presets.AEROSOL_PRESETS[spec["preset"]]
            o, M, A = cache.run_parcel(0, n_ptcl, nt, dt, T0, P0, RH, w, "linear",
                                       p["N_raw"], p["mu_um"], p["sig"], p["kappa"],
                                       spec["collisions"], False, 0.0, 0.0, 0.0)
            runs = [(spec["preset"], o, M, A)]

    tab1, tab2 = st.tabs(["Droplet size distribution", "Time series"])
    with tab1:
        if len(runs) == 1:
            st.plotly_chart(plots.parcel_dsd_contour(runs[0][1], dt),
                            use_container_width=True)
        else:
            import plotly.graph_objects as go
            fig = go.Figure()
            for label, o, *_ in runs:
                ts, radii, stack, _ = plots._dsd_stack(o)
                y = stack[-1].copy(); y[y <= 0] = np.nan
                fig.add_trace(go.Scatter(x=radii, y=y, mode="lines", name=label))
            fig.update_xaxes(type="log", title_text="Radius r (µm)")
            fig.update_yaxes(type="log", title_text="dN (cm⁻³)")
            fig.update_layout(height=520, title="Final DSD overlay")
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.plotly_chart(plots.parcel_timeseries(runs, dt), use_container_width=True)
