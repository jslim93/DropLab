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
    # LIVE reactivity (박사's call): the parcel is fast, so every widget change
    # recomputes immediately — cause -> effect with no extra click.
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
                      "scale with its multiplicity A. Drag the slider to any moment "
                      "of the ascent.")
        _ks = sorted(out)
        _kt = st.select_slider("Time (min)", options=_ks, value=_ks[-1],
                               format_func=lambda k: f"{k * dt / 60:.1f}",
                               key="parcel_pop_t")
        _snapM = out[_kt].get("M_snap"); _snapA = out[_kt].get("A_snap")
        if _snapM is not None:
            st.plotly_chart(plots.parcel_particles(np.asarray(_snapM),
                                                   np.asarray(_snapA)),
                            use_container_width=True)
        else:   # older cached runs without snapshots -> final state
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
        sc = demo["scenario"]
        st.session_state["twod_scenario"] = sc
        t = demo["toggles"]
        # widget keys are scenario-scoped (controls.py: f"twod_{scenario}_ice" etc.);
        # habit/electrification have no widgets — the UI derives them from ice.
        st.session_state[f"twod_{sc}_ice"] = t.get("ice", False)
        st.session_state[f"twod_{sc}_coll"] = t.get("collisions", True)
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
        # background field shaded behind the droplets — a pure RENDER choice (the
        # fields are already in the cached frames, so switching never re-runs).
        _bg = st.radio("Background field", ["q_c", "T", "q_v", "none"],
                       horizontal=True, key="twod_bg",
                       help="What to shade behind the droplets. T and q_v come from "
                            "the same cached run as q_c — no recompute. 'none' = "
                            "particles only.")
        _BG = {"q_c": "qc", "T": "T", "q_v": "qv", "none": "none"}
        bg_field = _BG[_bg]
        show_field = bg_field != "none"
        wind = st.radio("Wind overlay", ["off", "streamlines", "arrows"],
                        horizontal=True)
        run = st.button("▶ Run cloud", type="primary", use_container_width=True)
        st.caption("quick ≈ ½–1 min live, full ≈ 2–5 min. Repeats of the same "
                   "settings are cached and instant.")

    # The config is CAPTURED when ▶ Run cloud is pressed; rendering always uses the
    # captured config. Widget changes after a run are inert until the next press —
    # they must NOT silently trigger a fresh (expensive) 2-D computation.
    _args_now = (
        scenario, resolution, nt, dt, micro["collisions"], micro["ice"],
        micro["habit"], micro["electrification"], micro["freezing_mode"],
        micro["homogeneous"], micro["melt"], micro["hallett_mossop"],
        N_modes, mu_um, sig, kappa, seed_on, seed_kind, seed_N, seed_r,
        inject_min, wind_shear, dtheta_bubble, inp_n_cm3, inp_r_um,
        E_breakdown, charge_eff)
    if run or st.session_state.pop("twod_autorun", False):
        st.session_state["twod_cfg"] = _args_now
    twod_args = st.session_state.get("twod_cfg")
    if twod_args is None:
        st.info("Pick a scenario and microphysics on the left, then press "
                "**▶ Run cloud** — or open a curated demo from Home.")
        # scenario teaser: tell the user what THIS pick will show before they commit
        st.markdown(f"#### {smeta['label']}")
        st.markdown(smeta["blurb"])
        return
    if twod_args != _args_now:
        st.caption("⚙️ Settings changed — showing the LAST run. Press **▶ Run cloud** "
                   "to apply the new settings.")
    # the captured config drives everything below (incl. the live-stream labels)
    scenario, resolution, dt = twod_args[0], twod_args[1], twod_args[3]

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

    _render_twod_result(result, show_field, wind, bg_field)


def _render_twod_result(result, show_field, wind, bg_field="qc"):
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
                    theme.animated_gif(
                        plots.scene_and_series_gif(result, show_field, wind,
                                                   field=bg_field),
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
                                               frame_idx=k, field=bg_field),
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
                                 200.0, 5.0,
                                 help="Clean marine ~20-65 (drizzling: seeding "
                                      "brightens strongly); ~200 typical; polluted "
                                      "~400 (already bright, seeding adds little).")
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
        # key includes the run length: a stored value above the new max raises
        # StreamlitValueAboveMaxError when the user shortens the run (the 2-D page's
        # inject slider already keys by run_min for exactly this reason).
        inject_min = st.slider("Inject at (simulated min)", 0.0, round(run_min, 1),
                               0.0,
                               step=max(0.5, round(run_min / 40, 1)),
                               disabled=not seed_on, key=f"clim_inject_{run_min:g}",
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
    # Same capture-on-Run pattern as the 2-D page: widget changes after a run are
    # inert (and the last result STAYS on screen) until ▶ Run deck is pressed again.
    _clim_now = (background_N, ihmd, seed_on, seed_kind, seed_N, seed_r,
                 inject_min, nt, Nx, Nz, n_super, 1.0, background)
    if run:
        st.session_state["clim_cfg"] = _clim_now
    clim_args = st.session_state.get("clim_cfg")
    if clim_args is None:
        st.info("Set the controls on the left, then press **▶ Run deck**.")
        return
    if clim_args != _clim_now:
        st.caption("⚙️ Settings changed — showing the LAST run. Press **▶ Run deck** "
                   "to apply the new settings.")
    (background_N, ihmd, seed_on, seed_kind, seed_N, seed_r,
     inject_min, nt, Nx, Nz, n_super, _dt_c, background) = clim_args

    # Both decks stream LIVE, side by side, as they compute (sequentially): the
    # seeded deck first, then the unseeded control — so the control is visibly running
    # instead of a silent spinner that reads as "frozen". Each run draws into its own
    # placeholder via an on_frame callback.
    _want_twin = bool(seed_on and compare)

    def _stream_cb(ph, bar):
        fails = [0]

        def _cb(step, total, frame, flow):
            try:
                fig = plots.live_frame_fig(flow, frame, "", 1.0, True, "off",
                                           max(0.3, float(frame["qc"].max())))
                ph.pyplot(fig, clear_figure=True)
                plots.close(fig)      # clear_figure clears but doesn't free it
            except Exception:
                fails[0] += 1
                if fails[0] == 3:
                    ph.caption("Live preview unavailable — still computing.")
            try:
                bar.progress(min(1.0, step / max(1, total)),
                             text=f"step {step} / {total}")
            except Exception:
                pass
        return _cb

    if _want_twin:
        _lc = st.columns(2)
        _lc[0].caption("seeded deck"); _lc[1].caption("control (unseeded)")
        ph_seed, ph_ctrl = _lc[0].empty(), _lc[1].empty()
    else:
        ph_seed, ph_ctrl = st.empty(), None

    # --- seeded deck ---
    if cache.climate_is_cached(*clim_args):
        out = cache.run_climate(*clim_args)          # already computed → instant
    else:
        if ph_ctrl is not None:
            ph_ctrl.info("queued — runs right after the seeded deck")
        bar = st.progress(0.0, text="seeded deck…")
        out = cache.run_climate(*clim_args, on_frame=_stream_cb(ph_seed, bar))
        bar.empty()

    if out.get("unstable"):
        st.error("⚠️ This deck went **numerically unstable** (NaN). Try fewer "
                 "seed particles, a weaker background change, or the default "
                 "run length.")
        return

    # --- unseeded control twin (also streamed live, into its own column) ---
    twin = None
    if _want_twin:
        ctrl_args = (background_N, ihmd, False, seed_kind, seed_N, seed_r,
                     inject_min, nt, Nx, Nz, n_super, 1.0, background)
        if cache.climate_is_cached(*ctrl_args):
            twin = cache.run_climate(*ctrl_args)
        else:
            bar = st.progress(0.0, text="control deck…")
            twin = cache.run_climate(*ctrl_args, on_frame=_stream_cb(ph_ctrl, bar))
            bar.empty()
        if twin.get("unstable"):
            st.warning("The unseeded control twin went unstable — showing the "
                       "seeded run without the comparison overlay.")
            twin = None

    # clear the live previews before the final looping animation replaces them
    ph_seed.empty()
    if ph_ctrl is not None:
        ph_ctrl.empty()
    ctrl_ts = twin["ts"] if twin is not None else None

    # FULL-WIDTH result (the combined scene+graphs image is wide — don't squeeze it
    # into a narrow column), with the headline metrics in a row underneath.
    _dual = twin is not None
    theme.whatami("Stratocumulus deck (seeded droplets ringed magenta)"
                  + (" beside the unseeded control deck" if _dual else "")
                  + " with the MCB metrics — N_d, albedo, CRE — building in sync. "
                  "The dotted line is the unseeded control.")
    animate = st.toggle("Animate", value=True, key="clim_anim",
                        help="Loop the deck with the graphs growing in sync. "
                             "Turn off to freeze: final deck + interactive graphs.")
    if animate:
        theme.animated_gif(
            plots.climate_scene_series_gif(out, ctrl_ts=ctrl_ts, ctrl_result=twin),
            caption=("Seeded vs control deck + MCB metrics building together "
                     "(loops from frame 0)." if _dual else
                     "Deck + MCB metrics building together (loops from frame 0)."))
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
    # Warm-parcel lessons follow the same Predict→Observe→Explain arc as the
    # mixed-phase/showcase lessons, with ONE live control each: the lesson's
    # "notice" line must always name something the student can actually turn.
    "1 · Aerosol activation": dict(
        objective="predict how the activated droplet number N_c responds to "
                  "updraft speed, and explain why via peak supersaturation.",
        text="As the parcel rises it cools and supersaturation grows. Once it "
             "crosses each particle's critical value, that haze particle "
             "*activates* into a cloud droplet. Watch Nc climb with updraft/RH.",
        eqn=r"S_{crit} \propto \sqrt{A^3 / B}",
        notice="Raise the updraft and watch more aerosol activate into droplets.",
        predict=("If you double the updraft speed w, the activated droplet "
                 "number N_c will…",
                 ["Increase — faster cooling → higher peak S → smaller aerosol activate",
                  "Stay the same — N_c is fixed by the aerosol population",
                  "Decrease — the parcel has less time to activate"], 0),
        knob=dict(label="Updraft w (m/s)", lo=0.2, hi=5.0, default=1.0,
                  step=0.1, param="w"),
        run=dict(collisions=False, preset="default")),
    "2 · Condensational growth": dict(
        objective="see that diffusional growth *narrows* the droplet spectrum, "
                  "because dr/dt ∝ 1/r lets small drops catch up.",
        text="With collisions OFF, droplets grow only by vapour diffusion. Small "
             "droplets grow faster than large ones (dr/dt ∝ 1/r), so the spectrum "
             "**narrows** with height.",
        eqn=r"\frac{dr}{dt} = \frac{G\,S}{r}",
        notice="Change the updraft and watch the spectrum-width σ_r: growth "
               "tightens the DSD no matter how fast you rise.",
        predict=("With collisions off, as the parcel rises the spectral width "
                 "σ_r will…",
                 ["Narrow — small drops grow fastest and catch up",
                  "Broaden — big drops pull ahead",
                  "Stay constant — every drop grows at the same rate"], 0),
        knob=dict(label="Updraft w (m/s)", lo=0.2, hi=5.0, default=1.0,
                  step=0.1, param="w"),
        run=dict(collisions=False, preset="default")),
    "3 · Collision → rain": dict(
        objective="connect aerosol number to rain formation: more CCN → smaller "
                  "droplets → collision–coalescence stalls.",
        text="With collisions ON (maritime aerosol), larger droplets fall faster "
             "and collect smaller ones. The distribution **broadens** and a rain "
             "mode (Nr, qr) develops.",
        eqn=r"K(r_1,r_2) = \pi (r_1+r_2)^2\,|v_1-v_2|\,E",
        notice="Pollute the air (more small CCN) and watch the total rain "
               "collapse; clean it (×0.25) and the parcel rains out completely.",
        predict=("Make the air heavily polluted — ×8 more small CCN (same "
                 "updraft, same water). The total rain produced will…",
                 ["Decrease — many small droplets collide too slowly to make rain",
                  "Increase — more droplets means more collisions",
                  "Stay the same — rain only depends on total water"], 0),
        knob=dict(label="Small-CCN (accumulation-mode) number ×", lo=0.25,
                  hi=8.0, default=1.0, step=0.25, param="n_mult"),
        run=dict(collisions=True, preset="maritime")),
    "4 · Maritime vs continental": dict(
        objective="run the SAME updraft through two real aerosol regimes and "
                  "identify which one rains — the aerosol–precipitation link.",
        text="Same updraft, different aerosol: maritime air has *few large* CCN "
             "that rain quickly; continental air has *many small* CCN that make "
             "many tiny droplets and **suppress** rain at the same water content.",
        eqn=r"N_c \uparrow \Rightarrow \bar r \downarrow \Rightarrow \text{rain suppressed}",
        notice="Compare the two final spectra — narrow vs broad — and try a "
               "different updraft: does the contrast survive?",
        predict=("Same updraft, same water content: which parcel makes rain "
                 "first?",
                 ["Maritime — few large CCN grow big enough to collide",
                  "Continental — more droplets, more collisions",
                  "Both at the same time — rain depends only on water content"], 0),
        knob=dict(label="Updraft w (m/s), applied to BOTH parcels", lo=0.2,
                  hi=5.0, default=1.0, step=0.1, param="w"),
        run=dict(collisions=True, preset="compare")),
    "5 · Entrainment mixing (IHMD)": dict(
        objective="distinguish homogeneous from inhomogeneous mixing by what "
                  "each one does to N_c and mean radius.",
        text="Dry environmental air is entrained and evaporates cloud water. "
             "*Homogeneous* mixing (IHMD=0) shrinks every droplet but keeps the "
             "number; *inhomogeneous* (IHMD=1) evaporates whole droplets but keeps "
             "the survivors' size.",
        eqn=r"N_c / N_{c,0} = \left(q_c / q_{c,0}\right)^{\mathrm{IHMD}}",
        notice="Turn the entrainment strength up and compare the three IHMD "
               "curves: 0 keeps Nc while r shrinks; 1 drops Nc.",
        predict=("Which mixing scenario keeps the droplet NUMBER but shrinks "
                 "every drop?",
                 ["Homogeneous (IHMD=0) — all drops share the evaporation",
                  "Inhomogeneous (IHMD=1) — whole drops evaporate",
                  "Neither — entrainment always removes drops"], 0),
        knob=dict(label="Entrainment strength λ (s⁻¹)", lo=1e-4, hi=2e-3,
                  default=5e-4, step=1e-4, param="lambda_ent", fmt="%.4f"),
        run=dict(collisions=False, preset="ihmd")),
}


def render_lecture():
    from app.ui import lessons as _showcase

    theme.apply("Lecture", "📘")
    theme.header("04 · Guided", "Lecture mode",
                 "Narrated lessons: read the objective, commit to a prediction, "
                 "then turn the lesson's control and watch the model answer.",
                 accent=theme.MODE_ACCENT["lecture"])
    st.caption("🚧 Lecture mode is still under active development — lessons, "
               "controls, and self-checks are being expanded. For free "
               "experimentation use the Parcel / 2-D / Climate sandboxes.")

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
    st.markdown(f"🎯 **Objective** — by the end of this lesson you can "
                f"{L['objective']}")
    st.markdown(L["text"])
    st.latex(L["eqn"])

    # ② Predict — commit before seeing the run (same POE pattern as the
    # mixed-phase lessons; index=None forces an actual choice, no default)
    q, opts, correct = L["predict"]
    st.markdown(f"**Predict** — {q}")
    pred = st.radio("Your prediction:", opts, index=None,
                    key=f"lec_pred_{lesson}", label_visibility="collapsed")

    # ③ the lesson's ONE live control — re-runs immediately (cached per value)
    k = L["knob"]
    knob_val = st.slider(f"🎛 Try it: {k['label']}", k["lo"], k["hi"],
                         k["default"], k["step"], key=f"lec_knob_{lesson}",
                         format=k.get("fmt", "%g"))
    if k["param"] == "w":
        w = float(knob_val)
    n_mult = float(knob_val) if k["param"] == "n_mult" else 1.0
    lam = float(knob_val) if k["param"] == "lambda_ent" else 5e-4

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
                                           lam, val)
                runs.append((f"IHMD={val:g}", o, M, A))
    else:
        with st.spinner("Running parcel ascent…"):
            p = presets.AEROSOL_PRESETS[spec["preset"]]
            # the pollution knob scales ONLY the small accumulation mode —
            # scaling the large (sea-salt) mode too would ADD rain embryos and
            # invert the suppression signal (verified: x4 all-modes raised q_r)
            N_run = (p["N_raw"][0] * n_mult,) + tuple(p["N_raw"][1:])
            o, M, A = cache.run_parcel(0, n_ptcl, nt, dt, T0, P0, RH, w, "linear",
                                       N_run, p["mu_um"], p["sig"], p["kappa"],
                                       spec["collisions"], False, 0.0, 0.0, 0.0)
            runs = [(spec["preset"], o, M, A)]

    # outcome metrics — the numbers that move when the knob moves
    def _last(out):
        return out[max(out)]

    def _peak(out, key):
        return max(float(v.get(key, 0.0)) for v in out.values())

    mcols = st.columns(len(runs) if len(runs) > 1 else 2)
    if spec["preset"] == "compare":
        for c, (nm, o, *_r) in zip(mcols, runs):
            total_rain = float(_last(o).get("precip", 0.0)) + _last(o)["qr"]
            c.metric(f"Total rain — {nm}", f"{total_rain:.2f} g/kg",
                     delta=f"peak N_c {_peak(o, 'NC'):.0f} cm⁻³",
                     delta_color="off")
    elif spec["preset"] == "ihmd":
        for c, (nm, o, *_r) in zip(mcols, runs):
            c.metric(f"Final N_c — {nm}", f"{_last(o)['NC']:.0f} cm⁻³",
                     delta=f"mean r {float(_last(o).get('rv', 0.0)):.1f} µm",
                     delta_color="off")
    else:
        o = runs[0][1]
        mcols[0].metric("Peak activated N_c", f"{_peak(o, 'NC'):.0f} cm⁻³")
        if spec["collisions"]:
            # fallen + still-in-parcel: in-parcel q_r alone misses rain that
            # already sedimented out and reads non-monotonic vs aerosol
            total_rain = float(_last(o).get("precip", 0.0)) + _last(o)["qr"]
            mcols[1].metric("Total rain produced (fallen + in parcel)",
                            f"{total_rain:.2f} g/kg")
        else:
            mcols[1].metric("Final spectral width σ_r",
                            f"{float(_last(o).get('rv_std', 0.0)):.2f} µm")

    if pred is not None:
        mark = "✅" if pred == opts[correct] else "❌"
        st.info(f"{mark} Your prediction: **{pred}** — the model shows: "
                f"**{opts[correct]}**")

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
