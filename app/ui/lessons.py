"""Showcase lessons for Lecture mode — H1 ice habit, S1 snow, D1 deep
convection, L1 lightning (docs/LECTURE_MODE_CONTENT.md §B, implemented to the
6-step Lesson Pattern of docs/EDU_FRAMEWORK.md §1: Frame → Predict → Observe →
Explain → Refine → Self-check).

Design decisions:
- Lesson configs are FIXED (curated) — that separates Lecture from the sandbox.
- H1 and S1 deliberately SHARE one deep_cold run (ice+habit) so the pair costs a
  single cached simulation.
- ALL runs go through ``cache.run_twod`` (process-cached, golden-safe couplings
  enforced). D1's Refine reuses L1's no-ice counterfactual run (shared cache).
  NOTE: the design doc's original D1 Refine (anelastic-vs-Boussinesq cloud-top
  contrast) did NOT reproduce — with the strong 5 K bubble + capped-CAPE
  CUMULONIMBUS sounding the Boussinesq core also reaches ~9 km (verified at
  b_max 0.6 AND the default 0.12), so the honest Refine is the ice on/off
  lifecycle contrast instead.
- Honesty rule (EDU_FRAMEWORK): teach sign and mechanism, not digits; the DBM
  bolt is labelled an illustrative visualization (ELECTRIFICATION_AUDIT.md).

No physics here — fixed configs in, returned frames/fields out.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from app.ui import cache, plots


# --------------------------------------------------------------------------- #
# fixed lesson runs (quick grids, <~30 s cold, instant when cached)
# --------------------------------------------------------------------------- #
def _run_cold_storm():
    """The shared H1+S1 run: deep cold storm, ice + habit, 30 simulated min."""
    return cache.run_twod(
        scenario="deep_cold", resolution="quick", nt=900, dt=2.0,
        collisions=True, ice=True, habit=True, electrification=False,
        freezing_mode="abifm", homogeneous=True, melt=True, hallett_mossop=True,
        N_modes=(150.0,), mu_um=(0.08,), sig=(2.0,), kappa=(0.6,),
        seed_on=False, seed_kind="MCB sea-salt", seed_N=200.0, seed_r=0.1,
        inject_min=None, wind_shear=0.0, dtheta_bubble=None,
        inp_n_cm3=0.5, inp_r_um=3.0, E_breakdown=400.0, charge_eff=0.3)


def _run_thunderstorm(ice=True):
    """L1: the validated lightning-demo config (ice=False = the Refine
    counterfactual: no ice → no charging → no bolt)."""
    return cache.run_twod(
        scenario="deep_convection", resolution="quick", nt=560, dt=4.0,
        collisions=True, ice=ice, habit=False, electrification=ice,
        freezing_mode="abifm", homogeneous=True, melt=True, hallett_mossop=True,
        N_modes=(200.0,), mu_um=(0.08,), sig=(2.0,), kappa=(0.6,),
        seed_on=False, seed_kind="MCB sea-salt", seed_N=200.0, seed_r=0.1,
        inject_min=None, wind_shear=0.0, dtheta_bubble=None,
        inp_n_cm3=0.5, inp_r_um=3.0, E_breakdown=400.0, charge_eff=0.3)


def _run_cumulonimbus():
    """D1: the full anelastic cumulonimbus, 60 simulated min."""
    return cache.run_twod(
        scenario="deep_convection", resolution="quick", nt=900, dt=4.0,
        collisions=True, ice=True, habit=False, electrification=False,
        freezing_mode="abifm", homogeneous=True, melt=True, hallett_mossop=True,
        N_modes=(150.0,), mu_um=(0.08,), sig=(2.0,), kappa=(0.6,),
        seed_on=False, seed_kind="MCB sea-salt", seed_N=200.0, seed_r=0.1,
        inject_min=None, wind_shear=0.0, dtheta_bubble=None,
        inp_n_cm3=0.5, inp_r_um=3.0, E_breakdown=400.0, charge_eff=0.3)


def _cloud_top_series(result):
    dz = result["meta"]["dz"]
    fr = result["frames"]
    return ([f["step"] * result["meta"]["dt"] for f in fr],
            [plots._cloud_top(f, dz) for f in fr])


def _run_arctic(inp, nt=1800):
    """Mixed-phase lesson base: the MOSAiC Arctic deck (~−15…−24 °C) for 30
    simulated min at a given INP loading. inp=0 → the CLEAN cloud of I1."""
    return cache.run_twod(
        scenario="arctic", resolution="quick", nt=nt, dt=1.0,
        collisions=True, ice=True, habit=False, electrification=False,
        freezing_mode="abifm", homogeneous=True, melt=True, hallett_mossop=True,
        N_modes=(60.0,), mu_um=(0.08,), sig=(2.0,), kappa=(0.6,),
        seed_on=False, seed_kind="MCB sea-salt", seed_N=200.0, seed_r=0.1,
        inject_min=None, wind_shear=0.0, dtheta_bubble=None,
        inp_n_cm3=inp, inp_r_um=4.0, E_breakdown=400.0, charge_eff=0.3)


def _run_deep_clean():
    """I1 Refine: deep convection with NO INP — the only ice source left is
    homogeneous freezing, which needs T below ≈−38 °C (high in the tower)."""
    return cache.run_twod(
        scenario="deep_convection", resolution="quick", nt=560, dt=4.0,
        collisions=True, ice=True, habit=False, electrification=False,
        freezing_mode="abifm", homogeneous=True, melt=True, hallett_mossop=True,
        N_modes=(150.0,), mu_um=(0.08,), sig=(2.0,), kappa=(0.6,),
        seed_on=False, seed_kind="MCB sea-salt", seed_N=200.0, seed_r=0.1,
        inject_min=None, wind_shear=0.0, dtheta_bubble=None,
        inp_n_cm3=0.0, inp_r_um=4.0, E_breakdown=400.0, charge_eff=0.3)


def _wp(result):
    """(LWP_end, IWP_end, ice fraction) domain sums from the last frame."""
    f = result["frames"][-1]
    lwp = float(f.get("q_liquid", f["qc"]).sum())
    iwp = float(f.get("q_ice", np.zeros(1)).sum())
    return lwp, iwp, iwp / max(lwp + iwp, 1e-9)


# --------------------------------------------------------------------------- #
# the 6-step renderer
# --------------------------------------------------------------------------- #
def _predict(key, question, options):
    """POE Predict step: commit to a prediction before seeing the run."""
    st.markdown(f"**② Predict** — {question}")
    return st.radio("Your prediction:", options, index=None,
                    key=f"lesson_pred_{key}", label_visibility="collapsed")


def _selfcheck(key, question, answer):
    st.markdown("**⑥ Self-check**")
    with st.expander(question):
        st.markdown(answer)


def _honesty(text):
    st.caption(f"⚖️ *Honesty note: {text}*")


# --------------------------------------------------------------------------- #
# lessons
# --------------------------------------------------------------------------- #
def _lesson_h1():
    st.markdown("**① Frame** — a crystal's *shape* — flat **plate** vs long "
                "**column** — is set by **temperature**, not chance.")
    pred = _predict("h1", "At −15 °C vs −6 °C, will crystals grow *flatter* or *longer*?",
                    ["−15 °C flatter (plates), −6 °C longer (columns)",
                     "−15 °C longer (columns), −6 °C flatter (plates)",
                     "Shape is random — temperature doesn't matter"])
    with st.spinner("③ Observe — growing a cold storm (cached after first run)…"):
        r = _run_cold_storm()
    st.markdown("**③ Observe** — the crystal-shape gallery, coloured by aspect "
                "ratio φ = c/a (φ<1 plate ▮, φ>1 column ▯):")
    st.image(plots.habit_image(r), use_container_width=True)
    phi = r["frames"][-1].get("phi")
    if phi is not None:
        grown = phi[(phi > 0) & (phi != 1.0)]
        if grown.size:
            st.metric("Plates vs columns (final frame)",
                      f"{(grown < 1).mean():.0%} plates · {(grown > 1).mean():.0%} columns")
    st.markdown("**④ Explain** — the inherent growth ratio Γ(T) (Chen–Lamb, "
                "cross-validated vs SAM-LCM): near **−15 °C** vapour deposits "
                "preferentially on the basal face → **plates** (Γ<1); near "
                "**−6 °C** and below ~**−22 °C** → **columns** (Γ>1). Same "
                "vapour, temperature-dependent growth anisotropy — the Nakaya "
                "diagram. Growth is capacitance-based deposition.")
    st.latex(r"\frac{dc}{da} \propto \Gamma(T)\quad(\Gamma<1:\text{ plate},"
             r"\ \Gamma>1:\text{ column})")
    if pred is not None:
        st.info("Your prediction: **" + pred + "** — correct is: −15 °C → plates, "
                "−6 °C → columns.")
    st.markdown("**⑤ Refine** — this storm spans −3…−28 °C, so BOTH habits "
                "coexist at different heights. Look again: plates cluster in the "
                "−10…−20 °C band, columns near cloud base (−3…−8 °C) and top "
                "(<−22 °C).")
    _selfcheck("h1", "Why does identical vapour make different shapes at different T?",
               "Deposition is *anisotropic*: the relative growth rate of the basal "
               "vs prism crystal faces (Γ(T)) flips sign with temperature, so the "
               "same supersaturation elongates the c-axis at one temperature and "
               "the a-axis at another. Shape records the temperature history.")


def _lesson_s1():
    st.markdown("**① Frame** — cold clouds precipitate **ice (snow)**, which "
                "falls far slower than rain.")
    pred = _predict("s1", "In this sub-freezing storm, will the precipitation "
                          "reaching the surface be liquid or ice?",
                    ["Ice (snow) — the whole cloud is below 0 °C",
                     "Liquid rain — precipitation is always rain",
                     "Nothing falls — ice stays in the cloud"])
    with st.spinner("③ Observe — cold storm (shared with the habit lesson)…"):
        r = _run_cold_storm()
    st.markdown("**③ Observe** — liquid vs ice condensate, and the ice "
                "sedimenting toward the ground:")
    st.image(plots.phase_image(r), use_container_width=True)
    st.plotly_chart(plots.twod_timeseries(r), use_container_width=True)
    st.metric("Accumulated surface precipitation", f"{r['surf_precip']:.2e} kg",
              help="In this cloud that is snow — sedimenting ice, not rain.")
    st.markdown("**④ Explain** — crystals grow by vapour deposition (WBF), then "
                "sediment at a habit-dependent fall speed (Böhm / "
                "Locatelli–Hobbs) — much slower than an equal-mass raindrop. "
                "“Snow” in the model is exactly this **sedimenting q_ice**; there "
                "is no separate snow species.")
    if pred is not None:
        st.info(f"Your prediction: **{pred}** — correct: ice. The whole column "
                "is sub-freezing, so nothing melts on the way down.")
    st.markdown("**⑤ Refine** — a ~1 mm raindrop falls at ~4–6 m/s (≈3 min from "
                "1 km); a dendritic snowflake at ~0.5–1 m/s (≈20–30 min). Same "
                "mass, ~10× the fall time — watch the ice in the animation drift "
                "rather than plummet. *(Textbook magnitudes, quoted not computed.)*")
    _selfcheck("s1", "Why does snow take so much longer to reach the ground than rain?",
               "Ice crystals are far less dense per cross-section (open habits, "
               "low apparent density), so drag balances gravity at a much lower "
               "terminal velocity than for a compact liquid drop of equal mass.")


def _lesson_d1():
    st.markdown("**① Frame** — a cumulonimbus has a **lifecycle**: buoyant tower "
                "→ glaciating anvil → snow → dissipation.")
    pred = _predict("d1", "As the tower rises past the 0 °C level, what happens "
                          "to its water — and will it reach the tropopause?",
                    ["It freezes (glaciates) and the tower can reach ~10 km",
                     "It stays liquid all the way up",
                     "The cloud stops growing at the 0 °C level"])
    with st.spinner("③ Observe — 60 min of deep convection (cached after first run)…"):
        r = _run_cumulonimbus()
    st.markdown("**③ Observe** — the tower rising, glaciating into an anvil, and "
                "snowing out (scene + cloud-top / liquid-vs-ice series in sync):")
    st.image(plots.scene_and_series_gif(r), use_container_width=True)
    st.markdown("**④ Explain** — latent-heat release drives the buoyant tower; "
                "the **anelastic** core treats the height-varying base density "
                "ρ₀(z), so the same condensed mass is a larger mixing-ratio "
                "increment aloft — realistic deep towers need it. Past the 0 °C "
                "level the water glaciates (q_liquid → q_ice); the slow-falling "
                "ice spreads as an **anvil** and sediments out as snow.")
    if pred is not None:
        st.info(f"Your prediction: **{pred}** — correct: it glaciates and tops "
                "near ~10 km.")
    t_an, z_an = _cloud_top_series(r)
    st.metric("Cloud-top height reached", f"{max(z_an) / 1000:.1f} km")
    st.markdown("**⑤ Refine** — the SAME storm with **ice turned off** (an "
                "idealized counterfactual): does the lifecycle still complete?")
    with st.spinner("Running the no-ice counterfactual (shared with the "
                    "lightning lesson)…"):
        ctl = _run_thunderstorm(ice=False)
    iwp_max = max(float(f.get("q_ice", np.zeros(1)).max()) for f in r["frames"])
    c1, c2 = st.columns(2)
    c1.metric("Peak ice content — with ice", f"{iwp_max:.1f} g/kg",
              help="The glaciating anvil: slow-falling crystals spread and "
                   "sediment as snow.")
    c2.metric("Peak ice content — no ice", "0 g/kg",
              help="The tower still rises, but there is no anvil glaciation and "
                   "no snow — precipitation falls out quickly as rain. The icy "
                   "half of the lifecycle is missing.")
    st.caption("Same bubble, same sounding — only the ice microphysics differs.")
    _selfcheck("d1", "Why does the cloud turn to ice near its top, not its base?",
               "Temperature falls with height; the upper tower sits far below "
               "0 °C where freezing (immersion + homogeneous) and vapour "
               "deposition convert liquid to ice, while the base is still warm.")


def _lesson_l1():
    st.markdown("**① Frame** — a thundercloud charges by **ice-particle "
                "collisions**; lightning is the discharge of that charge.")
    pred = _predict("l1", "What separates the charge — colliding raindrops, or "
                          "graupel–ice-crystal collisions? And where do + and − end up?",
                    ["Graupel–crystal collisions; + on crystals aloft, − on graupel below",
                     "Raindrop collisions; + at cloud base",
                     "The ground induces the charge from below"])
    with st.spinner("③ Observe — charging a cumulonimbus (cached after first run)…"):
        r = _run_thunderstorm(ice=True)
    st.markdown("**③ Observe** — the charge dipole building and the discharge:")
    st.image(plots.electric_image(r), use_container_width=True)
    st.metric("Lightning flashes in this run", f"{r['n_flashes']}")
    _honesty("the charging (Saunders non-inductive) is physically grounded; the "
             "bolt is a dielectric-breakdown *visualization* on the static "
             "field, not a simulated leader — see docs/ELECTRIFICATION_AUDIT.md.")
    st.markdown("**④ Explain** — rebounding graupel↔crystal collisions transfer "
                "charge whose sign flips at a reversal temperature (Saunders "
                "et al.); gravity then separates heavy graupel (down, −) from "
                "lofted crystals (up, +) → a vertical **dipole** → Gauss-law "
                "field → dielectric breakdown → the leader.")
    st.latex(r"\nabla^2 \phi = -\rho_q/\varepsilon_0 \;\rightarrow\; "
             r"|E| > E_{crit}(z) \Rightarrow \text{breakdown}")
    if pred is not None:
        st.info(f"Your prediction: **{pred}** — correct: graupel–crystal "
                "collisions; crystals carry + aloft, graupel − below.")
    st.markdown("**⑤ Refine** — the SAME storm with **ice turned off** (an "
                "idealized counterfactual — the tower stays liquid):")
    with st.spinner("Running the no-ice counterfactual…"):
        r0 = _run_thunderstorm(ice=False)
    c1, c2 = st.columns(2)
    c1.metric("Flashes — with ice", f"{r['n_flashes']}")
    c2.metric("Flashes — no ice", f"{r0['n_flashes']}",
              help="No graupel–crystal collisions → no charge separation → no "
                   "lightning. Warm clouds don't thunder.")
    _selfcheck("l1", "Why does a warm (ice-free) cloud not make lightning?",
               "The non-inductive charging mechanism needs *rebounding* "
               "graupel–crystal collisions in the presence of supercooled "
               "liquid. No ice → no charge separation → no field → no "
               "breakdown, however hard it rains.")


def _lesson_i1():
    st.markdown("**① Frame** — a cloud below 0 °C is usually **not** all ice.")
    pred = _predict("i1", "Cool a CLEAN cloud (no ice-nucleating particles) to "
                          "−15…−24 °C. What phase is it?",
                    ["All ice — it's below freezing",
                     "Supercooled liquid — nothing triggers freezing",
                     "Half ice, half liquid"])
    with st.spinner("③ Observe — a clean Arctic deck at −15…−24 °C (cached)…"):
        r = _run_arctic(0.0)
    lwp, iwp, frac = _wp(r)
    st.markdown("**③ Observe** — the deck after 30 min, split into liquid vs ice:")
    st.image(plots.phase_image(r), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Liquid water (Σq_c)", f"{lwp:.0f} g/kg")
    c2.metric("Ice (Σq_i)", f"{iwp:.1f} g/kg",
              help="Zero: with no INP and T warmer than ≈−38 °C, nothing freezes.")
    st.markdown("**④ Explain** — the whole cloud is **supercooled liquid**. Pure "
                "droplets do not freeze at 0 °C; without an ice-nucleating "
                "particle (INP) they supercool until homogeneous freezing at "
                "≈−38 °C. A −15 °C cloud has no trigger — so it stays liquid. "
                "If you predicted \"all ice\", that is misconception **Mi1**.")
    if pred is not None:
        st.info(f"Your prediction: **{pred}** — correct: supercooled liquid.")
    st.markdown("**⑤ Refine** — the same CLEAN air in a cloud that reaches "
                "**below −38 °C** (a deep tower, still zero INP):")
    with st.spinner("Running the clean deep tower…"):
        rd = _run_deep_clean()
    _, iwp_d, _ = _wp(rd)
    c1, c2 = st.columns(2)
    c1.metric("Ice — clean Arctic deck (−24 °C)", "0 g/kg")
    c2.metric("Ice — clean deep tower (<−38 °C aloft)", f"{iwp_d:.1f} g/kg",
              help="Homogeneous freezing fires ONLY where the tower is colder "
                   "than ≈−38 °C — no INP needed there.")
    _selfcheck("i1", "A cloud at −15 °C is most likely…?",
               "Mixed-phase or supercooled liquid (+ perhaps some ice). Liquid "
               "persists far below 0 °C; glaciation needs a trigger — an INP, or "
               "≈−38 °C for homogeneous freezing.")


def _lesson_i2():
    st.markdown("**① Frame** — what makes a drop freeze: **heterogeneous** (an "
                "INP inside it) vs **homogeneous** (≈−38 °C, no INP needed).")
    pred = _predict("i2", "Starting from the supercooled deck of I1, add a few "
                          "INP. At what temperature does ice appear?",
                    ["Near 0 °C — freezing starts at the melting point",
                     "Well below 0 °C, and only where INP are present",
                     "Nothing freezes until −38 °C even with INP"])
    st.markdown("**③ Observe** — the SAME −15…−24 °C deck at four INP loadings "
                "(30 min each, cached):")
    rows = []
    for inp in (0.0, 0.05, 0.2, 0.5):
        with st.spinner(f"INP = {inp:g} cm⁻³ …"):
            lwp, iwp, frac = _wp(_run_arctic(inp))
        rows.append((inp, lwp, iwp, frac))
    cols = st.columns(4)
    for c, (inp, lwp, iwp, frac) in zip(cols, rows):
        c.metric(f"INP {inp:g} cm⁻³", f"{frac:.0%} ice",
                 help=f"Σq_liq {lwp:.0f} · Σq_ice {iwp:.0f} g/kg")
    st.markdown("**④ Explain** — the cloud is far below 0 °C in every run, yet "
                "ice appears **only when INP are added** (ABIFM immersion "
                "freezing) and grows with their number. Pure water does not "
                "freeze at its melting point — that is misconception **Mi4**; "
                "freezing needs a *nucleus* (or ≈−38 °C).")
    if pred is not None:
        st.info(f"Your prediction: **{pred}** — correct: well below 0 °C, and "
                "only where INP are present.")
    st.markdown("**⑤ Refine** — read the sweep: at what INP loading is this "
                "cloud roughly **half ice**? (Between 0.2 and 0.5 cm⁻³ — "
                f"{rows[2][3]:.0%} at 0.2 vs {rows[3][3]:.0%} at 0.5.)")
    _selfcheck("i2", "A pure droplet cooled below 0 °C freezes at about…?",
               "≈−38 °C (homogeneous freezing) — unless an ice-nucleating "
               "particle inside it triggers freezing at warmer temperatures.")


def _lesson_i3():
    st.markdown("**① Frame** — ice and supercooled liquid do **not** stably "
                "coexist: the ice grows at the droplets' expense.")
    pred = _predict("i3", "A mixed-phase cloud holds some ice crystals among many "
                          "supercooled drops. Over the next minutes…",
                    ["Ice grows while the droplets evaporate",
                     "They stay in balance — both phases coexist stably",
                     "The droplets freeze onto the crystals at 0 °C"])
    with st.spinner("③ Observe — a glaciating cold storm (shared run, cached)…"):
        r = _run_cold_storm()
    st.markdown("**③ Observe** — liquid- and ice-water paths over time (the "
                "hand-off):")
    st.plotly_chart(plots.bergeron_figure(r), use_container_width=True)
    st.markdown("**④ Explain** — **Wegener–Bergeron–Findeisen**: saturation "
                "vapour pressure over ice is LOWER than over liquid, so "
                "mixed-phase air is supersaturated w.r.t. ice while "
                "sub-saturated w.r.t. liquid — vapour deposits on the crystals "
                "as the droplets evaporate to feed them. Mixed-phase is "
                "*transient* unless something replenishes the liquid. "
                "\"They stay in balance\" is misconception **Mi2**.")
    st.latex(r"e_s^{ice}(T) < e_s^{liq}(T) \;\Rightarrow\; S_{ice}>1>S_{liq}")
    if pred is not None:
        st.info(f"Your prediction: **{pred}** — correct: ice grows, droplets "
                "evaporate.")
    st.markdown("**⑤ Refine** — so why do Arctic mixed-phase decks *persist for "
                "days*? Watch the SAME process in the MOSAiC deck, where the "
                "radiatively-driven overturning keeps **replenishing** the "
                "liquid:")
    with st.spinner("Running the persistent Arctic deck (cached)…"):
        ra = _run_arctic(0.2)
    lwp, iwp, _ = _wp(ra)
    c1, c2 = st.columns(2)
    c1.metric("Liquid after 30 min", f"{lwp:.0f} g/kg",
              help="Still there — condensation in the updrafts resupplies what "
                   "WBF takes.")
    c2.metric("Ice grown meanwhile", f"{iwp:.0f} g/kg")
    st.caption("The persistent-mixed-phase 'paradox': WBF always drains the "
               "liquid, but a steady supply of freshly condensed water keeps the "
               "deck alive — the real Arctic balance.")
    _selfcheck("i3", "Ice + supercooled drops, left alone over time…?",
               "The ice grows while the droplets evaporate (WBF). Coexistence "
               "lasts only while updrafts/cooling replenish the liquid faster "
               "than the ice consumes it.")


def _lesson_i4():
    st.markdown("**① Frame** — glaciogenic seeding: does more INP simply mean "
                "more snow?")
    pred = _predict("i4", "Sweep the Arctic deck's INP from clean to heavily "
                          "seeded (0 → 2 cm⁻³). After an hour, what happened?",
                    ["More INP → more snow, and the cloud is fine",
                     "More INP → more snow, but the cloud itself is consumed",
                     "More INP makes no difference below −38 °C"])
    st.markdown("**③ Observe** — snow at the surface AND the cloud that's left, "
                "after 60 min at three INP loadings (cached):")
    rows = []
    for inp in (0.2, 0.5, 2.0):
        with st.spinner(f"INP = {inp:g} cm⁻³ (60 min)…"):
            r = _run_arctic(inp, nt=3600)
        lwp, iwp, _ = _wp(r)
        rows.append((inp, r["surf_precip"], lwp))
    cols = st.columns(3)
    for c, (inp, snow, lwp) in zip(cols, rows):
        c.metric(f"INP {inp:g} cm⁻³", f"{snow:.1f} kg snow",
                 delta=f"liquid left: {lwp:.0f} g/kg", delta_color="off")
    st.markdown("**④ Explain** — seeding works: INP glaciate the supercooled "
                "liquid and it precipitates as snow (the mixed-phase analogue "
                "of rain seeding). But the snow **comes out of the cloud "
                "itself** — at 2 cm⁻³ the deck fully glaciates and "
                "*self-destructs* (liquid → 0). No cloud means no future snow "
                "and no radiative blanket. \"More INP → more snow, forever\" "
                "fails because the cloud dies — that is misconception **Mi3**.")
    _honesty("within this 60-min idealized run, cumulative snow still rises "
             "monotonically with INP — the seeding *penalty* appears here as "
             "the destroyed cloud (liquid → 0), not yet as reduced snow. The "
             "non-monotonic precipitation response develops on multi-hour "
             "cloud-lifetime scales, beyond this run's horizon.")
    if pred is not None:
        st.info(f"Your prediction: **{pred}** — the model shows: more snow, but "
                "the cloud is consumed.")
    st.markdown("**⑤ Refine** — you're designing a seeding operation: which "
                "loading would you pick to *harvest snow without killing the "
                "deck*? Read the liquid-left numbers: 0.2 keeps the deck alive "
                "(~72 g/kg), 0.5 nearly drains it (14), 2.0 destroys it (0). "
                "The trade-off IS the design problem.")
    _selfcheck("i4", "Greatly increasing INP does what?",
               "Glaciates the cloud and depletes its supercooled liquid — snow "
               "is gained at the cost of the cloud itself, and over a cloud's "
               "lifetime the precipitation response is non-monotonic (an "
               "over-seeded cloud stops snowing because it no longer exists).")


# --------------------------------------------------------------------------- #
# registry (ordering per LECTURE_MODE_CONTENT.md §C: habit → snow → deep → storm)
# --------------------------------------------------------------------------- #
MIXED_LESSONS = {
    "I1 · Supercooled liquid — below 0 °C is not all ice": dict(
        render=_lesson_i1,
        misconception="“Below 0 °C, clouds are ice.” (Mi1)"),
    "I2 · What makes a drop freeze": dict(
        render=_lesson_i2,
        misconception="“Drops freeze at 0 °C.” (Mi4)"),
    "I3 · Wegener–Bergeron–Findeisen — ice eats the liquid": dict(
        render=_lesson_i3,
        misconception="“Ice and liquid coexist in equilibrium.” (Mi2)"),
    "I4 · Glaciogenic seeding — snow costs the cloud": dict(
        render=_lesson_i4,
        misconception="“More INP → more ice → more snow, always.” (Mi3)"),
}

SHOWCASE_LESSONS = {
    "H1 · Ice habit — why snowflakes have shapes": dict(
        render=_lesson_h1,
        misconception="“Ice crystals are all alike / shape is random.”"),
    "S1 · Snow — precipitation can be ice": dict(
        render=_lesson_s1,
        misconception="“All precipitation is rain.”"),
    "D1 · Deep convection — the cumulonimbus lifecycle": dict(
        render=_lesson_d1,
        misconception="“Clouds are static / rain and snow come from different clouds.”"),
    "L1 · Lightning — charge comes from ice, not rain": dict(
        render=_lesson_l1,
        misconception="“Lightning comes from rain / from the ground.”"),
}


# curriculum spine order (EDU_FRAMEWORK §4): warm parcel → mixed-phase → showcase
ALL_LESSONS = {**MIXED_LESSONS, **SHOWCASE_LESSONS}


def render_showcase(name):
    """Render one mixed-phase or showcase lesson (6-step pattern)."""
    spec = ALL_LESSONS[name]
    st.subheader(name)
    st.caption(f"Target misconception: {spec['misconception']}")
    spec["render"]()
