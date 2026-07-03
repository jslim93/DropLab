# Instructor handbook — Climate Intervention chapter

A teaching companion for `notebooks/Climate_Intervention.ipynb` and
`app/streamlit_climate.py`. It gives you (1) ready-to-run session plans, (2) a
per-experiment instructor card with expected answers and student pitfalls, (3) the
self-check answer key, and — most importantly — (4) a **reusable template for
writing your own questions, rubrics, and assessments** so you can take the chapter
well beyond what we ship.

> Scope note: this covers the *climate-intervention* chapter specifically. The
> whole-tool instructor guide planned in `docs/PEDAGOGY_DESIGN.md §D` is a separate,
> broader artifact.

---

## 1. At a glance

| | |
|---|---|
| **Level** | Upper-undergraduate / early-graduate atmospheric science; also works as a non-specialist "geoengineering" module |
| **Prerequisites** | Saturation/condensation basics; what an aerosol/CCN is; albedo as "fraction of sunlight reflected". No coding required for the app; light Python literacy helps for the notebook. |
| **Core idea taught** | A cloud is not a fixed object — it is an *aerosol-mediated state*. MCB and rain-seeding are the same lever pulled opposite ways. |
| **Delivery options** | (a) Streamlit app projected, class predicts together; (b) students run the notebook in a lab; (c) flipped — students run at home, discuss in class. |
| **Frameworks** | Predict–Observe–Explain (POE); ICAP; constructive alignment; named misconceptions M1–M7. Citations in `docs/CLIMATE_INTERVENTION.md`. |

### Runtime budget (measured, so you can plan a lab slot)

The simulations are real physics. Approximate wall-clock on a modern laptop:

| Notebook cell | ~Time | Note |
|---|---|---|
| Exp 1 — deck baseline | ~15 s | |
| **Exp 2 — MCB brightening** | **~100 s** | the headline cell; full-resolution so the model's cooling signal is clean (a few W/m² — a genuine *model output*, not a real-world forcing) |
| Exp 3 — precip seeding | ~70 s | surface rain needs a long spin-up |
| Exp 4 — clean vs polluted | ~50 s | two decks |
| Exp 5 — entrainment IHMD | ~35 s | two runs |
| Exp 6 — completion challenge | ~13 s each attempt | students iterate |
| **Full "Run All"** | **~5–6 min** | plus first-run numba warm-up (~20 s extra the very first time) |

**Tip:** run the notebook once before class to warm the numba cache; subsequent
runs are at the times above. For the app, `@st.cache_data` makes repeated identical
slider settings instant.

---

## 2. Session plans

### A. Single 50-minute session (app-led, no coding)
1. **(8 min)** Motivate: show a satellite ship-track image; pose "can we do this on
   purpose to cool the planet?" Introduce the deck (Exp 1).
2. **(12 min)** MCB (Exp 2). Whole class **predicts** albedo direction on the app's
   Predict radio, then run. Reveal the model's cooling (a few W/m² — stress it is a
   *model output*, not a real-world number). Confront misconception M2 out loud.
3. **(10 min)** Rain seeding (Exp 3). Predict; run; emphasise the 71× rain/seed-mass
   honesty test (M3/M6).
4. **(10 min)** Challenge mode: hand the class the target "albedo 0.55, one knob."
   Let a student drive; the rest predict each step. (Completion rung.)
5. **(10 min)** Debrief with the wrap-up table; assign one homework prompt from §5.

### B. Two-session lab (notebook, with coding)
- **Session 1 (Exp 1–3):** foundations + the two interventions. Students fill in
  their predictions in the markdown before each run; collect the self-check answers.
- **Session 2 (Exp 4–6 + open panel):** background aerosol, entrainment mixing, the
  completion challenge, then 15 min of free exploration with a worksheet from §5.

---

## 3. Per-experiment instructor cards

Each card follows constructive alignment: **Learning Outcome ↔ activity ↔ what to
assess.** Expected numbers are from the shipped configuration; exact values vary a
little with the turbulence seed.

### Exp 1 — The marine stratocumulus deck
- **LO:** Identify a realistic Sc deck and the role of cloud-top radiative cooling.
- **Misconception:** (light) M1 — clouds need large supersaturation. Peak S is small.
- **Predict prompt:** clean deck (N₀=120) → small or large droplets? Bright or dim?
- **Expected:** r_eff ≈ 10–13 µm, albedo ≈ 0.5–0.6. A recognisable deck.
- **Assess:** can the student point to the inversion and explain why the cloud is
  *thin and flat* rather than a tall cumulus?
- **Pitfall:** students expect a "fluffy cloud" picture; stress this is a *sheet*.

### Exp 2 — Marine Cloud Brightening (M2)
- **LO:** Explain the Twomey effect: more CCN → more, smaller drops → higher albedo
  at fixed water → TOA cooling.
- **Misconception:** **M2** — "more aerosol → bigger drops." It is backwards.
- **Predict prompt:** which way does the r_eff histogram shift? Sign of TOA forcing?
- **Expected:** Δr_eff ≈ −0.2 µm, Δalbedo ≈ +0.02, **TOA forcing ≈ −5 W/m² in this
  run** — a slab-albedo estimate over a 2-D deck; a *model output* that varies with
  seed/resolution. **Teach the sign and mechanism, not the digits**; don't write
  "−5 W/m²" on the board as a real-world MCB forcing.
- **Assess (good answer contains):** same water ÷ more nuclei = smaller drops; smaller
  drops = more surface area = more scattering; cooling because a brighter cloud over a
  dark ocean reflects more sunlight.
- **Pitfall / honest point:** the per-column signal is noisy; only the *domain mean*
  is trustworthy (why ship-tracks appear in time-mean composites). Don't let students
  over-read a single column.

### Exp 3 — Precipitation seeding (M3, M6)
- **LO:** Distinguish condensation from collision–coalescence; explain how a few giant
  CCN tip a marginal deck into rain.
- **Misconceptions:** **M3** (condensation makes rain) and **M6** (giant CCN
  negligible); also **M5** (uniform growth) in the explanation.
- **Predict prompt:** does rain increase? Is the extra rain the *cloud's* water or the
  seed drops falling out?
- **Expected:** rain ≈ 4× unseeded; precipitated mass ≈ **70× the seed mass** ⇒ the
  cloud rained.
- **Assess:** student must invoke collision–coalescence (not "more condensation") and
  must understand why the rain/seed-mass ratio ≫ 1 is the proof.
- **Pitfall:** the two honest caveats — 20 µm "seeding" is just seed mass; a far-from-
  threshold deck can't be tipped. Probe these explicitly.

### Exp 4 — Clean vs polluted decks (M2 again, natural)
- **LO:** Connect background aerosol regime to droplet size, brightness, drizzle.
- **Misconception:** **M2** in its background form.
- **Predict prompt:** which deck is brighter? which drizzles?
- **Expected:** clean N≈20 → r_eff ≈ 30+ µm, albedo ≈ 0.33; polluted N≈400 → r_eff ≈
  8 µm, albedo ≈ 0.66.
- **Assess:** "pollution brightens clouds" stated correctly, and why it's a climate
  *uncertainty* (cooling that partly masks greenhouse warming).
- **Pitfall:** short-run surface-rain numbers are unreliable; teach on r_eff/albedo
  (robust) and treat drizzle qualitatively here.

### Exp 5 — Entrainment mixing / IHMD (M7)
- **LO:** Contrast homogeneous vs inhomogeneous mixing and their different effect on
  number vs size.
- **Misconception:** **M7** — "entrainment just dilutes uniformly."
- **Predict prompt:** IHMD 0→1: number? size? albedo? drizzle?
- **Expected:** IHMD=1 → fewer drops (~9% fewer ΣA), larger r_eff (~+1 µm), lower
  albedo (~−0.02), much more drizzle (≈30–40×).
- **Assess:** student articulates that inhomogeneous mixing *removes whole droplets*
  while survivors keep their size — qualitatively different from uniform shrinkage.
- **Pitfall:** this is genuinely unsettled science; reward reasoning over a "right
  number."

### Exp 6 — Completion challenge (skills, not new physics)
- **LO:** *Invert* the Twomey relationship — choose an input to hit an output target.
- **Activity:** hit albedo 0.55 ± 0.02 using only background N.
- **Expected:** N ≈ 90–130/cm³ region (N=150 gives ~0.59, "too bright").
- **Assess:** does the student use the *mechanism* ("too bright → fewer/larger →
  lower N") rather than blind trial-and-error?

---

## 4. Self-check answer key (the ✅ boxes)

| Where | Question | Answer |
|---|---|---|
| Exp 2 | Double the sea-salt — does LWC change? Albedo? | LWC roughly unchanged; albedo still rises (same water, even more/smaller drops). |
| Exp 3 | Run the same seeding on a polluted (N≈400) deck — still ~4× rain? | **No** — far from threshold, can't be tipped. |
| Exp 4 | Factory doubles aerosol: name two climate effects & their sign | Brighter (Twomey) → cooling; rains less / lasts longer → cooling. Both same sign; the magnitude is the big uncertainty. |
| Exp 5 | Two parcels lose equal water, homo vs inhomo — which is brighter? | Homogeneous (keeps droplet *number*, which sets albedo). |

---

## 5. Write your own questions — reusable templates

The chapter ships a fixed set; here is how to author more that stay aligned to the
frameworks. Copy a template, fill the brackets.

### 5a. POE question template (drop into a notebook markdown cell or app text)
```
> ### 🤔 Predict
> [One concrete, committal question — ask for a DIRECTION or a NUMBER, not "what
>  happens". e.g. "Raise X. Does Y go up, down, or stay flat?"]
>
> ⚠️ Common wrong intuition ([M#] or your own): "[state the tempting wrong belief]".
>
> --- run the cell ---
>
> ### 📖 Explain — compare with your prediction
> [State the result, then the mechanism in one sentence, then explicitly tie back:
>  "If you predicted …, update because …".]
>
> <details><summary>✅ Check yourself</summary>
> Q: [a transfer question in a new context]
> A: [the answer + the one idea it tests]
> </details>
```

### 5b. Marking rubric template (4-point, mechanism-focused)
| Score | Criterion |
|---|---|
| 4 | Correct prediction *and* a mechanism-level explanation (names the causal chain) |
| 3 | Correct outcome, explanation partial or misses one link |
| 2 | Outcome right by intuition, explanation absent/wrong |
| 1 | Outcome wrong but reasoning shows a named, correctable misconception |
| 0 | No engagement / restates the question |

> Reward *named-misconception repair* (score ≥1 even when wrong) — conceptual change
> requires students to surface the wrong idea, not hide it.

> ⚠️ **Use formatively, not summatively (yet).** This rubric and the items in §6–§7
> are **unvalidated** teaching aids. The notebook also places each answer one click
> from its question, so a score mostly measures reading compliance, not understanding.
> Use these for in-class discussion and low-stakes feedback; do **not** base a
> meaningful grade or a research pre/post measurement on them until the items are
> validated (see `docs/PEDAGOGY_DESIGN.md` §1, §A).

### 5c. Worked example — authoring a brand-new question
*Goal: a question on updraft strength (not in the shipped chapter).*
> 🤔 Predict: In the app, leave aerosol fixed and imagine raising the updraft `w`.
> Will peak supersaturation go up or down? Will droplet *number* go up or down?
> ⚠️ Wrong intuition: "stronger updraft just makes a bigger cloud." (It also changes
> *how many* droplets activate.)
> 📖 Explain: faster cooling → higher peak S → more aerosol cross their critical size
> → more droplets activate. Tie back: if you predicted "no change in number," that's
> the activation link to revisit.
> ✅ Check: two parcels, same aerosol, different `w` — which has the higher CDNC?
> (The stronger updraft.)

---

## 6. Extensible question bank (homework / exam prompts)

Use, adapt, or extend. Bloom level in brackets. **Blank slots at the end — add your
own using §5.**

1. *(Understand)* Explain in 3 sentences why splitting cloud water among more droplets
   raises albedo without adding water.
2. *(Apply)* A deck has r_eff = 14 µm and albedo 0.45. You want albedo ≥ 0.55. In the
   app, find a background N that achieves it and report it. Was your first guess too
   high or too low — and what does that tell you about the N–albedo relationship?
3. *(Analyse)* Exp 3 reports rain ≈ 70× the seed mass. Explain why a ratio near 1
   would have *disproven* "real" seeding.
4. *(Analyse)* Why is MCB's cooling reported as a domain mean and not per column?
   What real-world observation does this mirror?
5. *(Evaluate)* A start-up proposes seeding heavily-polluted coastal stratus to make
   it rain. Using Exp 3's caveats, argue whether this is likely to work.
6. *(Evaluate)* Aerosol pollution cools low clouds (Exp 4). Does this mean pollution
   is "good for climate"? Write a 1-paragraph rebuttal.
7. *(Create)* Design an app experiment that would distinguish the Twomey (brightness)
   effect from the cloud-lifetime (rain-suppression) effect. What would you vary and
   measure?
8. *( ___ )* __________________________________________________
9. *( ___ )* __________________________________________________

---

## 7. Starter concept inventory (pre/post)

A short, extendable item bank to seed discussion (a starting point toward the
inventory in `PEDAGOGY_DESIGN.md §1`). Correct answer in **bold**; targeted
misconception noted.

> ⚠️ **Not a validated instrument.** These are instructor-experience items, **not**
> a validated concept inventory. Use them as formative discussion prompts. Do **not**
> treat them as a research-grade pre/post test or a basis for grades until validated
> against the atmospheric-science education literature. Note especially that **M5**
> (spectral broadening) and **M7** (homogeneous vs inhomogeneous mixing) are better
> framed as *expert-level distinctions / open research questions* than as documented
> novice misconceptions — for M7, that inhomogeneous mixing exists is real, but which
> regime dominates in nature is genuinely unsettled.

1. **(M2)** Adding more aerosol to a cloud (fixed water) makes the droplets:
   a) bigger  **b) smaller and more numerous**  c) unchanged  d) fewer and bigger
2. **(M1)** Peak supersaturation inside a forming cloud is typically:
   a) 50–150%  **b) a fraction of a percent (~0.1–1%)**  c) exactly 100%  d) negative
3. **(M3)** Warm rain forms mainly by:
   a) more condensation  **b) collision–coalescence of droplets**  c) freezing
   d) evaporation
4. **(M6)** A small number of giant CCN in a marginal deck:
   **a) can trigger rain that wouldn't otherwise form**  b) are always negligible
   c) only matter in ice clouds  d) reduce rain
5. **(M2/Twomey)** Compared with clean marine air, polluted air over the ocean makes
   low clouds: **a) brighter and less rainy**  b) darker and rainier  c) no different
6. **(M7)** Inhomogeneous entrainment mixing primarily:
   a) shrinks every droplet equally  **b) removes whole droplets, survivors keep size**
   c) adds droplets  d) has no effect on number
7. **(M5)** During collision growth, the droplet size distribution:
   a) stays uniform  **b) broadens (a few drops run away in size)**  c) narrows to one
8. *(Synthesis)* MCB and rain-seeding are best described as:
   **a) the same aerosol lever pulled in opposite directions**  b) unrelated
   c) both warming  d) both about ice

---

## 8. Deploying for a class

- **App, local:** `streamlit run app/streamlit_climate.py` (needs the droplab env).
- **App, hosted:** Streamlit Community Cloud or an HF Space (free CPU tier handles one
  run at a time). Numba runs server-side — do **not** try to host this on a
  client-only WASM runtime; the model won't execute there.
- **Notebook, lab:** distribute the repo; have students `Run All` once to warm numba,
  or pre-run and ship with outputs for a reading-only assignment.
- **Notebook, zero-install:** the notebook has an *Open in Colab* badge + a Colab
  bootstrap cell — students with only a Google account run it with no setup. Each
  Colab session is its own backend, which **also sidesteps the shared-kernel meltdown**
  of a 30-student JupyterHub: Colab (or per-student Binder/Codespaces) gives each
  learner an isolated runtime. Recommended for labs.
- **Budget the MCB cell:** it's ~100 s. In a live demo, start it, then talk through
  the prediction while it runs. (Don't try to shrink it with a smaller `nt` — the
  brightening signal needs the full run to develop, so a fast version shows nothing.)

## 9. Extending the model (for ambitious lecturers)

The notebook/app are *pure consumers* of the droplab API; you can author new experiments
without changing the physics. Useful entry points (read-only):
`droplab.climate_widget.simulate(...)`, `droplab.climate_diag.{column_optics,
twomey_report, toa_forcing}`, `droplab.flow2d_dynamic.run_flow2d_dynamic(...)`, and the
`examples/*.py` demos. Keep the honest caveats (domain-mean MCB; threshold-only
seeding; long spin-up for surface rain) whenever you add a new lesson.
