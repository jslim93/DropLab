# Climate Intervention — a student guide

Can we deliberately change a cloud — to cool the planet, or to make it rain?

This chapter uses **droplab**, an educational 2-D Lagrangian model of marine
**stratocumulus** (the low, flat cloud sheets that shade cool oceans), to explore
**aerosol–cloud–radiation coupling** and the two best-known cloud
geoengineering ideas: **marine cloud brightening** and **rain seeding**.

> **The one idea to take away:** a cloud is *not a fixed object* — it is an
> **aerosol-mediated state**. The same air can host a bright, long-lived,
> non-raining cloud *or* a dim, drizzly, short-lived one, depending on how many
> aerosol particles seed its droplets. Brightening and rain-seeding are the same
> lever pulled in **opposite directions**.

## Ways to explore

| | What it is | How to launch | Best for |
|---|-----------|---------------|----------|
| 📓 **Notebook** | A guided 5-experiment story with Predict → Observe → Explain prompts, figures, and one animation | `jupyter lab notebooks/Climate_Intervention.ipynb` | Learning the *why*, start to finish |
| 🎛️ **Guided app** | The marine-Sc model behind four sliders and a Run button — no code | `streamlit run app/streamlit_climate.py` | Live demos, focused "what if?" on the climate story |
| 🧪 **Sandbox** | Free play across *all* cloud types (Sc, cumulus, fog, sheared bubble) + intervention + diel cycle + wind shear + multimode aerosol + inversion/bubble knobs | `streamlit run app/streamlit_sandbox.py` | Open exploration beyond the guided chapter |
| 👩‍🏫 **Instructor handbook** | Session plans, per-experiment answer keys, a question-authoring template, an extensible question bank, and a starter concept inventory | `docs/CLIMATE_INTERVENTION_INSTRUCTOR.md` | Teaching it / writing your own assessment |

Both are **pure consumers** of the same tested model
(`droplab.climate_widget.simulate` and friends). They add **no new physics** — they
only re-package the existing climate-intervention code so students can use it.

## The five experiments (notebook)

1. **The marine stratocumulus deck.** A realistic DYCOMS-II deck with a sharp
   inversion and cloud-top radiative cooling — the canvas for everything else.
2. **Marine Cloud Brightening (MCB).** Spray *tiny* sea-salt → more, smaller
   droplets → the cloud reflects more sunlight (the **Twomey effect**) → a
   top-of-atmosphere **cooling of a few W/m²** (seeded vs unseeded, domain-mean).
3. **Precipitation seeding.** Inject a few *giant* CCN (~1.5 µm) into a deck sitting
   right at its precipitation threshold → rain is triggered, **several-fold more
   rain** and many times the injected seed mass.

> **Read the numbers as model outputs, not forecasts.** The specific values this
> chapter reports (e.g. a TOA cooling near −5 W/m², a rain/seed-mass ratio near 70×)
> are genuine outputs of *this* idealized 2-D deck with a simple slab-albedo radiation
> estimate — they depend on the turbulence seed and resolution and are **not**
> real-world forcings. What is robust and transferable is the **sign and mechanism**
> (more CCN → smaller drops → brighter → cooling; giant CCN → collision–coalescence →
> rain), not the exact magnitude.
4. **Clean vs polluted decks.** A clean deck (N≈20) makes large, dim, drizzly
   droplets; a polluted deck (N≈400) makes small, bright, rain-suppressed ones —
   the natural, always-on Twomey effect.
5. **Entrainment mixing (IHMD).** *How* dry air mixes into the cloud (homogeneous
   vs inhomogeneous) changes the droplet spectrum: inhomogeneous mixing leaves
   **fewer, larger** drops → **dimmer**, **drizzlier**.

## Two honest caveats (please remember these)

These are idealised experiments for *intuition*, not operational forecasts:

1. **"Rain" must be the cloud's water, not the seed mass.** Injecting large drops
   directly produces "rain" that is just the heavy seeds falling back out. Real
   seeding uses small *dry* giant CCN that must grow *inside* the cloud — which is
   why the demo checks the rain-to-seed-mass ratio (~70×).
2. **You can only tip a deck near its threshold.** A deck far from raining cannot
   be pushed into rain by a few nuclei. Likewise, MCB's cooling signal is robust
   only in the **domain mean** — per-column it is buried in turbulent noise (this is
   why real ship-tracks appear in *time-mean* satellite composites, not snapshots).

## A note on runtime

The simulations are real physics, so they take real time:

- Most notebook cells run in **15–50 s**.
- **Experiment 2 (brightening) is the slow one, ~2 minutes** — the cooling signal
  only emerges once the injected aerosol is lofted into the cloud and the deck
  re-equilibrates.
- **Surface rain needs a long spin-up.** Brightness reacts almost instantly, but
  rain must *grow and fall*, so short runs cannot show it. Trust surface-rain
  numbers only in the long runs (Experiments 2–3); for quick runs, read **droplet
  size** and **albedo**, which are robust.

## Accessibility & access

- **Zero-install (Colab).** The notebook carries an *Open in Colab* badge and a
  one-cell Colab bootstrap, so a student with only a Google account can run the whole
  chapter without installing anything. (Note: the model needs a Python backend —
  numba does not run in a browser-only WASM runtime, so a Colab/Binder/local kernel is
  required; there is no fully client-side version.)
- **Colour-vision safety.** Field panels use perceptually-ordered colourmaps and the
  droplet-size scale uses **viridis** (perceptually uniform and colour-vision-deficiency
  safe). Seeded droplets carry a **redundant cue** — a magenta ring *and* size — so
  they remain distinguishable without relying on colour alone.
- **Reading-only path.** Instructors can pre-run the notebook and share it with
  outputs for students who cannot run a kernel (this trades away interactivity).

## Under the hood (for the curious)

The notebook and app call these existing modules (do not need editing):

- `droplab/climate_widget.py` — `simulate()`, `figure()`, `climate_panel()`
- `droplab/climate_diag.py` — `column_optics`, `twomey_report`, `toa_forcing`
- `droplab/flow2d_viz.py` — `draw_frame_seeded`, `animate_mcb`, `animate_seeding_compare`
- `droplab/flow2d_dynamic.py` — `run_flow2d_dynamic` (the core 2-D solver)
- `droplab/soundings.py` — `DYCOMS`, `DYCOMS_RADIATION`
- `examples/{mcb_demo,precip_seeding,precip_vs_nonprecip,entrainment_mixing}.py`
  — each has `run()` / `figure()` and can be run standalone, e.g.
  `python -m examples.mcb_demo`.

## Pedagogical basis (for citation)

This chapter is designed around established learning-science frameworks (see also
`docs/PEDAGOGY_DESIGN.md`). The notebook's Predict → Observe → Explain → Refine
structure, the named-misconception flags (⚠️ M2/M3/M5/M6/M7), the one-knob
"completion" challenge, and the revealable self-checks map directly onto these.

> **Note:** bibliographic details below are recalled from the literature and should
> be cross-checked (DOIs, exact page numbers, edition) against a managed
> bibliography before use in a manuscript.

- **ICAP framework** (engagement: Interactive > Constructive > Active > Passive):
  Chi, M. T. H., & Wylie, R. (2014). The ICAP framework: Linking cognitive
  engagement to active learning outcomes. *Educational Psychologist*, 49(4),
  219–243. — earlier: Chi, M. T. H. (2009). Active–constructive–interactive: A
  conceptual framework for differentiating learning activities. *Topics in Cognitive
  Science*, 1(1), 73–105.
- **Predict–Observe–Explain (POE):** White, R. T., & Gunstone, R. F. (1992).
  *Probing Understanding*. London: Falmer Press.
- **Constructive alignment:** Biggs, J. (1996). Enhancing teaching through
  constructive alignment. *Higher Education*, 32(3), 347–364. — textbook treatment:
  Biggs, J., & Tang, C. (2011). *Teaching for Quality Learning at University*
  (4th ed.). Open University Press.
- **Cognitive load theory & the completion effect:** Sweller, J. (1988). Cognitive
  load during problem solving: Effects on learning. *Cognitive Science*, 12(2),
  257–285. — Sweller, J., van Merriënboer, J. J. G., & Paas, F. (1998). Cognitive
  architecture and instructional design. *Educational Psychology Review*, 10(3),
  251–296. — completion problems specifically: van Merriënboer, J. J. G. (1990).
  Strategies for programming instruction in high school: Program completion vs.
  program generation. *Journal of Educational Computing Research*, 6(3), 265–285.
- **Conceptual change / misconceptions:** Posner, G. J., Strike, K. A., Hewson,
  P. W., & Gertzog, W. A. (1982). Accommodation of a scientific conception: Toward a
  theory of conceptual change. *Science Education*, 66(2), 211–227. — diSessa, A. A.
  (1993). Toward an epistemology of physics. *Cognition and Instruction*, 10(2–3),
  105–225.
- **Multiple representations (DeFT):** Ainsworth, S. (2006). DeFT: A conceptual
  framework for considering learning with multiple representations. *Learning and
  Instruction*, 16(3), 183–198. — earlier: Ainsworth, S. (1999). The functions of
  multiple representations. *Computers & Education*, 33(2–3), 131–152.
- **Interactive simulations in science (PhET):** Wieman, C. E., Adams, W. K., &
  Perkins, K. K. (2008). PhET: Simulations that enhance learning. *Science*,
  322(5902), 682–683. — Perkins, K., et al. (2006). PhET: Interactive simulations
  for teaching and learning physics. *The Physics Teacher*, 44(1), 18–23.
