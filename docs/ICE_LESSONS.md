# Mixed-phase education lessons (designed with the Education Framework)

Four lessons that pair the new mixed-phase science (manuscript §3.5) with hands-on
investigation, each built on the **Lesson Pattern** (`docs/EDU_FRAMEWORK.md §1`) and the
`droplab.edu` scaffold. They target the mixed-phase misconceptions **Mi1–Mi4**
(`docs/EDU_FRAMEWORK.md §5`). This is the design; it becomes a
`notebooks/Mixed_Phase.ipynb` (and Streamlit lessons) implemented verbatim from the
six steps below.

**Model knobs used** (all already in `run_flow2d_dynamic`): `ice=True`,
`freezing_mode` (`"abifm"`/`"bigg"`), `inp_n_cm3`, `inp_r_um`, `inp_sigma`,
`inp_species`, and glaciogenic seeding via the `seeding` spec with `phase="ice"`.
**Frame outputs to read:** `q_liquid`, `q_ice`, `phase` (per-frame when `ice=True`).
**Reference demos:** `paper/figs_mixed_phase.py` → bergeron / arctic_mpc /
glaciogenic_seeding.

> Honesty (same as everywhere): all values are idealized-model outputs — teach the
> **sign and mechanism**, not the digits. Mi1–Mi4 are instructor-experience candidates
> (formative use; not validated).

---

## Lesson I1 — Supercooled liquid (Mi1)

- **① Frame.** *Outcome:* recognize that a cloud below 0 °C is usually **not** all ice.
  *Misconception Mi1:* "below 0 °C, clouds are ice."
- **② Predict.** Cool a clean cloud (no INP) to −15 °C. What phase is it — all ice, all
  liquid, or a mix? Will any ice appear?
- **③ Observe.** Run `ice=True, inp_n_cm3=0` on a cold (~−15 °C) sounding. Watch
  `q_liquid` vs `q_ice` and the `phase` field.
- **④ Explain.** It stays **supercooled liquid** — with no INP and T warmer than ≈−38 °C,
  there is nothing to freeze the drops. If you predicted "all ice," that is Mi1:
  liquid persists well below 0 °C; mixed-phase needs an ice *trigger*.
- **⑤ Refine.** Lower the sounding temperature step by step (one knob) — find the T at
  which a clean cloud *first* glaciates on its own (≈ −38 °C, homogeneous freezing).
- **⑥ Check** (inventory item 9): *A cloud at −15 °C is most likely…?* → mixed-phase /
  supercooled liquid + some ice.

## Lesson I2 — What makes a drop freeze (Mi4)

- **① Frame.** *Outcome:* distinguish heterogeneous (INP) from homogeneous freezing.
  *Misconception Mi4:* "drops freeze at 0 °C."
- **② Predict.** Starting from the supercooled cloud of I1, add a few INP
  (`inp_n_cm3`). At what temperature do ice crystals start to appear — near 0 °C, or
  colder?
- **③ Observe.** Sweep `inp_n_cm3` from 0 upward (`freezing_mode="abifm"`); watch when
  `q_ice` first rises and at what cloud temperature.
- **④ Explain.** Freezing begins **well below 0 °C** and needs an **INP** (ABIFM/Bigg);
  without INP the drops supercool to ≈−38 °C. If you predicted 0 °C, that is Mi4 —
  pure water does not freeze at its melting point.
- **⑤ Refine.** With T fixed, raise `inp_n_cm3` until ~half the condensate is ice — read
  off the INP concentration that "half-glaciates" this cloud.
- **⑥ Check** (item 10): *A pure droplet cooled below 0 °C freezes at about…?* → ≈−38 °C
  (homogeneous) unless an INP triggers it sooner.

## Lesson I3 — Wegener–Bergeron–Findeisen (Mi2)

- **① Frame.** *Outcome:* explain why ice and liquid do **not** stably coexist.
  *Misconception Mi2:* "ice and liquid coexist in equilibrium."
- **② Predict.** In a mixed-phase cloud (some ice, lots of supercooled drops), over the
  next minutes, what happens to the ice mass and to the liquid mass?
- **③ Observe.** Run the bergeron case (`paper/figs_mixed_phase.py`); track `q_ice` and
  `q_liquid` time series in the same column.
- **④ Explain.** **WBF**: because saturation vapour pressure over ice is lower than over
  liquid, the air is supersaturated w.r.t. ice but sub-saturated w.r.t. liquid — so
  **ice grows while the droplets evaporate**. Mixed-phase is *transient* unless updraft
  keeps replenishing liquid. If you predicted "they stay in balance," that is Mi2.
- **⑤ Refine.** Add a steady updraft (one knob) — find the updraft strength at which
  liquid is replenished fast enough to *survive* the WBF sink (persistent mixed-phase).
- **⑥ Check** (item 11): *Ice + supercooled drops over time…?* → ice grows, droplets
  evaporate (WBF).

## Lesson I4 — INP glaciation & glaciogenic seeding (Mi3)

- **① Frame.** *Outcome:* see that more INP does not simply mean more snow.
  *Misconception Mi3:* "more INP → more ice → more snow, always."
- **② Predict.** Sweep INP from clean to heavily seeded. Does surface snow increase
  monotonically? What happens to the cloud's liquid?
- **③ Observe.** Glaciogenic-seeding run (`seeding` with `phase="ice"`, or an
  `inp_n_cm3` sweep); watch the liquid→ice conversion, the snow at the surface, and the
  cloud's lifetime (the glaciogenic_seeding / arctic_mpc demos).
- **④ Explain.** A little INP turns supercooled liquid into precipitating ice
  (snow ↑ — the mixed-phase analogue of rain seeding). But **too much** INP glaciates
  the whole cloud, depletes the liquid, and can *shorten the cloud's life and reduce*
  precip — the response is **non-monotonic**. If you predicted "always more snow,"
  that is Mi3.
- **⑤ Refine.** Find the INP "sweet spot": the `inp_n_cm3` that **maximizes** surface
  snow before over-glaciation cuts it back.
- **⑥ Check** (item 12): *Greatly increasing INP…?* → glaciates and can deplete the
  liquid; precip response is non-monotonic.

---

## Curriculum placement & manuscript hook
These four lessons are the **mixed-phase rung** of the curriculum spine
(`docs/EDU_FRAMEWORK.md §4`): they reuse the same Predict→Observe→Explain→Refine→Check
ritual as the warm-phase and climate chapters, so the new ice science arrives as a
*designed* extension of one learning environment — supporting the manuscript's
educational-design claim for §3.5 + §7 (design claim, not learning-gain claim).
