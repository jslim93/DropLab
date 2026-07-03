# DropLab Education Framework

**Status:** adopted framework (this is the *standard* every lesson follows).
**Relationship:** `docs/PEDAGOGY_DESIGN.md` is the gap analysis (what's missing and
why, with theory citations); **this file is the concrete framework that closes those
gaps and makes "educational" a design property rather than a label.**

The problem the pedagogy review found: the educational artifacts are individually
fine but *inconsistent* — `PyLCM_Part2_Experiments.ipynb` is full Predict-Observe-
Explain, the climate chapter reaches ICAP-Constructive, but `PyLCM_edu.ipynb` and the
parcel Streamlit app sit at ICAP-Active ("twiddle a slider, watch a plot"). A paper
that claims an *educational design* needs one shared scaffold, not a pile of demos.

---

## 1. The Lesson Pattern (the atomic unit)

Every lesson — notebook section, app experiment, or instructor activity — follows the
same six-step pattern. The pattern is what lifts an activity from *Active* (manipulate
+ watch) to *Constructive/Interactive* (generate + reconcile), per ICAP (Chi & Wylie)
and POE (White & Gunstone).

| Step | What it is | Why (theory) |
|---|---|---|
| **① Frame** | State the **learning outcome** and name the **targeted misconception** (M#). | Constructive alignment (Biggs); conceptual change requires naming the wrong idea first (Posner). |
| **② Predict** | Elicit a **committed, specific** prediction *before* running (a direction or a number, not "what happens"). | POE; ICAP-Constructive needs generation before observation. |
| **③ Observe** | Run; read the result across **≥2 linked representations** (field + droplets + diagnostics). | Multiple representations (Ainsworth DeFT). |
| **④ Explain** | Reconcile the result **against the student's own prediction** ("if you predicted X, update because…"). | POE's reconciliation; this is where conceptual change happens. |
| **⑤ Refine** | A **completion task**: change *one* control to hit a **target**, inverting the mechanism. | Scaffolding fade / completion effect (Sweller; van Merriënboer) — the "missing middle." |
| **⑥ Check** | A revealable **self-check** (formative), mapped to the lesson's outcome + misconception. | Constructive alignment — an outcome you cannot check is not an outcome. |

**Conformance rule:** a lesson "follows the framework" iff it has all six steps, the
misconception is *named* (not just confronted), the Explain *references the prediction*,
and there is at least one Refine (completion) task. Anything missing ④–⑥ is ICAP-Active
and must be upgraded.

## 2. Reusable scaffold (`droplab/edu.py`)

To make the pattern uniform (and cheap to apply to new lessons, including ice), the
six steps are emitted by one small module. It renders consistent Markdown/HTML in
notebooks and returns plain strings the Streamlit apps can reuse. See `droplab/edu.py`:

```python
from droplab.edu import lesson, predict, explain, refine, self_check, misconception
lesson("Marine cloud brightening", outcome="explain the Twomey effect",
       misconception="M2")                      # ① Frame
predict("Will the droplets get bigger or smaller? Sign of the TOA forcing?")  # ②
# ③ run the model …
explain("more, smaller drops → brighter; if you predicted bigger, that's M2.")  # ④
refine("Using ONLY background N, hit albedo ≈ 0.55.")                            # ⑤
self_check("Double the salt — does cloud water change?", "Roughly no; albedo still rises.")  # ⑥
```

The module is a *pure presentation layer* — no physics. It guarantees the ⚠️ M# flags,
the ✅ self-check `<details>` boxes, and the 🎯 completion framing look the same
everywhere, so a learner meets one consistent ritual across notebook, app, and handbook.

## 3. Assessment framework (constructive alignment)

The biggest gap. A lesson's outcome must be *checkable*; the tool must be able to tell
whether a student reached it. Two instruments, both mapped to the misconception list:

- **Embedded formative checks** — the ⑥ Check in every lesson (low-stakes, self-graded).
- **Concept inventory** — a small pre/post item bank in `assessment/concept_inventory.md`,
  one or more items per misconception, with answer keys and a misconception↔outcome map.
  **Status: instructor-experience items, not yet validated.** Use formatively; do *not*
  grade or run a research pre/post until validated against the atmospheric-science
  education literature.

Alignment table (outcome ↔ activity ↔ assessment) lives in the curriculum spine (§4)
and is the backbone of the manuscript's educational-design claim.

## 4. Curriculum spine (a progression, not a feature list)

The tool's investigations are sequenced as a **learning progression**, each rung with a
learning outcome (LO) and the misconception it confronts. New modules (mixed-phase) are
*designed into* this spine, not bolted on.

| Module | Learning outcome | Misconception | Lessons / interface |
|---|---|---|---|
| Activation | aerosol + supersaturation set droplet number | M1 (big supersaturation), M4 (all activate) | Part 1 §2; parcel app L1 |
| Condensation vs collision | condensation alone can't rain; collision-coalescence does | M3, M5 (uniform growth) | Part 1 §3–4; Part 2 Exp 1–2 |
| Aerosol regime | more aerosol → more, smaller drops (Twomey) | M2 | Part 2 Exp 1; climate Exp 4 |
| 2-D dynamics | buoyancy-driven cloud; shear/bubble organize it | (structure intuitions) | sandbox (single-cloud mode) |
| Climate intervention | MCB brightens; giant-CCN seed rain; both honest-bounded | M2, M3, M6 | climate notebook + app |
| Entrainment mixing | homogeneous vs inhomogeneous change number vs size | M7 | Part 1 §5; climate Exp 5 |
| **Mixed-phase (new)** | supercooled liquid persists; WBF glaciates; INP nuance | **Mi1–Mi4 (see §5)** | ice lessons (this session, designed with this framework) |

## 5. Misconception register

Warm-phase **M1–M7** are in `docs/PEDAGOGY_DESIGN.md §A`. The framework adds the
mixed-phase set (instructor-experience candidates; validate before research use):

| # | Common wrong intuition | Correct idea |
|---|---|---|
| **Mi1** | Below 0 °C, clouds are ice | Supercooled **liquid** persists to ≈ −38 °C; mixed-phase is common |
| **Mi2** | Ice and liquid coexist in stable equilibrium | **WBF**: ice grows at the liquid's expense (e_s,ice < e_s,liq) → mixed-phase is transient unless replenished |
| **Mi3** | More INP → more ice → more snow, always | INP glaciate the cloud and can deplete the liquid; precip response is non-monotonic |
| **Mi4** | Drops freeze at 0 °C | Homogeneous freezing ≈ −38 °C; heterogeneous needs **INP** (Bigg/ABIFM); most drops supercool |

## 6. Conformance checklist (audit existing lessons)

Use to bring the whole suite to one standard:

- [ ] Frame names the LO **and** the misconception (M#/Mi#)?
- [ ] Predict elicits a committed, specific answer **before** the run?
- [ ] Observe shows ≥2 linked representations?
- [ ] Explain **references the student's prediction**?
- [ ] Refine has a one-control completion **target**?
- [ ] Check is a revealable self-check mapped to the outcome?

Current status (from the pedagogy review): Part 2 ✅ · climate chapter ✅ ·
Part 1 (partial, missing ②⑤⑥) · `PyLCM_edu.ipynb` & parcel app ❌ (Active) → upgrade
targets.

## 7. What the manuscript may claim (and the honest boundary)

**May claim:** DropLab implements a *uniform, theory-grounded lesson framework*
(POE + named misconceptions + completion scaffolding + formative checks) across one
tested model core exposed through notebooks, browser apps, and CLI — i.e. educational
design is a *property*, not a label.

**Must NOT claim** (no classroom data yet): that DropLab *improves learning*. The
concept inventory enables a future learning-gains study; until then this is a **design
claim, not a learning-gain claim** (consistent with `paper/pedagogy_positioning.md`).
