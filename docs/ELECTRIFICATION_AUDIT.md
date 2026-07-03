# Electrification Realism Audit

> **ADDENDUM (2026-06-22, after the audit): the charging was rebuilt to be physically grounded.**
> Per 박사's standard ("simplification is fine for a teaching model, but the physical basis must be
> correct"), the single biggest fake — the tuned `charge_coeff` — was removed. Charging is now an
> actual **rebounding graupel–crystal collision rate (gravitational kernel) × a laboratory
> per-collision charge δq ≈ 5 fC** (Takahashi 1978 / Saunders & Peck 1998), **local and
> collisional**, with graupel vs crystal a **persistent physical size** (D > 0.2 mm), not a
> mass-rank. Result: the charge density is now order-of-magnitude physical (~0.1–1 nC/m³) and
> charging only fires once the storm grows real graupel (vigorous deep convection) — honest.
> What 2-D still **cannot** make physical: the **field magnitude** (~100× weak in a 2-D grounded
> box), so `E_breakdown` is now an explicitly **illustrative** trigger, labelled as such. Items
> §2 (field BCs), §3 (static-field DBM, no LPCC) below remain open simplifications. The §1 charging
> critique is largely **addressed**; the rest of this document stands as the record of what was
> fixed and what is still a labelled toy.

---

# Electrification Realism Audit (original, pre-fix)

**Date:** 2026-06-21. **Trigger:** 박사 asked, correctly, why a feature that most LES/CRMs lack
was "so easy" to implement — and whether the visible lightning is physics or a glyph.
**Method:** an independent adversarial audit (a separate agent, storm-electrification skeptic
stance) cross-checked against the author's own assessment. Tags: [FACT] [INFERRED] [TOY]
[UNVERIFIED].

## One-line verdict
[FACT] **Not realistic in any predictive sense.** It is a self-consistent, charge-conserving,
nicely-animated **cartoon** whose every *quantitative* result — charge magnitude, field strength,
whether anything flashes, intracloud-vs-ground outcome — is set by **tuned knobs and numerical
artifacts**, not by storm physics. It is physically *illustrative*, not physically *predictive*.
The structure (charge separation → Poisson field → DBM discharge) is built from legitimate
pieces; the quantitative content and validation are absent.

## Why it was "easy" (the honest answer to the suspicion)
[FACT] Real storm electrification is hard for five reasons; this toy sidesteps every one:

| Hard part of real electrification | What this toy did instead |
|---|---|
| Explicit graupel + ice-crystal categories with **rebounding collision rates** | one ice category, **mass-median rank** proxy |
| Per-collision charge **δq(T, LWC, RAR, size, velocity)** from contested lab data (Takahashi 1978; Saunders & Peck 1998) | `W = charge_coeff·Σqsc·dt`, **charge_coeff a free knob tuned until flashes appear** |
| Self-consistent leader: re-solve Poisson with the channel as an internal equipotential (Mansell 2002) | DBM tree on a **static, frozen** φ field |
| Validation vs LMA / balloon soundings / field mills | **self-consistency identities** (charge conservation) + a plausible animation |
| (and: no dynamical feedback, so it's optional → most LES omit it) | diagnostic-only — which is *why* it slotted in trivially (golden gate "trivially safe" = it does nothing to the dynamics) |

[FACT] **"Easy + plausible picture" ≠ correct.** A real DBM tree on a real Poisson field *looks*
like lightning regardless of whether the charge that built the field is physical. Plausibility of
the image is uncorrelated with correctness of the physics. The suspicion was right.

## Component grades

### 1. Charging — mostly hollowed out
- [FACT real] Dipole narrative (NI graupel–crystal rebound, gravitational separation, reversal-
  temperature polarity); charge conserved exactly; the dipole forming "for free" from existing
  Lagrangian sedimentation/lofting is a genuinely elegant use of the super-droplet method.
- [TOY] `charge_coeff` is a **free knob**, not a physical quantity (the magnitude has *no* anchor
  in lab electrification). **No per-collision charge, no collision rate, no RAR, no size/velocity
  dependence.** The "rate" is `qsc·dt`, not (rebounding collisions)·δq.
- [TOY] One ice category + **instantaneous global mass-median** split → "graupel/crystal" identity
  flips as the median moves; the split is **non-local and non-collisional** (opposite charge can
  land on SDs in different cells that never collided). It is charge **bookkeeping shaped like** NI
  charging, not NI charging.
- [TOY] Single reversal temperature, **no LWC dependence → no tripole / lower positive charge
  center (LPCC)** — the charge feature most associated with initiating cloud-to-ground flashes.
- [TOY] `graupel_charge_sign` is sign-only; real charging magnitude varies strongly and
  non-monotonically with (T, LWC).

### 2. Field — right PDE, wrong boundaries, meaningless absolute scale
- [FACT real] `∇²φ = −ρ_q/ε₀`, `E = −∇φ`, ε₀, and the density-scaled breakdown threshold
  `E_crit(z)=E_breakdown·ρ/ρ₀` are correct.
- [TOY] **φ=0 Dirichlet on ALL walls (grounded box), including the top lid and sides.** A real
  solver grounds only z=0 (Earth) with open/decaying top and sides + a fair-weather field. The
  grounded cavity shorts the upper dipole to the lid and clamps the lateral field; `max|E|` (the
  flash trigger) becomes **domain-size dependent** — an artifact. (Periodic-x mode at least frees
  the sides and should be preferred.)
- [TOY] **2-D.** A "point" charge is an infinite line charge in y → potential ~ln(r), field ~1/r
  (not 1/r²). The **absolute field scale is physically meaningless**; only relative behavior is
  interpretable. The out-of-plane `depth` (default 1 m) linearly scales ρ_q→φ→E: another hidden
  knob feeding `max|E|`.

### 3. Discharge (DBM) — real tree skeleton, missing the leader physics
- [FACT real] The growth rule `P_i ∝ |φ_i−φ₀|^η / Σ|φ_j−φ₀|^η` **is** the Niemeyer–Pietronero–
  Wiesmann (1984) DBM, correctly implemented (branching, η, 8-connectivity). Channel-mean
  neutralisation conserves charge (Kasemir 1960 net-neutral leader spirit). Not a cosmetic line.
- [TOY] **The channel is NOT re-solved as an equipotential.** φ₀ is frozen at the initiation
  cell; the φ field is never updated as the channel grows. So it **traces a static field** rather
  than dynamically screening it (the defining behaviour of a real leader; Mansell 2002). No
  internal/critical channel field, no bidirectional equipotential propagation.
- [TOY] Initiation = deterministic `argmax|E|`. "Reached ground" = the tree's z-index hits row 0
  — a **geometric** event, no leader–ground attachment / return stroke.
- [FACT/INFERRED] **Intracloud dominance is an ARTIFACT, not a physical IC:CG ratio.** CG flashes
  are suppressed by the combination of `max_cells=180` (cap), **η=3** (greedy → grows back into the
  high-field mid-level gap, statistically repelled from the weak sub-cloud descent), the grounded
  floor, and the **absent LPCC**. Real IC:CG (~2:1–10:1) emerges from charge structure + leader
  physics; here it is hard-wired by numerics.

### 4. Validation — none against observations
- [FACT real] Charge conservation (`Σcharge + charge_to_ground = 0`) and the golden gate are real
  *software* invariants and pass.
- [FACT] **Charge conservation proves NOTHING about realism** — it is an arithmetic identity any
  conserving scheme satisfies. `test_dipole_forms` checks only SIGN (cannot detect the missing
  tripole); `test_field_and_flash_fire` is a smoke test that the code runs. **No observational
  validation exists.** Realism currently rests on the picture.

## Bugs / doc-code mismatches found (fixed in this pass)
1. [FACT] **η doc/code mismatch.** `dbm_leader` defaulted `eta=1.0` with a docstring claiming
   "η=1 → D≈1.7", but `flash()` passes **`eta=3.0`** (what actually runs), and the design doc said
   η=1. → aligned everything to the running value and removed the wrong D≈1.7 claim.
2. [FACT] **"faithful leader" over-claim** → docstring/design now state the channel traces a
   **static** field and is NOT re-solved as an equipotential (a deliberate simplification vs
   Mansell 2002).
3. [FACT] **"two invariants make this testable" over-claim** → reworded: charge conservation is
   *self-consistency*, not evidence of realism.

## The single biggest unphysical assumption
[FACT] That `W = charge_coeff·Σqsc·dt`, split by an instantaneous global mass-median, stands in for
non-inductive collisional charging. This one substitution removes the collision rate, the
per-collision charge, the graupel/crystal identity, the RAR/LWC/T dependence, **and** locality at
once. Runner-up: the grounded box + 2-D, which make `max|E|` — hence *whether anything flashes* —
an artifact of geometry and the free `depth`.

## Top 5 changes to make it *defensible* (not validated — defensible)
1. Replace `charge_coeff` with a **literature charging law**: rebounding graupel–crystal collision
   rate × δq(T, LWC/RAR) from a Saunders & Peck (1998) or Takahashi (1978) lookup.
2. **Real graupel vs crystal**: persistent per-SD class tags (e.g. a rimed-mass/density threshold),
   charging between two co-located populations, local (not global-zone) split.
3. **Fix BCs + dimensionality honesty**: conducting ground only at z=0, open/Neumann top & sides;
   report fields per-unit-depth; state that absolute field/flash thresholds are not physical in a
   2-D grounded box.
4. **True equipotential leader**: re-solve Poisson with the channel as an internal φ=const boundary
   (Mansell 2002), density-scaled initiation, real ground attachment → IC:CG from physics.
5. **LWC-dependent reversal** (→ tripole/LPCC) + **validate one observed proxy** (flash-rate vs
   graupel/updraft, Deierling & Petersen 2008; or a balloon charge profile) **with a knob
   sensitivity sweep** proving the headline results aren't knob-determined.

## References
Takahashi (1978) JAS 35:1536; Reynolds, Brook & Gourley (1957) J.Meteor. 14:426; Saunders & Peck
(1998) JGR 103:13949; Saunders, Keith & Mitzeva (1991) JGR 96:11007; Kasemir (1960) JGR 65:1873;
Niemeyer, Pietronero & Wiesmann (1984) PRL 52:1033; Mansell, MacGorman, Ziegler & Straka (2002)
JGR 107:4075; Mansell et al. (2005) JGR 110:D12101; Stolzenburg, Rust & Marshall (1998) JGR
103:14097; Deierling & Petersen (2008) JGR 113:D16210; Boccippio et al. (2001) J.Climate (IC:CG).
