# Ice + radiation development roadmap (branch `feature/radiation-ice`)

Staged with the [목표][계획][가정][실행][증거][검증][리스크][다음] framework. A and B are
DONE and verified; C and D and the cirrus case remain. Golden (`ice=False`) stays
bit-identical at every stage.

## Stage A — long-wave + short-wave cloud radiative effect  ✅ DONE (commit d071808)
- `droplab/climate_diag.py`: `lw_emissivity` (grey Stephens-type), `cloud_radiative_effect`
  (per-column SW/LW/net CRE incl. ice path + cloud-top temperature), `radiative_report`
  (intervention dSW/dLW/dNet). `T_col` exposed in run output (golden-safe).
- Figure 7 `fig_radiative_signature` (SW/LW/net of MCB / precip / glaciogenic) in the paper.
- Evidence: Sc deck net CRE −52 W/m² (SW cooling); polar-night Arctic deck +25.6 W/m² LW
  warming. 4 radiation tests + golden green.

## Stage B — homogeneous freezing  ✅ DONE (commit cf4dc7d)
- Kuhn (2011) classical-nucleation-theory port from SAM6-LCM: `homogeneous_prob`,
  `_homogeneous_freeze` in `droplab/ice_microphysics.py`; wired in the driver
  (`homogeneous=True` default, into the same `frozen_mass`/l_f). Active below 239.1 K.
- 3 tests (threshold, deep-cold glaciation + conservation, ice=False unaffected); two ABIFM
  tests isolated with `homogeneous=False`. golden bit-identical.

## Cold-cloud cycle (박사 chose 2026-06-19): riming -> aggregation -> melting -> Hallett-Mossop
Test bed READY: `deep_cold` case (DEEP_COLD sounding, ~270 K surface to ~7 km / -36 C tops;
deep convection glaciates aloft, q_ice ~3 g/kg, sediments as snow). Collision chain fully
traced: `_collide_cells` (flow2d_driver) -> `_collide_all` (flow2d_driver:312, per-cell njit
loop, Fisher-Yates shuffle) -> `_collision_kernel` (collision_soa:175). The merge branch is
collision_soa lines 267-274 (`x_rand` drawn BEFORE the branch, so a phase-aware branch keeps
the all-liquid path bit-identical). Smaller-A gains (collision.py rule).

## Stage C — riming (ice collects supercooled liquid)  ✅ DONE (golden bit-identical)
- Implemented: `_rime_update` in `droplab/collision_soa.py` (liquid mass transfers onto the
  ice super-droplet, result keeps phase=ice, rimed mass accumulated for l_f); phase-aware
  merge branch (`x_rand` drawn BEFORE the branch, so phase-all-zero is byte-identical);
  `phase`/`rimed` threaded through `collide_soa` -> `_collide_all` -> `_collide_cells`;
  `flow2d_dynamic` passes `rimed_out` and releases l_f (`theta += l_f/cp * rimed/air_mass`).
- Evidence: `deep_cold` snow case, sediment off -> IWP grows with collisions on vs off
  (22.7 vs 21.1), total condensate conserved to 0.13 %, column no colder (l_f). 2 riming
  tests + golden bit-identical (`tests/test_flow2d_golden.py` 1 passed) + 22-test sweep green.

### (historical) Stage C plan  (golden-sensitive)
- **[목표]** ice super-droplet collides with a liquid one -> liquid mass rimes onto the ice;
  l_f released; the riming pathway grows graupel/snow in mixed-phase decks.
- **[계획]** make the numba collision core phase-aware (MVP: reuse the existing gravitational
  kernel + collision efficiency; SAM's habit-dependent Boehm/EM17 efficiency = ice-habits
  tier, OUT of MVP scope). Precise hooks:
  1. `_collision_kernel(...)` in `droplab/collision_soa.py`: add `phase`, `frozen_mass`
     (per-droplet phase in, per-cell fusion accumulator out).
  2. At the merge point, branch on phase: SAME phase -> current `_liquid_update_collection`
     / `_same_weights_update` (UNCHANGED); MIXED (one ice, one liquid) -> a new
     `_rime_update`: the liquid mass transfers to the ice super-droplet, the result keeps
     `phase=ice`, and the transferred liquid mass is added to `frozen_mass` (l_f).
  3. Thread `phase`/`frozen_mass` through `collide_soa` and `_collide_cells`.
  4. `flow2d_dynamic`: pass `phase`; release l_f from the riming `frozen_mass` (same pattern
     as the freezing block).
- **[가정]** collection efficiency from the existing gravitational kernel + ice fall speed
  (`_fall_speeds` is already phase-aware); rime adds to ice mass with ρ_ice bookkeeping.
- **[실행]** implement hooks 1–4; new `tests/test_riming.py` (mixed-phase box: ice grows,
  liquid depletes, water + energy conserved).
- **[증거]** in a supercooled mixed deck, ice mass grows faster and liquid depletes faster
  with riming on vs off.
- **[검증 — HARD GATE]** `pytest tests/test_flow2d_golden.py` BIT-IDENTICAL after EVERY edit
  (phase all-zero must take the exact current path with the same RNG draw order); plus
  ice-only / liquid-only regressions; mass + energy conservation.
- **[리스크]** the collision core is the golden lynchpin — any change to the phase=0 RNG/
  arithmetic order breaks bit-identity and invalidates every figure. Do it edit-by-edit,
  golden-gated; revert immediately on any golden diff.
- **[다음]** D.

## Cold-cloud cycle — melting  ✅ DONE (not golden-sensitive)
- `_melt` in `droplab/ice_microphysics.py`: instantaneous phase flip 1->0 for any ice
  super-droplet in a cell warmer than `T_MELT` (273.15 K, strict). Mass M and solute Ns are
  untouched (phase flag only), so freeze<->rime<->melt conserves water. Wired in
  `flow2d_dynamic` after the freezing block behind a `melt=True` toggle; releases l_f with
  the OPPOSITE sign of freezing (`theta -= l_f/cp * melted/air_mass`, a heat sink).
- Test bed reality: in a deep storm the homogeneous ice forms ~4 km above the 0 C level and
  cannot sediment that far against the updraft in a short run. The affordable melt regime is
  a SHALLOW mixed-phase deck (warm base under a Bigg-glaciated supercooled cloud ~1 km up).
  `tests/test_melting.py`: deterministic `_melt` unit tests + an instrumented shallow-deck
  run where melting demonstrably fires (126 steps, 263 units melted) and conserves. golden
  bit-identical.

## Cold-cloud cycle — Hallett-Mossop (rime splintering)  ✅ DONE (golden bit-identical)
- Does NOT touch the collision core: H-M is a function of the per-cell rimed mass that Stage
  C already returns, so it runs as a post-collision step in `flow2d_dynamic`, gated by `ice`.
- `_hallett_mossop` in `ice_microphysics.py`: in cells that rimed inside the -3..-8 C window
  (triangular `hm_factor`, peak -5 C), N = 350 splinters per mg of rime. Following the
  breakup scheme's `_merge_fragments_into_nearest`, the splinter NUMBER is added to the
  multiplicity of an existing ice SD (the rimer/graupel with the most hosting capacity) --
  NO new super-droplets, so the population never grows unbounded (박사's constraint). Host
  mass unchanged -> exactly mass-conserving; a cap keeps per-crystal mass >= one splinter.
- Evidence: shallow mixed-phase Bigg deck -> 228 steps splinter, 6.7e11 crystals added; the
  extra ice number enhances deposition so q_ice grows faster with H-M on (1024 vs 755) --
  the ice-multiplication feedback. SD count never exceeds the initial allocation.
- `tests/test_hallett_mossop.py` (4): triangular factor, mass-conserving renumber + fixed
  count + number multiply, splinter cap, integration multiplication. Toggle `hallett_mossop`.
- DROPPED: the LBA sounding (couldn't glaciate in an affordable idealized-bubble run; 박사
  flagged it as not meaningful). Deep convection remains a future goal (needs real dynamics /
  long integration), not a stub sounding.

## Stage D — integration + validation
- Turn riming + homogeneous on in the Arctic/glaciogenic scenarios; show LW+SW changes
  (Stage A) of the richer ice. Conservation figures; compare to SAM6-LCM where possible.
  All tests green; ice=False forever bit-identical.

## Cirrus case  DONE (homogeneous-freezing test bed; riming deferred)
- SAM6-LCM has NO dedicated cirrus prm (confirmed: TWPICE/BOMEX/GATE/KWAJEX/DYCOMS/MOSAiC
  only; "cirrus" appears solely in the M2005 graupel module). So BUILD a cirrus case in
  `examples/cloud_cases.py`: a high, cold deck (cloud top ~ −40 to −60 °C) where
  homogeneous freezing forms the ice -> the natural test bed for Stage B. Then a short
  demo/figure (ice forms via homogeneous freezing; net CRE positive = warming, Stage A).

## Merge policy (decided 2026-06-15, 박사 "안전·효율")
`feature/radiation-ice` is stacked on the droplab rename + the other agent's edu-framework
work (≈20 commits ahead of main). The radiation work depends on the rename (imports use
`droplab.`), so cherry-picking onto main is not clean. SAFE + EFFICIENT = keep developing
C/D/cirrus on this branch and merge the whole coherent stack to main once everything is
green and coordinated with the other agent — not a premature partial merge.
