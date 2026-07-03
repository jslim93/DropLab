# DropLab Technical Note — 2-D Dynamics, Electrification, and the Smagorinsky SGS Closure

*A from-the-physics-up explanation of three DropLab subsystems: what the real-world physics is,
how DropLab codes it, how much is tuned, how it differs from conventional research codes (SAM,
WRF, Mansell's COMMAS), and an honest realism verdict for each. Written so it can be taught.*

Conventions: `θ` potential temperature, `b` buoyancy, `q_v`/`q_c` vapour/cloud mixing ratios,
`ω` vorticity, `ψ` streamfunction, `ε₀` vacuum permittivity, `φ` electric potential.

---

## PART 1 — The 2-D dynamics: vorticity–streamfunction convection

### 1.1 The real physics

DropLab solves **2-D incompressible convection under the Boussinesq (or anelastic)
approximation**. Start from the momentum + continuity equations for a velocity `(u, w)` in the
`x–z` plane with buoyancy `b` acting upward:

```
Du/Dt = −(1/ρ)∂p/∂x                      (horizontal momentum)
Dw/Dt = −(1/ρ)∂p/∂z + b                  (vertical momentum, buoyancy source)
∂u/∂x + ∂w/∂z = 0                        (incompressible continuity)
```

The pressure `p` is a nuisance: it's whatever it has to be to keep the flow divergence-free.
The classic trick to **eliminate pressure entirely** is the *vorticity–streamfunction* form:

1. **Streamfunction `ψ`** automatically satisfies continuity. Define `u = −∂ψ/∂z`, `w = ∂ψ/∂x`.
   Then `∂u/∂x + ∂w/∂z = −∂²ψ/∂x∂z + ∂²ψ/∂x∂z = 0` — divergence-free *by construction*. No
   pressure solve needed to enforce it.

2. **Vorticity** `ω = ∂u/∂z − ∂w/∂x` (the single out-of-plane component in 2-D). Substituting
   the streamfunction gives a **Poisson equation linking them**: `∇²ψ = −ω`.

3. **Take the curl of the momentum equations.** The curl of a gradient (the pressure term) is
   zero — that's the whole point, pressure drops out. What survives is the vorticity equation:

   ```
   Dω/Dt = −∂b/∂x + ν∇²ω
   ```

   The source `−∂b/∂x` is **baroclinic generation**: a *horizontal gradient* of buoyancy spins
   up vorticity. Physically — a warm bubble (high `b` in the middle) is lighter than its flanks,
   so the buoyancy gradient on each side drives a circulation that lifts the centre. That is how
   a thermal becomes an updraft. (The minus sign is just because DropLab defines `ω = ∂u/∂z −
   ∂w/∂x`; flip the sign convention and it's the textbook `+∂b/∂x`.)

The buoyancy itself couples the cloud to the flow:

```
b = g·( θ'/θ₀  +  0.608·q_v'  −  q_c )
```

- `θ'/θ₀` — warmer-than-environment air is buoyant (latent heat of condensation warms it).
- `0.608·q_v'` — water vapour is lighter than dry air (virtual-temperature effect).
- `−q_c` — **condensate loading**: suspended droplets/ice are dead weight that drags the parcel
  down. This is the term that lets heavy precipitation kill an updraft.

So the loop is: **buoyancy gradient → vorticity → streamfunction → velocity → lifts air →
condensation → latent heat → more buoyancy.** That's self-organizing moist convection.

### 1.2 How DropLab codes it (`droplab/flow2d_dynamic.py`)

Per timestep, in order:
1. `psi = _poisson(-omega, …)` — solve `∇²ψ = −ω` with an FFT/spectral Poisson solver
   (`droplab/poisson.py`).
2. `flow.u, flow.w = _faces(psi, …)` — velocities from `ψ` derivatives on a staggered grid.
3. `omega = _upwind(omega, Cx, Cz)` — advect vorticity (upwind); scalars (`θ`, `q_v`) are
   advected by **MPDATA** (`droplab/mpdata.py`), a positive-definite, low-diffusion scheme.
4. `omega += dt·(−∂b/∂x + ν·∇²ω)` — baroclinic source + viscous diffusion (line ~546).
5. Condensation/collision/sedimentation act on the super-droplets; latent heating updates `θ`.

The Lagrangian super-droplets ride the resolved `(u, w)` velocity field; condensation uses each
droplet's local cell thermodynamics.

### 1.3 What's tuned, and how it differs from conventional codes (SAM, WRF)

This is the important part for honesty:

- **Hard caps `b_max`, `omega_max`, `v_max`.** DropLab *clips* buoyancy, vorticity, and velocity
  every step. **Conventional LES does NOT do this.** These caps exist because a 2-D model with a
  single constant viscosity and no proper subgrid/entrainment closure will otherwise run a
  grid-scale moist instability and blow up. The buoyancy cap doubles as a crude **entrainment
  closure** (real updrafts dilute with dry air to a few K of buoyancy; the cap mimics that
  ceiling). **This is the single most "tuned" part of DropLab** — the caps are knobs chosen so
  the convection looks right and stays stable, not first-principles physics.
- **Constant viscosity `ν`.** Real LES uses a *strain-dependent* eddy viscosity (Part 3). DropLab
  defaults to a constant `ν` for stability; the Smagorinsky closure (Part 3) is the optional
  upgrade.
- **2-D.** SAM/WRF are 3-D. **2-D turbulence has an *inverse* energy cascade** (energy flows to
  *large* scales), the opposite of real 3-D turbulence (energy → small scales → dissipation). So
  2-D "turbulence" is not real turbulence — it's organized overturning that looks cloud-like.
- **Vorticity–streamfunction vs primitive equations.** SAM/WRF solve the primitive equations with
  an anelastic/compressible pressure solver. DropLab's `ω–ψ` form is elegant in 2-D but doesn't
  generalize to 3-D (vorticity is a vector with stretching/tilting in 3-D) — another reason this
  is a 2-D-only design.

### 1.4 Realism verdict

**Qualitatively convection-like; not quantitatively predictive.** The governing equations are
correct Boussinesq/anelastic dynamics, and the microphysics riding on top is faithful. But the
caps make the dynamics partly *tuned for appearance and stability*, and 2-D is the wrong
dimensionality for turbulence. Use it to *show* how buoyancy drives an updraft that makes a
cloud — not to predict updraft speeds or cloud structure.

---

## PART 2 — Electrification and lightning

This is the subsystem you said you wanted to actually understand. Two separate questions: (a) how
does a thundercloud *charge up*, and (b) how is the *electric field* generated and discharged.

### 2.1 How a thundercloud charges — the non-inductive mechanism

The dominant charging mechanism in real storms (and the only one DropLab models) is
**non-inductive charging** (Takahashi 1978; Saunders & Peck 1998):

- In the mixed-phase zone (roughly −10 to −40 °C) you have three ingredients together: **graupel**
  (dense, riming ice, ~mm, falls fast), small **ice crystals** (vapour-grown, ~tens of µm, lofted
  by the updraft), and **supercooled liquid droplets**.
- When a falling graupel pellet **collides and rebounds** off a small ice crystal, a tiny amount
  of charge (~1–100 fC, femtocoulombs) transfers between them.
- **The sign of the transfer depends on temperature and cloud liquid water content.** There is a
  **charge-reversal temperature** (around −10 to −20 °C, really a curve in T–LWC space): *colder*
  than reversal, graupel charges **negative**; *warmer*, graupel charges **positive**. The light
  crystal always takes the opposite sign. (Why: the "relative diffusional growth rate" theory —
  the ice surface growing faster from vapour ends up charging positive relative to the other.
  Baker, Dash, Saunders.)
- **Gravity then separates the charge.** Heavy graupel falls; light crystals are carried up by the
  updraft. So the two opposite charges end up at different heights → a vertical **charge dipole**
  (real storms make a **tripole**: main negative near −15 °C, upper positive aloft, and a smaller
  lower-positive pocket).

There is also an **inductive** mechanism (droplets polarized by the *existing* field exchange
charge on collision) — secondary, and **not** in DropLab.

### 2.2 How the electric field is generated (electrostatics)

Once charge is separated, the field follows from **classical electrostatics** — this is the part
that's just physics, no storm-specific magic:

1. The separated charges define a **charge density** `ρ_q` (C/m³) on the grid.
2. **Gauss's law**: `∇·E = ρ_q/ε₀`. Because the field is (electro)static, `E = −∇φ` for a
   potential `φ`. Substituting gives **Poisson's equation**:

   ```
   ∇²φ = −ρ_q/ε₀
   ```

3. **Solve Poisson for `φ`, then `E = −∇φ`.** A dipole of separated charge produces a strong field
   *between* the charge centres (and toward the ground). This is identical in spirit to the
   streamfunction solve in Part 1 — same elliptic solver, different right-hand side.
4. As more charge piles up, `ρ_q` grows, so `|E|` grows. When `|E|` exceeds the **breakdown field**
   somewhere, the air can no longer insulate and a discharge starts.

The **breakdown threshold** falls with altitude: `E_crit(z) = E_break·ρ(z)/ρ₀` (thinner air breaks
down at a lower field — "runaway breakdown"). DropLab's default `E_break ≈ 1.5×10⁵ V/m` is in the
observed lightning-*initiation* range (real initiation fields are ~10× below classical
sea-level breakdown, because hydrometeors and energetic electrons help it along).

### 2.3 How the lightning discharge works

**Real lightning** is a sequence: a local electron avalanche → a **streamer** → a hot, conducting,
nearly **equipotential leader channel** that propagates, intensifying the field at its tip and
**redistributing/screening** charge as it goes (it's *bidirectional* — Kasemir 1960 — growing both
ways and carrying no net charge), and for cloud-to-ground, a **return stroke** (the bright surge of
current when the channel touches ground).

**The DBM (Dielectric Breakdown Model, Niemeyer–Pietronero–Wiesmann 1984)** is a stochastic
*fractal-growth* model for the **channel geometry**: the channel grows one cell at a time from the
initiation point, adding a neighbouring cell with probability

```
P_i  ∝  |φ_i − φ_0|^η
```

i.e. it preferentially grows along the steepest potential gradient. `η` tunes the shape: `η=1` is
bushy/space-filling, larger `η` is more filamentary (DropLab uses `η=3` so it reads as a forked
bolt). Mansell et al. (2002) is the 3-D thunderstorm version.

### 2.4 How DropLab codes it (`droplab/electrification.py`)

- `deposit_charge` — non-inductive charging: graupel (ice with size > 0.2 mm) in the −10…−20 °C
  zone sweeps through crystals; charge `≈ N_collisions · δq` with the reversal sign; crystals get
  the exact equal-and-opposite total (charge conserved to machine precision).
- `charge_density` → `solve_potential` (`∇²φ = −ρ_q/ε₀`, same Poisson solver as the dynamics) →
  `efield` (`E = −∇φ`).
- `breakdown_field` (altitude-scaled threshold), `dbm_leader` (the fractal channel),
  `flash` (fires if `|E| > E_crit`, neutralizes channel charge toward its mean — a net-neutral
  bidirectional-leader idea).

### 2.5 What's tuned, and how it differs from conventional codes (Mansell COMMAS, Barthe)

- **Dipole, not tripole.** Single reversal temperature (Saunders) vs the full T–LWC sign chart →
  no lower-positive pocket. Conventional storm-electrification codes carry the full
  parameterization on each hydrometeor category.
- **The "bolt" is a *static-field drawing*, not a leader.** DropLab freezes `φ₀` and **never
  re-solves the field as the channel grows**. A real leader is conducting and *screens* the field;
  DropLab's channel just traces the *pre-existing* gradient. So there's **no leader physics, no
  return stroke, no current, no charge redistribution feedback**. The code itself notes the
  intracloud-vs-cloud-to-ground behaviour is "partly a numerical artifact" of the `max_cells` cap
  and grounded-box boundary.
- **Diagnostic only.** Charge/field/flash do **nothing** to the dynamics, microphysics, or
  temperature. No lightning heating, no NOₓ.
- **2-D geometry.** Line charges (field falls as `1/r`, not `1/r²`) and a `φ=0` grounded box on all
  walls — both distort the field structure.
- **The fundamental 2-D wall.** A clean vertical dipole needs organized flow to separate graupel
  from crystals, but the *same* organized flow breaks the *local* collisional charging (they have
  to be colliding to charge). In 2-D you can't have both well — which is exactly why real storm
  electrification is done in 3-D.
- **Tuned knobs:** `charge_eff` (rebound fraction ~0.1), `E_break`, `max_cells`, `η`. The charging
  *constants* (`δq ≈ 5 fC`) are lab-grounded and *not* tuned to force flashes.

### 2.6 Realism verdict

**The least physical subsystem — a pedagogical visualization of the causal chain.** The
**charging** step is genuinely physically motivated (right mechanism, lab-grounded constants, a
defensible dipole). The **field** is correct electrostatics but on the wrong (2-D, grounded-box)
geometry. The **discharge** is a DBM *drawing* on a frozen field, not a simulated leader. It's
diagnostic-only and 2-D can't make a proper charge structure. Use it to *teach the chain* —
charging → field → breakdown → forked channel — never as a lightning prediction.

---

## PART 3 — The Smagorinsky subgrid-scale (SGS) closure

### 3.1 The real physics: why any LES needs a closure

An LES (Large-Eddy Simulation) **resolves the big, energy-containing eddies and models the small
ones**. Mathematically you *filter* the Navier–Stokes equations at the grid scale. The nonlinear
advection term produces a leftover you can't compute from the resolved field — the **subgrid stress**
`τ_ij = ⟨u_i u_j⟩ − ū_i ū_j`. You need a **closure**: an expression for `τ_ij` in terms of the
resolved velocity.

The physical job of that closure is to **drain energy at the right rate**. In 3-D turbulence,
energy is injected at large scales and *cascades* down to ever-smaller eddies until viscosity
dissipates it at the Kolmogorov scale. An LES grid can't reach that scale, so the SGS model must
remove energy *as if* the cascade-to-dissipation were happening — otherwise energy piles up at the
grid scale and the simulation goes noisy/unstable.

**Eddy-viscosity closure** (Boussinesq's analogy to molecular viscosity): model the subgrid stress
as an enhanced viscosity acting on the **resolved strain rate** `S_ij = ½(∂ū_i/∂x_j + ∂ū_j/∂x_i)`:

```
τ_ij − ⅓δ_ij τ_kk = −2 ν_t S_ij
```

**Smagorinsky (1963)** sets the eddy viscosity from dimensional analysis — a subgrid eddy has size
~`Δ` (the grid) and velocity ~`Δ|S|`, so `ν_t ~ length × velocity ~ Δ²|S|`:

```
ν_t = (C_s · Δ)² · |S| ,    |S| = √(2 S_ij S_ij) ,    Δ = √(Δx Δz)
```

`C_s ≈ 0.1–0.2` (DropLab uses 0.17). The **SGS dissipation rate** — the energy the model removes —
is `ε = 2 ν_t S_ij S_ij = ν_t |S|²`. That `ε` is *also* the natural local turbulence intensity, and
DropLab feeds it to the LEM so subgrid supersaturation broadening scales with the resolved strain.

### 3.2 How DropLab codes it (`droplab/sgs_smagorinsky.py`)

- `strain_viscosity(uc, wc, …)` — compute the strain components from the resolved velocity, form
  `|S|` and `ν_t = (C_s Δ)²|S|`.
- `div_nu_grad(ω, ν+ν_t, …)` — apply it to vorticity as a **variable-coefficient diffusion**
  `∇·((ν+ν_t)∇ω)` (a constant-`ν` Laplacian can't represent spatially-varying eddy viscosity).
- `dissipation(ν_t, |S|)` → `ε = ν_t|S|²`, fed per-cell to the LEM (`droplab/lem_driver.py`).
- Wired in `flow2d_dynamic.py` behind `smagorinsky=True`; `False` keeps the constant-`ν` path
  **bit-identical**.

### 3.3 What's tuned, and how it differs from conventional codes

- **Static Smagorinsky with fixed `C_s=0.17`.** Real LES often uses the **dynamic Smagorinsky**
  (Germano 1991 — `C_s` computed from the flow itself) or a **1.5-order prognostic-TKE closure**
  (which is what **SAM** actually uses: it carries a subgrid TKE equation, `ν_t = C_k l √e`).
  Static Smagorinsky is the simplest, most diffusive choice and is known to be over-dissipative
  near walls and in laminar shear.
- **2-D.** The closure is *built for the 3-D forward cascade*. In 2-D the cascade goes the wrong
  way, so `ν_t` is physically-shaped but modeling a cascade that doesn't exist correctly — "LES-
  flavored", not LES.
- **Strain is capped.** Because DropLab caps velocity/vorticity (Part 1), the strain `|S|` — and
  thus `ν_t` and `ε` — are limited. So the strain-derived turbulence is honestly *mild*
  (`ε ~ 5×10⁻⁴`), and the resulting LEM broadening is modest (≈×1.04) rather than the ×1.32 you get
  by prescribing an aggressive `ε`.

### 3.4 Realism verdict

**A correct closure of the wrong dimensionality.** The implementation is a faithful static
Smagorinsky and it behaves well (doesn't over-damp the cloud; ties the LEM to the resolved strain).
But it's modeling 3-D subgrid turbulence inside a 2-D flow, with capped strain. It makes DropLab
**LES-flavored for teaching** — a student can watch an eddy-viscosity closure act and feed the
microphysics — but a real LES needs 3-D + a proper (ideally TKE or dynamic) closure, which is the
3-D/GPU roadmap.

---

## PART 4 — Realism & tuning at a glance

| Subsystem | Core physics | Faithful? | Most-tuned piece | Differs from SAM/WRF/COMMAS | Predictive? |
|---|---|---|---|---|---|
| 2-D dynamics | ω–ψ Boussinesq/anelastic | equations yes | `b_max`/`omega_max`/`v_max` caps | 2-D; caps replace SGS+entrainment; ω–ψ vs primitive eqns | No — illustrative |
| Microphysics (context) | Köhler growth, ice habit, collision | **yes, bit-validated vs Fortran** | — | matches LCM lineage | Mechanism yes |
| LEM broadening | supersat-fluctuation broadening | mechanism yes (triplet map bit-for-bit) | `s_max` clamp; `eps` | run below its resolution regime | Direction yes, magnitude no |
| Electrification | non-inductive charging | charging yes (lab δq) | `charge_eff`, `E_break` | dipole not tripole; 2-D | No |
| Lightning channel | DBM fractal geometry | geometry yes | `η`, `max_cells` | **frozen field — not a real leader** | No — a drawing |
| Smagorinsky SGS | eddy-viscosity closure | implementation yes | `C_s=0.17`; capped strain | static (vs dynamic/TKE); 2-D | LES-*flavored* |

**One-line summary to tell people:** *DropLab faithfully implements the real microphysical and
electrostatic equations and validates them against the actual research Fortran — but it runs them
in an idealized, stabilized 2-D world, so it's a rigorous instrument for **seeing the mechanisms
work**, not a predictive storm model. The lightning channel in particular is a physically-motivated
**visualization**, not a simulated leader.*

---

### Key references
- Smagorinsky, J. (1963). *Mon. Wea. Rev.* 91, 99–164. (eddy-viscosity SGS)
- Germano et al. (1991). *Phys. Fluids A* 3, 1760. (dynamic Smagorinsky)
- Takahashi, T. (1978). *J. Atmos. Sci.* 35, 1536–1548. (riming/non-inductive charging)
- Saunders, C.P.R. & Peck, S.L. (1998). *J. Geophys. Res.* 103, 13949. (charge reversal)
- Kasemir, H.W. (1960). *J. Geophys. Res.* 65, 1873. (bidirectional, net-neutral leader)
- Niemeyer, Pietronero & Wiesmann (1984). *Phys. Rev. Lett.* 52, 1033. (DBM discharge)
- Mansell et al. (2002). *J. Geophys. Res.* 107, 4075. (3-D branched lightning)
- See also `docs/ELECTRIFICATION_AUDIT.md` for the electrification limits in detail.
