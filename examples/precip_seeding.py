"""Precipitation cloud-seeding — tipping a marginal cloud into rain with giant CCN.

Warm-rain ("hygroscopic") cloud seeding adds a small number of GIANT cloud
condensation nuclei (~1-2 µm dry sea-salt). Because of their large solute mass they
activate at once and grow fast, becoming the first drizzle drops that fall through
the cloud collecting the small droplets (collision-coalescence) and kick-starting a
rain the cloud was on the edge of producing.

The physics here is honest and subtle — it taught us two things while building it:

  1. A heavily polluted, far-from-raining deck CANNOT be tipped by a few nuclei: the
     seeds grow to drizzle size but no deck-wide rain follows.
  2. Injecting big drops directly ("seeding" with 20 µm embryos) makes rain, but it
     is just the SEED MASS falling out — not the cloud raining. That is not seeding.

So this demo seeds a deck near its PRECIPITATION THRESHOLD (background N≈160/cm³,
barely drizzling) with a small dose of realistic 1.5 µm dry giant CCN injected as
haze INTO the cloud, where they activate alongside the cloud droplets. The proof that
it is real seeding is the ratio of precipitated mass to injected seed mass: here the
rain is ~70x the seed mass, so essentially all of it is CLOUD water the seeds
triggered, and surface precipitation roughly quadruples.

This is the precipitation-side counterpart to marine cloud brightening (examples/
mcb_demo.py): both are aerosol interventions, but seeding pushes a deck toward
drizzle while brightening pushes it the other way.

Run:  python -m examples.precip_seeding [--gif]
Saves: /tmp/precip_seeding.png  (+ /tmp/precip_seeding.gif with --gif)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from droplab.parameters import rho_liq, pi
from droplab.flow2d import Flow2D
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.flow2d_viz import draw_frame, draw_frame_seeded, animate_seeding_compare
from droplab.soundings import DYCOMS, DYCOMS_RADIATION

# background N≈160/cm³ — a deck sitting right at its precipitation threshold (only a
# trickle of natural drizzle), the regime where seeding can actually tip the balance.
CFG = dict(dt=1.0, Nx=64, Nz=40, X=3200.0, Z=1200.0, n_super=40000,
           sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True, pert_amp=0.1,
           nu=6, nu_scalar=1.5, collisions=True, switch_TICE=True, eps=0.01,
           sediment=True, collect_every=50, seed=2, N_modes=(160.,))

# a small dose of realistic 1.5 µm dry giant CCN, injected as haze into the cloud
# layer (z 650-830 m) so they activate together with the cloud droplets.
GCCN = dict(t_inject=150.0, x_frac=(0.0, 1.0), z_lo=650.0, z_hi=830.0,
            N_cm3=15.0, r_um=1.5, kappa=1.2, n_super=4000)


def _seed_mass():
    """Total liquid mass of the injected nuclei at injection (haze, r=r_um)."""
    V_reg = (GCCN["x_frac"][1] - GCCN["x_frac"][0]) * CFG["X"] \
        * (GCCN["z_hi"] - GCCN["z_lo"]) * 1.0
    N_real = GCCN["N_cm3"] * 1e6 * V_reg
    return N_real * 4.0 / 3.0 * pi * rho_liq * (GCCN["r_um"] * 1e-6) ** 3


def run(nt=1700):
    base = run_flow2d_dynamic(nt=nt, **CFG)
    seeded = run_flow2d_dynamic(nt=nt, seeding=GCCN, **CFG)
    return base, seeded


def figure(base, seeded, path="/tmp/precip_seeding.png"):
    flow = Flow2D(X=CFG["X"], Z=CFG["Z"], Nx=CFG["Nx"], Nz=CFG["Nz"])
    fig = plt.figure(figsize=(12, 7))
    axL = fig.add_subplot(2, 2, 1); axR = fig.add_subplot(2, 2, 2)
    axT = fig.add_subplot(2, 1, 2)
    qmax = max(base["frames"][-1]["qc"].max(), seeded["frames"][-1]["qc"].max())
    draw_frame(axL, flow, base["frames"][-1], "qc", qmax, r_max=80.0)
    axL.set_title("marginal deck (N≈160) — barely drizzling")
    draw_frame_seeded(axR, flow, seeded["frames"][-1], qmax, r_max=80.0)
    axR.set_title("+ small giant-CCN seeding — rain triggered")
    tb = np.array([f["step"] for f in base["frames"]]) * CFG["dt"]
    pb = np.array([f.get("surf_precip", 0.0) for f in base["frames"]])
    ts = np.array([f["step"] for f in seeded["frames"]]) * CFG["dt"]
    ps = np.array([f.get("surf_precip", 0.0) for f in seeded["frames"]])
    axT.plot(tb, pb, "C0", label="unseeded (N≈160)")
    axT.plot(ts, ps, "C3", label="seeded (1.5 µm giant CCN)")
    axT.axhline(_seed_mass(), color="0.5", ls=":", lw=1, label="injected seed mass")
    axT.axvline(GCCN["t_inject"], color="0.5", ls="--", lw=1, label="injection")
    axT.set_xlabel("time [s]"); axT.set_ylabel("cumulative surface precip [kg]")
    axT.legend(loc="upper left", fontsize=8)
    fig.suptitle("Precipitation cloud-seeding: a small dose of giant CCN tips a marginal deck into rain")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main(make_gif=False):
    base, seeded = run()
    path = figure(base, seeded)
    sm = _seed_mass()
    print(f"\nunseeded (N≈160) surface precip: {base['surf_precip']:.3e} kg")
    print(f"seeded (1.5 µm GCCN) surface precip: {seeded['surf_precip']:.3e} kg")
    print(f"  -> {seeded['surf_precip'] / max(base['surf_precip'], 1e-30):.1f}x more rain")
    print(f"  injected seed mass: {sm:.3e} kg")
    print(f"  precipitated / seed mass = {seeded['surf_precip'] / sm:.0f}x"
          f"  (>>1 => the cloud rained, not the seeds)")
    print(f"\nfigure -> {path}")
    if make_gif:
        fig, anim = animate_seeding_compare(base, seeded, fps=8, r_max=80.0,
                                            dt=CFG["dt"], metric="precip",
                                            title="giant-CCN seeding")
        gif = "/tmp/precip_seeding.gif"
        anim.save(gif, writer="pillow", fps=8)
        print(f"animation -> {gif}")


if __name__ == "__main__":
    main(make_gif="--gif" in sys.argv)
