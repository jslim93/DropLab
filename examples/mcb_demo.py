"""Marine Cloud Brightening (MCB) demonstration — the climate-intervention chapter.

Runs a DYCOMS-II marine stratocumulus deck TWICE with the SAME meteorology (same
turbulence seed): an unseeded control and an MCB run in which sub-micron sea-salt
aerosol is injected across the boundary layer (a fleet of CCN sprayers). The extra
nuclei activate into more cloud droplets, so the SAME liquid water is shared among
MORE, SMALLER droplets — the effective radius drops, the optical depth rises, and
the deck reflects more sunlight. This is the Twomey effect, the physical basis of
marine cloud brightening.

Why a DOMAIN-MEAN comparison, not a single "ship track": a cloud-resolving model is
chaotic, so injecting aerosol also nudges the turbulence and the two runs decorrelate
column-by-column within minutes. The Twomey signal is robust only in the spatial
mean (the chaos averages out, the systematic radius shift remains) — which is exactly
why real ship tracks show up in TIME-MEAN satellite composites, not single snapshots.
This demo therefore reports the distribution shift and the domain-mean forcing.

Run:  python -m examples.mcb_demo
Saves: /tmp/mcb_brightening.png  and prints the Twomey radiative-forcing summary.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from droplab.flow2d import Flow2D
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.flow2d_viz import animate_mcb
from droplab.soundings import DYCOMS, DYCOMS_RADIATION
from droplab.climate_diag import column_optics, twomey_report

# domain / deck configuration (shared by both runs so the meteorology is identical).
# Background N≈250/cm³ is a realistic NON-precipitating marine stratocumulus
# (r_eff≈10 µm, the textbook Sc droplet size) — the proper target to brighten. A
# much cleaner deck would already be drizzling itself away (big drops, raining),
# which is the wrong baseline for a marine-cloud-brightening demonstration.
CFG = dict(dt=1.0, Nx=96, Nz=48, X=4800.0, Z=1200.0, n_super=60000,
           sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
           N_modes=(250.,), pert_amp=0.1, nu=6, nu_scalar=1.5, collisions=True,
           switch_TICE=True, eps=0.01, sediment=True, collect_every=50, seed=3)

# MCB seeding: sub-micron sea-salt across the whole boundary layer mid-run
MCB_SEED = dict(t_inject=400.0, x_frac=(0.0, 1.0), z_lo=50.0, z_hi=500.0,
                N_cm3=200.0, r_um=0.1, kappa=1.2, n_super=12000)


def run(nt=1500):
    base = run_flow2d_dynamic(nt=nt, **CFG)
    seeded = run_flow2d_dynamic(nt=nt, seeding=MCB_SEED, **CFG)
    return base, seeded


def figure(base, seeded, path="/tmp/mcb_brightening.png"):
    flow = Flow2D(X=CFG["X"], Z=CFG["Z"], Nx=CFG["Nx"], Nz=CFG["Nz"])
    ob = column_optics(base["M"], base["A"], base["x"], base["z"], flow)
    os_ = column_optics(seeded["M"], seeded["A"], seeded["x"], seeded["z"], flow)
    cb, cs = ob["lwp"] > 1e-4, os_["lwp"] > 1e-4         # cloudy columns only

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # effective-radius distribution shift (the Twomey fingerprint)
    bins_r = np.linspace(0, 30, 26)
    ax[0].hist(ob["reff"][cb] * 1e6, bins_r, color="C0", alpha=0.55, label="unseeded")
    ax[0].hist(os_["reff"][cs] * 1e6, bins_r, color="C3", alpha=0.55, label="seeded (MCB)")
    ax[0].axvline(ob["reff_mean"] * 1e6, color="C0", lw=2)
    ax[0].axvline(os_["reff_mean"] * 1e6, color="C3", lw=2)
    ax[0].set_xlabel("effective radius [µm]"); ax[0].set_ylabel("cloudy columns")
    ax[0].set_title("droplets shrink (more CCN, same water)")
    ax[0].legend(fontsize=8)
    # albedo distribution shift
    bins_a = np.linspace(0, 0.7, 29)
    ax[1].hist(ob["albedo"][cb], bins_a, color="C0", alpha=0.55, label="unseeded")
    ax[1].hist(os_["albedo"][cs], bins_a, color="C3", alpha=0.55, label="seeded (MCB)")
    ax[1].axvline(ob["albedo_mean"], color="C0", lw=2)
    ax[1].axvline(os_["albedo_mean"], color="C3", lw=2)
    ax[1].set_xlabel("cloud albedo"); ax[1].set_ylabel("cloudy columns")
    ax[1].set_title("cloud brightens (higher albedo)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Marine cloud brightening in a stratocumulus deck — the Twomey effect")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main(make_gif=False):
    base, seeded = run()
    path = figure(base, seeded)
    flow = Flow2D(X=CFG["X"], Z=CFG["Z"], Nx=CFG["Nx"], Nz=CFG["Nz"])
    rep = twomey_report(base, seeded, flow)
    b, s = rep["base"], rep["seeded"]
    print(f"\n{'':14s}{'unseeded':>12s}{'seeded(MCB)':>14s}")
    print(f"{'r_eff [µm]':14s}{b['reff_mean']*1e6:>12.2f}{s['reff_mean']*1e6:>14.2f}")
    print(f"{'optical τ':14s}{b['tau_mean']:>12.2f}{s['tau_mean']:>14.2f}")
    print(f"{'albedo':14s}{b['albedo_mean']:>12.3f}{s['albedo_mean']:>14.3f}")
    print(f"\n  Δr_eff = {rep['d_reff_um']:+.2f} µm   Δτ = {rep['d_tau']:+.2f}"
          f"   Δalbedo = {rep['d_albedo']:+.4f}")
    print(f"  TOA shortwave forcing ≈ {rep['forcing_wm2']:+.2f} W/m²  (negative = cooling)")
    print(f"\n  figure → {path}")
    if make_gif:
        fig, anim = animate_mcb(base, seeded, t_inject=MCB_SEED["t_inject"],
                                dt=CFG["dt"], fps=8, r_max=40.0)
        gif = "/tmp/mcb_demo.gif"
        anim.save(gif, writer="pillow", fps=8)
        print(f"  animation → {gif}")


if __name__ == "__main__":
    main(make_gif="--gif" in sys.argv)
