"""Precipitating vs non-precipitating stratocumulus — aerosol controls the drizzle.

The same DYCOMS marine stratocumulus, run at two aerosol loadings that differ ONLY
in the background CCN number:

  CLEAN  (N~20/cm^3):  few nuclei -> large drops -> efficient collision-coalescence
                       -> heavy DRIZZLE rains the water out -> low liquid water path,
                       DIM cloud.

  POLLUTED (N~400/cm^3): many nuclei -> small drops -> drizzle is SUPPRESSED -> the
                       water stays aloft -> high liquid water path, BRIGHT cloud.

What this model actually resolves is the precipitation contrast: a drizzling, dim
deck versus a non-precipitating, bright one. (In nature this same aerosol-drizzle
switch is what flips a deck between the broken "open-cell" regime and the solid
"closed-cell" regime — but the horizontal cellular morphology itself is not what a
2D slab at this resolution shows, so we report it honestly as precip vs non-precip.)

This aerosol-precipitation feedback is a cloud "lifetime/adjustment" effect that sits
ON TOP of the instantaneous Twomey brightening, and is one of the largest
uncertainties in the climate forcing of aerosols. It is also the flip side of the two
interventions: marine cloud brightening pushes a deck toward the bright
non-precipitating state, while precipitation cloud-seeding (giant CCN) pushes a
non-raining deck toward drizzle.

Run:  python -m examples.precip_vs_nonprecip
Saves: /tmp/precip_vs_nonprecip.png  and prints the radiative / precipitation summary.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from droplab.flow2d import Flow2D
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.soundings import DYCOMS, DYCOMS_RADIATION
from droplab.climate_diag import column_optics

CFG = dict(dt=1.0, Nx=64, Nz=40, X=3200.0, Z=1200.0, n_super=40000,
           sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
           pert_amp=0.1, nu=6, nu_scalar=1.5, collisions=True, switch_TICE=True,
           eps=0.01, sediment=True, collect_every=100, seed=5)

CASES = [("clean (N≈20)", (20.,), "precipitating — drizzling, dim"),
         ("polluted (N≈400)", (400.,), "non-precipitating — solid, bright")]


def run(nt=1400):
    out = []
    for label, N_modes, sub in CASES:
        r = run_flow2d_dynamic(nt=nt, N_modes=N_modes, **CFG)
        out.append((label, sub, r))
    return out


def figure(results, path="/tmp/precip_vs_nonprecip.png"):
    flow = Flow2D(X=CFG["X"], Z=CFG["Z"], Nx=CFG["Nx"], Nz=CFG["Nz"])
    xkm = (np.arange(CFG["Nx"]) + 0.5) * flow.dx / 1000.0
    fig, ax = plt.subplots(2, 2, figsize=(11, 6),
                           gridspec_kw=dict(height_ratios=[2, 1]))
    for j, (label, sub, r) in enumerate(results):
        # top: final cloud water field (x-z) — the cellular structure
        qc = r["frames"][-1]["qc"].T                    # (Nz, Nx), g/kg
        zc = (np.arange(CFG["Nz"]) + 0.5) * flow.dz
        im = ax[0, j].pcolormesh(xkm, zc, qc, cmap="Blues", vmin=0,
                                 vmax=max(0.3, qc.max()), shading="auto")
        ax[0, j].set_title(f"{label}\n{sub}", fontsize=10)
        ax[0, j].set_ylabel("z [m]"); ax[0, j].set_xlabel("x [km]")
        fig.colorbar(im, ax=ax[0, j], label="q_c [g/kg]", fraction=0.046)
        # bottom: per-column liquid water path — open=broken, closed=full
        o = column_optics(r["M"], r["A"], r["x"], r["z"], flow)
        ax[1, j].fill_between(xkm, o["lwp"] * 1e3, color=f"C{j}", alpha=0.6)
        ax[1, j].set_ylabel("LWP [g/m²]"); ax[1, j].set_xlabel("x [km]")
        ax[1, j].set_ylim(0, 220)
    fig.suptitle("Aerosol controls the drizzle: precipitating vs non-precipitating stratocumulus")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main():
    results = run()
    flow = Flow2D(X=CFG["X"], Z=CFG["Z"], Nx=CFG["Nx"], Nz=CFG["Nz"])
    path = figure(results)
    print(f"\n{'':18s}{'precip[kg]':>12s}{'LWP[g/m²]':>11s}{'r_eff[µm]':>11s}{'albedo':>9s}")
    for label, _sub, r in results:
        o = column_optics(r["M"], r["A"], r["x"], r["z"], flow)
        print(f"{label:18s}{r['surf_precip']:>12.2e}{o['lwp_mean']*1e3:>11.1f}"
              f"{o['reff_mean']*1e6:>11.1f}{o['albedo_mean']:>9.3f}")
    print(f"\n  figure → {path}")


if __name__ == "__main__":
    main()
