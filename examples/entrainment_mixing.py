"""Entrainment mixing — homogeneous vs inhomogeneous (the IHMD switch).

At a stratocumulus cloud's edges, dry environmental air is mixed in (entrainment) and
some cloud water evaporates. HOW that evaporation is shared among the droplets is one
of the long-standing uncertainties in cloud microphysics, and it is set here by a
single Inhomogeneous Mixing Degree (IHMD; Lim & Hoffmann 2023):

  IHMD = 0  HOMOGENEOUS mixing: the dry air is assumed to mix instantly, so EVERY
            droplet feels the same subsaturation and shrinks a little. Droplet number
            is conserved; the spectrum just shifts to smaller sizes.

  IHMD = 1  INHOMOGENEOUS mixing: the dry air evaporates the droplets it meets first
            COMPLETELY before mixing spreads, so a FRACTION of droplets disappears
            while the survivors keep their original size. Droplet number falls; the
            mean size is preserved.

Both remove the same amount of liquid water, so the cloud-water field is the same —
but the droplet SPECTRUM differs, and that propagates: inhomogeneous mixing leaves
fewer, larger drops, which are LESS reflective (lower albedo) and MORE collision-
prone (more drizzle). This demo runs the same deck at IHMD=0 and IHMD=1 and shows
that divergence.

Run:  python -m examples.entrainment_mixing
Saves: /tmp/entrainment_mixing.png  and prints the spectrum / radiative summary.
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
           sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True, pert_amp=0.1,
           nu=6, nu_scalar=1.5, collisions=True, switch_TICE=True, eps=0.01,
           sediment=True, collect_every=100000, seed=3, N_modes=(250.,))


def run(nt=1300):
    homo = run_flow2d_dynamic(nt=nt, ihmd=0.0, **CFG)
    inho = run_flow2d_dynamic(nt=nt, ihmd=1.0, **CFG)
    return homo, inho


def figure(homo, inho, path="/tmp/entrainment_mixing.png"):
    flow = Flow2D(X=CFG["X"], Z=CFG["Z"], Nx=CFG["Nx"], Nz=CFG["Nz"])
    oh = column_optics(homo["M"], homo["A"], homo["x"], homo["z"], flow)
    oi = column_optics(inho["M"], inho["A"], inho["x"], inho["z"], flow)
    ch, ci = oh["lwp"] > 1e-4, oi["lwp"] > 1e-4

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bins_r = np.linspace(0, 25, 26)
    ax[0].hist(oh["reff"][ch] * 1e6, bins_r, color="C0", alpha=0.55, label="IHMD=0 homogeneous")
    ax[0].hist(oi["reff"][ci] * 1e6, bins_r, color="C3", alpha=0.55, label="IHMD=1 inhomogeneous")
    ax[0].axvline(oh["reff_mean"] * 1e6, color="C0", lw=2)
    ax[0].axvline(oi["reff_mean"] * 1e6, color="C3", lw=2)
    ax[0].set_xlabel("effective radius [µm]"); ax[0].set_ylabel("cloudy columns")
    ax[0].set_title("inhomogeneous → fewer, larger drops")
    ax[0].legend(fontsize=8)
    bins_a = np.linspace(0, 0.7, 29)
    ax[1].hist(oh["albedo"][ch], bins_a, color="C0", alpha=0.55, label="IHMD=0 homogeneous")
    ax[1].hist(oi["albedo"][ci], bins_a, color="C3", alpha=0.55, label="IHMD=1 inhomogeneous")
    ax[1].axvline(oh["albedo_mean"], color="C0", lw=2)
    ax[1].axvline(oi["albedo_mean"], color="C3", lw=2)
    ax[1].set_xlabel("cloud albedo"); ax[1].set_ylabel("cloudy columns")
    ax[1].set_title("larger drops → dimmer cloud")
    ax[1].legend(fontsize=8)
    fig.suptitle("Entrainment mixing: homogeneous vs inhomogeneous (the IHMD switch)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main():
    homo, inho = run()
    flow = Flow2D(X=CFG["X"], Z=CFG["Z"], Nx=CFG["Nx"], Nz=CFG["Nz"])
    path = figure(homo, inho)
    oh = column_optics(homo["M"], homo["A"], homo["x"], homo["z"], flow)
    oi = column_optics(inho["M"], inho["A"], inho["x"], inho["z"], flow)
    print(f"\n{'':22s}{'homogeneous':>14s}{'inhomogeneous':>16s}")
    print(f"{'droplet number ΣA':22s}{homo['A'].sum():>14.3e}{inho['A'].sum():>16.3e}")
    print(f"{'effective radius [µm]':22s}{oh['reff_mean']*1e6:>14.2f}{oi['reff_mean']*1e6:>16.2f}")
    print(f"{'cloud albedo':22s}{oh['albedo_mean']:>14.3f}{oi['albedo_mean']:>16.3f}")
    print(f"{'surface precip [kg]':22s}{homo['surf_precip']:>14.2e}{inho['surf_precip']:>16.2e}")
    print(f"\n  figure → {path}")


if __name__ == "__main__":
    main()
