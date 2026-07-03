"""Render a STILL of the toy lightning on the proper anelastic cumulonimbus tower
(deep_convection case): condensate shroud + charge dipole (red +/blue -) + the branched
dielectric-breakdown discharge channel. Saves a PNG (Desktop + repo).

Electrification is a pure diagnostic (no feedback): the cloud field is identical to the
same run with electrification=False; only the charge/field/flash overlay is added.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter

from examples.cloud_cases import CASES
from droplab.flow2d_dynamic import run_flow2d_dynamic

OUT = os.path.expanduser("~/Desktop/droplab_lightning_still.png")
OUT_REPO = "lightning_still.png"


def main():
    cfg = dict(CASES["deep_convection"])           # the real anelastic cumulonimbus tower
    # n_super=160000: high ice resolution -> extended charge structure (not a 2-cell artifact)
    cfg.update(n_super=160000, nt=1400, collect_every=8, electrification=True,
               charge_eff=0.3,
               E_breakdown=400.0)   # illustrative 2-D threshold (see ELECTRIFICATION_AUDIT.md)
    print("running anelastic cumulonimbus with electrification ...")
    o = run_flow2d_dynamic(**cfg)
    flow, frames = o["flow"], o["frames"]
    total = sum(len(f.get("flashes", [])) for f in frames)
    flashed = [f for f in frames if f.get("flashes")]
    print(f"frames={len(frames)} total flashes={total} ; frames-with-flash={len(flashed)}")
    if not flashed:
        print("NO FLASHES — lower E_breakdown (2-D field is weak)."); return
    # pick the single longest (most segments) discharge across the run, and show ONLY it,
    # so the branched channel is legible (overlaying several flashes makes a blob)
    best = max(((f, fl) for f in frames for fl in f["flashes"]),
               key=lambda p: len(p[1]["segments"]))
    pick, one_flash = best
    pick = dict(pick); pick["flashes"] = [one_flash]

    xe = np.linspace(0, flow.X, flow.Nx + 1)
    ze = np.linspace(0, flow.Z, flow.Nz + 1)
    cd = pick["charge_density"]
    cloud = gaussian_filter(pick["qc"] + pick.get("q_ice", 0.0), 0.7)  # smooth the SD speckle
    cloud = np.where(cloud < 0.02, np.nan, cloud)  # transparent where no condensate

    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    ax.set_facecolor("#070b16")                    # night sky
    ax.pcolormesh(xe, ze, cloud.T, cmap="bone", vmin=0.0, vmax=max(0.6, np.nanmax(cloud)),
                  shading="flat", alpha=0.8, zorder=1)               # cloud = light shroud
    lim = np.percentile(np.abs(cd[cd != 0]), 97) if (cd != 0).any() else 1e-12
    ax.pcolormesh(xe, ze, np.where(cd == 0, np.nan, cd).T, cmap="bwr",
                  vmin=-lim, vmax=lim, shading="flat", alpha=0.7, zorder=2)
    for fl in pick["flashes"]:
        seg = fl["segments"]
        if len(seg) == 0:
            continue
        segs = seg.reshape(-1, 2, 2)
        ax.add_collection(LineCollection(segs, colors="white", linewidths=3.4, alpha=0.9, zorder=5))
        ax.add_collection(LineCollection(segs, colors="#fff2a8", linewidths=1.4, alpha=1.0, zorder=5.1))
    # zoom to the charged storm region
    cdmask = np.abs(cd) > 0
    if cdmask.any():
        xcc = 0.5 * (xe[:-1] + xe[1:]); zcc = 0.5 * (ze[:-1] + ze[1:])
        ii, jj = np.where(cdmask)
        ax.set_xlim(max(xcc[ii].min() - 2000, 0), min(xcc[ii].max() + 2000, flow.X))
        ax.set_ylim(0, min(zcc[jj].max() + 2000, flow.Z))
    else:
        ax.set_xlim(0, flow.X); ax.set_ylim(0, flow.Z)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    grounded = sum(fl["grounded"] for fl in pick["flashes"])
    ax.set_title(f"DropLab toy lightning on the anelastic cumulonimbus — step {pick['step']}\n"
                 f"{len(pick['flashes'])} discharge(s) ({grounded} to ground)   |   "
                 f"charge: red + (crystals) / blue − (graupel)   |   {total} flashes total",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120, facecolor="#0a0e1a")
    fig.savefig(OUT_REPO, dpi=120, facecolor="#0a0e1a")
    print("saved", OUT, "and", OUT_REPO)


if __name__ == "__main__":
    main()
