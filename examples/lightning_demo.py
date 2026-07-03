"""Toy lightning demo on the anelastic cumulonimbus (deep_convection). Non-inductive
graupel/crystal charge separation builds a vertical dipole; where the diagnosed electric
field breaks down, a stochastic dielectric-breakdown (DBM) channel grows along the field
and branches. ONE simulation -> a dark-sky life-cycle GIF and a single-bolt still.

    python -m examples.lightning_demo

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
from droplab.flow2d_viz import animate_electric, draw_storm_electric

GIF = os.path.expanduser("~/Desktop/droplab_lightning.gif")
STILL = os.path.expanduser("~/Desktop/droplab_lightning_still.png")


def _save_still(o):
    """The single longest discharge of the run, drawn alone so the branched channel reads."""
    flow, frames = o["flow"], o["frames"]
    pairs = [(f, fl) for f in frames for fl in f.get("flashes", [])]
    if not pairs:
        print("no flashes -> no still"); return
    f, fl = max(pairs, key=lambda p: len(p[1]["segments"]))
    one = dict(f); one["flashes"] = [fl]
    xe = np.linspace(0, flow.X, flow.Nx + 1)
    ze = np.linspace(0, flow.Z, flow.Nz + 1)
    vmax = max(0.6, (f["qc"] + f.get("q_ice", 0.0)).max())
    cd = f["charge_density"]
    lim = np.percentile(np.abs(cd[cd != 0]), 97) if (cd != 0).any() else 1e-12
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    fig.patch.set_facecolor("#070b16")
    draw_storm_electric(ax, flow, one, xe, ze, vmax, lim, from_gaussian=0.7)
    # zoom to the charged region
    if (cd != 0).any():
        xcc = 0.5 * (xe[:-1] + xe[1:]); zcc = 0.5 * (ze[:-1] + ze[1:])
        ii, jj = np.where(cd != 0)
        ax.set_xlim(max(xcc[ii].min() - 2000, 0), min(xcc[ii].max() + 2000, flow.X))
        ax.set_ylim(0, min(zcc[jj].max() + 2000, flow.Z))
    fig.savefig(STILL, dpi=120, facecolor="#070b16")
    print("saved", STILL)


def main():
    cfg = dict(CASES["deep_convection"])           # the real anelastic cumulonimbus tower
    # E_breakdown is ILLUSTRATIVE: the 2-D grounded-box field is ~100x weaker than a real
    # 3-D storm, so physical charging cannot reach the real ~1.5e5 V/m threshold. ~1500 V/m
    # matches this storm's 2-D field (see docs/ELECTRIFICATION_AUDIT.md). Charging is physical.
    # n_super=160000: high ice resolution so graupel populates many cells -> the charge
    # structure is spatially extended (42k gives only ~2-3 graupel SDs -> a 2-cell artifact).
    cfg.update(n_super=160000, nt=1400, collect_every=8, electrification=True,
               charge_eff=0.3, E_breakdown=400.0)
    print("running anelastic cumulonimbus with electrification ...")
    o = run_flow2d_dynamic(**cfg)
    nfl = sum(len(f.get("flashes", [])) for f in o["frames"])
    resid = float(o["charge"].sum()) + float(o["charge_to_ground"])
    print(f"frames={len(o['frames'])}  flashes={nfl}  charge residual={resid:.2e}")

    _save_still(o)
    fig, anim = animate_electric(o, fps=12)
    anim.save(GIF, writer="pillow", fps=12, dpi=90)
    print("saved", GIF)


if __name__ == "__main__":
    main()
