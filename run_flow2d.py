#!/usr/bin/env python
"""Run the 2D kinematic cumulus and save an animation.

    python run_flow2d.py                # default demo -> flow2d.gif
    python run_flow2d.py --collisions   # include collision/coalescence (rain)
    python run_flow2d.py --field supersat --out cloud.gif

A central updraft with flanking downdrafts (closed box) lifts moist air; droplets
activate and grow in the rising core, the larger ones fall, and (with collisions)
coalesce into rain. Background = cloud water q_c (or supersaturation); coloured
dots = super-droplets by radius.
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

from droplab.flow2d_driver import run_flow2d
from droplab.flow2d_viz import animate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nt", type=int, default=900)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--grid", type=int, default=48, help="Nx=Nz")
    ap.add_argument("--n-super", type=int, default=90000,
                    help="super-droplets (aim for >=25 per cell)")
    ap.add_argument("--w0", type=float, default=2.0, help="peak updraft (m/s)")
    ap.add_argument("--pattern", default="cumulus",
                    choices=["cumulus", "single_eddy"], help="flow geometry")
    ap.add_argument("--rh0", type=float, default=0.95)
    ap.add_argument("--z-bl", type=float, default=600.0, help="moist layer top (m)")
    ap.add_argument("--rh-top", type=float, default=0.2, help="RH aloft")
    ap.add_argument("--collisions", action="store_true")
    ap.add_argument("--field", default="qc", choices=["qc", "supersat", "qv"])
    ap.add_argument("--every", type=int, default=15, help="capture every N steps")
    ap.add_argument("--out", default="flow2d.gif")
    a = ap.parse_args()

    print(f"running 2D cumulus: {a.grid}x{a.grid} grid, {a.n_super} super-droplets, "
          f"nt={a.nt}, collisions={a.collisions} ...")
    result = run_flow2d(nt=a.nt, dt=a.dt, Nx=a.grid, Nz=a.grid, W0=a.w0,
                        pattern=a.pattern, n_super=a.n_super, RH0=a.rh0,
                        z_bl=a.z_bl, RH_top=a.rh_top, collisions=a.collisions,
                        collect_every=a.every)
    print(f"  -> {len(result['frames'])} frames, "
          f"{len(result['M'])} super-droplets at end")

    fig, anim = animate(result, field=a.field)
    anim.save(a.out, writer="pillow", fps=10, dpi=100)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
