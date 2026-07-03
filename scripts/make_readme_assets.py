"""Generate the README gallery assets into docs/assets/.

Run from the repo root:  python scripts/make_readme_assets.py
Uses the app's OWN renderers (app.ui.plots + cache) so the README shows exactly what the
sandbox shows. Quick-look configs — the whole script runs in a few minutes.
Assets: parcel_dsd.png, twod_cumulus.png, twod_stratocumulus.png, twod_mixedphase.png,
        deep_convection.gif (hero). The Streamlit screenshot is captured separately
        (headless Chrome; see the shell snippet in the module docstring history).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")

from app.ui import cache, plots, presets

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "assets")
os.makedirs(OUT, exist_ok=True)


def _quick_args(scenario, **over):
    """run_twod kwargs for a quick-look run of `scenario` (mirrors the app page)."""
    cfg = presets.sized_config(scenario, "quick")
    base = dict(scenario=scenario, resolution="quick", nt=cfg["nt"], dt=cfg["dt"],
                collisions=True, ice=False, habit=False, electrification=False,
                freezing_mode="abifm", homogeneous=True, melt=True, hallett_mossop=True,
                N_modes=(150.0,), mu_um=(0.08,), sig=(2.0,), kappa=(0.6,),
                seed_on=False, seed_kind="MCB sea-salt", seed_N=200.0, seed_r=0.1,
                inject_min=None, wind_shear=0.0, dtheta_bubble=None,
                inp_n_cm3=None, inp_r_um=None, E_breakdown=400.0, charge_eff=0.3)
    base.update(over)
    return base


def parcel_still():
    out, M, A = cache.run_parcel(0, 2000, 1500, 1.0, 293.2, 1.013e5, 0.92, 1.0,
                                 "linear", (118.0, 11.0), (0.019, 0.056), (3.3, 1.6),
                                 0.6, True, False, 0.0, 0.0, 0.0)
    fig = plots.parcel_dsd_contour(out, 1.0)
    fig.update_layout(width=760, height=430)
    fig.write_image(os.path.join(OUT, "parcel_dsd.png"), scale=2)
    print("parcel_dsd.png")


def twod_still(name, scenario, **over):
    res = cache.run_twod(**_quick_args(scenario, **over))
    if res.get("unstable"):
        raise RuntimeError(f"{scenario} quick-look went unstable")
    png = (plots.phase_image(res) if over.get("ice") else plots.scene_image(res))
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(png)
    print(name)
    return res


def hero_gif():
    res = cache.run_twod(**_quick_args("deep_convection", ice=True,
                                       electrification=True, N_modes=(200.0,)))
    gif = plots.scene_and_series_gif(res, show_field=True, wind="off", duration=160)
    path = os.path.join(OUT, "deep_convection.gif")
    with open(path, "wb") as f:
        f.write(gif)
    print(f"deep_convection.gif  ({os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == "__main__":
    parcel_still()
    twod_still("twod_cumulus.png", "congestus")
    twod_still("twod_stratocumulus.png", "dycoms")
    twod_still("twod_mixedphase.png", "arctic", ice=True)
    hero_gif()
    print("done ->", OUT)
