"""A simple, student-friendly control panel for the climate-intervention chapter.

Four knobs and a Run button let a student explore the whole chapter without touching
code: the background aerosol of the stratocumulus deck, the entrainment-mixing degree
(IHMD), and an optional aerosol injection (marine cloud brightening or precipitation
seeding) with its size and amount. Each run reports the droplet effective radius,
cloud albedo and surface rain, and draws the deck with the injected droplets ringed.

Usage in a notebook:
    from droplab.climate_widget import climate_panel
    climate_panel()

The heavy lifting is in `simulate()` (a plain function) so it can be tested and reused
without a notebook front-end.
"""
import numpy as np
import matplotlib.pyplot as plt

from droplab.flow2d import Flow2D
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.flow2d_viz import draw_frame_seeded
from droplab.soundings import DYCOMS, DYCOMS_RADIATION
from droplab.climate_diag import column_optics

# fixed stratocumulus set-up (everything the student does NOT need to touch)
_BASE = dict(sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
             pert_amp=0.1, nu=6, nu_scalar=1.5, collisions=True, switch_TICE=True,
             eps=0.01, sediment=True, collect_every=100000)


def _seeding_spec(kind, seed_N, seed_r, nt, dt):
    """Build an injection spec for the chosen intervention. MCB sprays small sea-salt
    low in the boundary layer; precipitation seeding drops giant CCN into the cloud."""
    t_inject = max(50.0, 0.25 * nt * dt)
    if kind == "GCCN (precip)":
        return dict(t_inject=t_inject, x_frac=(0.0, 1.0), z_lo=650.0, z_hi=830.0,
                    N_cm3=seed_N, r_um=seed_r, kappa=1.2, n_super=4000)
    return dict(t_inject=t_inject, x_frac=(0.0, 1.0), z_lo=50.0, z_hi=500.0,
                N_cm3=seed_N, r_um=seed_r, kappa=1.2, n_super=8000)


def simulate(background_N=200.0, ihmd=0.0, seed_on=False, seed_kind="MCB sea-salt",
             seed_N=200.0, seed_r=0.1, nt=1000, Nx=64, Nz=40, X=3200.0, Z=1200.0,
             n_super=30000, dt=1.0, seed=3):
    """Run one stratocumulus simulation with the student's settings and return
    (result, summary). `summary` has reff_um, albedo, precip_kg and droplet_number."""
    spec = _seeding_spec(seed_kind, seed_N, seed_r, nt, dt) if seed_on else None
    res = run_flow2d_dynamic(nt=nt, dt=dt, Nx=Nx, Nz=Nz, X=X, Z=Z, n_super=n_super,
                             N_modes=(float(background_N),), ihmd=float(ihmd),
                             seeding=spec, seed=seed, **_BASE)
    flow = Flow2D(X=X, Z=Z, Nx=Nx, Nz=Nz)
    o = column_optics(res["M"], res["A"], res["x"], res["z"], flow)
    summary = dict(reff_um=o["reff_mean"] * 1e6, albedo=o["albedo_mean"],
                   precip_kg=res["surf_precip"], droplet_number=float(res["A"].sum()))
    return res, summary


def figure(res, summary, title="", path=None):
    """Draw the deck (injected droplets ringed) with the run's diagnostics."""
    flow = res["flow"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    qmax = max(res["frames"][-1]["qc"].max(), 0.3)
    draw_frame_seeded(ax, flow, res["frames"][-1], qmax, r_max=60.0)
    ax.set_title(f"{title}\nr_eff={summary['reff_um']:.1f} µm   albedo={summary['albedo']:.3f}"
                 f"   surface rain={summary['precip_kg']:.2e} kg", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig


def climate_panel(Nx=64, Nz=40, X=3200.0, Z=1200.0, n_super=30000):
    """Display the interactive control panel (call from a Jupyter notebook)."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    style = {"description_width": "180px"}
    lay = {"width": "440px"}
    w_bg = widgets.IntSlider(min=20, max=500, step=10, value=200, style=style,
                             layout=lay, description="background aerosol N [cm⁻³]")
    w_ihmd = widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.0, style=style,
                                 layout=lay, description="entrainment mixing IHMD")
    w_on = widgets.Checkbox(value=False, description="inject aerosol (seeding)")
    w_kind = widgets.ToggleButtons(options=["MCB sea-salt", "GCCN (precip)"],
                                   value="MCB sea-salt", description="intervention")
    w_sN = widgets.FloatLogSlider(base=10, min=-1, max=3, value=200.0, style=style,
                                  layout=lay, description="seed amount N [cm⁻³]")
    w_sr = widgets.FloatSlider(min=0.05, max=3.0, step=0.05, value=0.1, style=style,
                               layout=lay, description="seed dry radius [µm]")
    w_nt = widgets.IntSlider(min=400, max=2000, step=100, value=1000, style=style,
                             layout=lay, description="run length [steps]")
    w_run = widgets.Button(description="▶ Run simulation", button_style="success")
    out = widgets.Output()

    def on_kind(change):                          # sensible presets per intervention
        if change["new"] == "MCB sea-salt":
            w_sN.value, w_sr.value = 200.0, 0.1
        else:
            w_sN.value, w_sr.value = 15.0, 1.5
    w_kind.observe(on_kind, names="value")

    def on_run(_):
        with out:
            clear_output(wait=True)
            print("running… (this takes ~20-40 s)")
            res, summ = simulate(background_N=w_bg.value, ihmd=w_ihmd.value,
                                 seed_on=w_on.value, seed_kind=w_kind.value,
                                 seed_N=w_sN.value, seed_r=w_sr.value, nt=w_nt.value,
                                 Nx=Nx, Nz=Nz, X=X, Z=Z, n_super=n_super)
            clear_output(wait=True)
            tag = (f"{w_kind.value} seeding" if w_on.value else "no seeding")
            mix = "homogeneous" if w_ihmd.value == 0 else f"IHMD={w_ihmd.value:.2f}"
            figure(res, summ, title=f"N={w_bg.value} cm⁻³, {mix}, {tag}")
            plt.show()
    w_run.on_click(on_run)

    seeding_box = widgets.VBox([w_on, w_kind, w_sN, w_sr])
    display(widgets.VBox([w_bg, w_ihmd, seeding_box, w_nt, w_run, out]))
