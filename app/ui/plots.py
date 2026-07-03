"""Visualization for the sandbox — regime-aware 2-D renders plus the parcel and
overlay charts.

The 2-D field renders REUSE the proven droplab renderers
(``flow2d_viz.draw_frame``, ``flow2d_viz.draw_storm_electric``) by passing a
lightweight flow proxy rebuilt from the cached ``meta`` scalars — so the look
matches the notebooks and example scripts exactly, and the cache never has to
pickle a Flow2D object. The crystal-habit gallery and the mixed-phase / Bergeron
views are adapted from ``examples/_habit_still.py`` and
``examples/cloud_cases.py``.

``regime_views`` is the key UX helper: given a run payload it decides which views
to show purely from which fields the frames actually contain (q_ice/phase only
when ice ran, phi only with habit, flashes only with electrification).

Everything here is presentation only — all field math reads the returned frame
dicts; no physics.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from droplab.flow2d_viz import draw_frame, draw_storm_electric, draw_frame_seeded
from droplab.parameters import rho_liq, pi

VIRIDIS = "viridis"
PLATE = "#3B6FB5"
COLUMN = "#C0504D"
NIGHT = "#070B16"


def flow_proxy(meta):
    """Rebuild the minimal flow interface (X, Z, Nx, Nz, dx, dz) the renderers
    use, from the cached meta scalars."""
    return SimpleNamespace(X=meta["X"], Z=meta["Z"], Nx=meta["Nx"],
                           Nz=meta["Nz"], dx=meta["dx"], dz=meta["dz"])


def _r_max(scenario):
    if scenario == "fog":
        return 20.0
    if scenario in ("deep_convection", "deep_cold", "congestus", "cirrus"):
        return 140.0
    return 80.0


def _png(fig, dpi=115, facecolor=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=facecolor or fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# --- regime selection ------------------------------------------------------ #
def regime_views(result):
    """Ordered list of view specs to render for this run, chosen from the fields
    present in the frames. Each spec: (key, title, caption)."""
    meta = result["meta"]
    f0 = result["frames"][0]
    views = [("scene", "Cloud scene",
              "Super-droplets coloured by radius (rain stands out) over the "
              "cloud-water field q_c. The headline view for every regime.")]
    if "q_ice" in f0:
        views.append(("phase", "Liquid vs ice",
                      "The condensate split into supercooled liquid (q_c) and "
                      "ice (q_i). Sedimenting ice is the model's snow."))
        views.append(("bergeron", "Bergeron hand-off",
                      "Domain liquid- and ice-water paths over time — supercooled "
                      "liquid converting to ice (the Wegener–Bergeron–Findeisen process)."))
    if "phi" in f0:
        views.append(("habit", "Ice-habit gallery",
                      "Crystal SHAPES predicted from temperature: aspect ratio "
                      "φ = c/a, plates (φ<1, blue) ↔ columns (φ>1, red)."))
    if meta.get("electrification") and result.get("n_flashes", 0) >= 0:
        views.append(("electric", "Charge & lightning",
                      "Dark-sky view: the charge dipole (red + on crystals, blue − "
                      "on graupel) and the dielectric-breakdown discharge channel."))
    return views


# --- the headline scene (reuses draw_frame) -------------------------------- #
def _vis_vmax(result):
    """A robust colour-scale max for the scene: the 80th percentile of the
    per-frame q_c maxima, so a brief convective SPIKE (e.g. deep_cold) doesn't
    wash out the rest of the run. Falls back to qc_max for steady cases."""
    fmax = [float(f["qc"].max()) for f in result["frames"]]
    return max(0.5, float(np.percentile(fmax, 80))) if fmax else result.get("qc_max", 0.5)


def _draw_scene(ax, flow, frame, qc_max, r_max, show_field, wind, dt, scenario):
    if not show_field:
        frame = dict(frame)
        frame["qc"] = frame["qc"] * 0.0
    draw_frame(ax, flow, frame, "qc", vmax=qc_max, r_max=r_max,
               show_aerosol=(scenario != "fog"), quiver=(wind != "off"),
               quiver_style=("arrows" if wind == "arrows" else "streamlines"))
    ax.set_title(f"{scenario}   t = {frame['step'] * dt:.0f} s")


def live_frame_fig(flow, frame, scenario, dt, show_field, wind, qc_max):
    """A single frame drawn DURING a streaming run, straight from the engine's
    real Flow2D + raw frame dict (no proxy) — the live 'watch it compute' view
    before the run is cached and looped. The caller MUST ``plots.close(fig)``
    after rendering (st.pyplot(clear_figure=True) clears but does not close, so
    figures would otherwise accumulate over a long live run)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _draw_scene(ax, flow, frame, qc_max, _r_max(scenario), show_field, wind,
                dt, scenario)
    fig.tight_layout()
    return fig


def close(fig):
    """Close a Matplotlib figure to free it (prevents the live-render leak)."""
    plt.close(fig)


def scene_image(result, show_field=True, wind="off"):
    """Static final-frame cloud scene."""
    meta = result["meta"]
    flow = flow_proxy(meta)
    fig, ax = plt.subplots(figsize=(9, 4.7))
    _draw_scene(ax, flow, result["frames"][-1], _vis_vmax(result),
                _r_max(meta["scenario"]), show_field, wind, meta["dt"],
                meta["scenario"])
    return _png(fig)


def scene_gif(result, show_field=True, wind="off", duration=120):
    """Looping GIF of the cloud scene over the collected frames."""
    meta = result["meta"]
    flow = flow_proxy(meta)
    r_max = _r_max(meta["scenario"])
    imgs = []
    for fr in result["frames"]:
        fig, ax = plt.subplots(figsize=(9, 4.7))
        _draw_scene(ax, flow, fr, result["qc_max"], r_max, show_field, wind,
                    meta["dt"], meta["scenario"])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=92)   # no bbox → identical frame size
        plt.close(fig)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB"))
    out = io.BytesIO()
    imgs[0].save(out, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0, optimize=True)
    return out.getvalue()


def climate_gif(res, seed_on, duration=120):
    """Looping GIF of the stratocumulus deck evolving — the cloud-water field
    with droplets, seeded drops ringed in magenta. Takes the raw run dict (it is
    built inside the cached climate wrapper, where the real Flow2D is available).
    """
    flow, frames = res["flow"], res["frames"]
    if len(frames) < 2:
        return None
    qmax = max(0.3, float(np.percentile([f["qc"].max() for f in frames], 90)))
    imgs = []
    for k, fr in enumerate(frames):
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        if seed_on:
            draw_frame_seeded(ax, flow, fr, qmax, r_max=60.0)
        else:
            draw_frame(ax, flow, fr, "qc", vmax=qmax, r_max=60.0)
        ax.set_title(f"stratocumulus deck   frame {k + 1}/{len(frames)}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=92)
        plt.close(fig)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB"))
    out = io.BytesIO()
    imgs[0].save(out, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0, optimize=True)
    return out.getvalue()


# --- mixed-phase liquid/ice two-panel -------------------------------------- #
def phase_image(result):
    """Final-frame liquid (q_c) vs ice (q_i) fields with their super-droplets."""
    meta = result["meta"]
    flow = flow_proxy(meta)
    fr = result["frames"]
    f = fr[-1]
    xe = np.linspace(0, flow.X, flow.Nx + 1)
    ze = np.linspace(0, flow.Z, flow.Nz + 1)
    lmax = max(0.3, float(np.percentile([x["q_liquid"].max() for x in fr], 95)))
    imax = max(0.1, float(np.percentile([x["q_ice"].max() for x in fr], 98)))
    ph = f["phase"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.0))
    for ax, fld, cm, vm, ttl, mask in (
            (axL, f["q_liquid"], "Blues", lmax, "liquid  q$_c$ (g/kg)", ph == 0),
            (axR, f["q_ice"], "BuPu", imax, "ice  q$_i$ (g/kg)", ph == 1)):
        ax.pcolormesh(xe, ze, fld.T, cmap=cm, vmin=0, vmax=vm, shading="flat")
        s = np.flatnonzero(mask & (f["r_um"] > 3.0))
        if s.size > 4000:
            s = s[np.linspace(0, s.size - 1, 4000).astype(int)]
        ax.scatter(f["x"][s], f["z"][s], s=2.5, c="0.15", alpha=0.25,
                   edgecolors="none")
        ax.set_xlim(0, flow.X); ax.set_ylim(0, flow.Z)
        ax.set_title(ttl); ax.set_xlabel("x (m)")
    axL.set_ylabel("z (m)")
    fig.suptitle(f"mixed-phase: liquid vs ice   t = {f['step'] * meta['dt']:.0f} s")
    fig.tight_layout()
    return _png(fig)


def _cloud_top(f, dz):
    """Highest level z (m) whose domain column-max condensate (q_c [+ q_ice])
    exceeds 0.01 g/kg — the cloud top. 0 if no cloud."""
    cond = f["qc"] + (f["q_ice"] if "q_ice" in f else 0.0)   # (Nx, Nz)
    colmax = cond.max(axis=0)                                 # over x → (Nz,)
    lev = np.where(colmax > 0.01)[0]
    return float((lev.max() + 0.5) * dz) if lev.size else 0.0


def _twod_panels(result):
    """Regime-aware (title, traces) panels for the 2-D mechanism time series, all
    from frame keys vs t = step·dt. SHARED by the static plotly chart and the
    synced animation, so both show identical series.

      ALWAYS  — cloud water LWP (∝ Σ q_c), cumulative surface precip, max updraft.
      ICE     — IWP (∝ Σ q_ice) on the SAME panel as LWP (the LWP↓/IWP↑ glaciation
                / Wegener–Bergeron–Findeisen crossover).
      DEEP/anelastic — cloud-top height. LIGHTNING — cumulative flashes.
      WARM (no ice)  — N_d (Σ A of cloudy SDs) and mean droplet radius.

    Returns (t, panels) with panels = [(title, [(name, y, color), ...]), ...].
    """
    meta = result["meta"]
    fr = result["frames"]
    dt = meta["dt"]
    has_ice = "q_ice" in fr[0]
    t = [float(f["step"]) * dt for f in fr]

    # panels are (title, traces, twin); twin=True draws trace[1] on a 2nd y-axis.
    lwp = [float(f["qc"].sum()) for f in fr]
    panels = []
    if has_ice:
        iwp = [float(f["q_ice"].sum()) for f in fr]
        panels.append(("Water path: Liquid vs ice (∝ Σ g/kg)", [
            ("liquid q_c", lwp, "#2D6BE0"), ("ice q_i", iwp, "#0FB5C4")], False))
    else:
        panels.append(("Cloud water LWP (∝ Σ q_c, g/kg)",
                       [("LWP", lwp, "#2D6BE0")], False))
    panels.append(("Surface precip (kg, cumulative)",
                   [("precip", [float(f.get("surf_precip", 0.0)) for f in fr],
                     "#E8743B")], False))
    panels.append(("Max updraft w (m/s)",
                   [("w_max", [float(f["w"].max()) for f in fr], "#46566B")], False))
    if meta.get("anelastic") or meta.get("scenario") == "deep_convection":
        panels.append(("Cloud-top height (m)",
                       [("z_top", [_cloud_top(f, meta["dz"]) for f in fr],
                         "#6B5BD2")], False))
    if meta.get("electrification"):
        cflash = list(np.cumsum([len(f.get("flashes", [])) for f in fr]).astype(float))
        panels.append(("Cumulative flash count", [("flashes", cflash, "#C0504D")],
                       False))
    if not has_ice:
        # in-cloud droplet number CONCENTRATION (cm^-3), not raw Σ A — and on a
        # twin axis with mean radius (the two have different scales).
        cell_cm3 = meta["dx"] * meta["dz"] * meta.get("depth", 1.0) * 1e6
        nd, rmean = [], []
        for f in fr:
            cloudy = f["r_um"] > 2.0
            ncell = max(1, int((f["qc"] > 0.01).sum()))         # cloudy cells
            nd.append(float(f["A"][cloudy].sum()) / (ncell * cell_cm3))
            rmean.append(float(f["r_um"][cloudy].mean()) if cloudy.any() else 0.0)
        panels.append(("Droplet number N_d & mean radius", [
            ("N_d (cm⁻³)", nd, "#2f9e44"), ("mean r (µm)", rmean, "#f59f00")], True))
    return t, panels


def twod_timeseries(result):
    """Static regime-aware time series of the mechanism (interactive plotly).
    Twin panels put their 2nd trace on a secondary y-axis."""
    t, panels = _twod_panels(result)
    n = len(panels)
    cols = 2
    rows = -(-n // cols)
    specs = [[None] * cols for _ in range(rows)]
    for i, (_t, _tr, twin) in enumerate(panels):
        specs[i // cols][i % cols] = {"secondary_y": bool(twin)}
    fig = make_subplots(rows=rows, cols=cols, specs=specs,
                        subplot_titles=[p[0] for p in panels])
    for i, (_title, traces, twin) in enumerate(panels):
        r, c = i // cols + 1, i % cols + 1
        multi = len(traces) > 1
        for j, (name, y, col) in enumerate(traces):
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name,
                                     line=dict(color=col), showlegend=multi),
                          row=r, col=c, secondary_y=bool(twin and j == 1))
    fig.update_xaxes(title_text="time (s)")
    fig.update_layout(height=250 * rows, margin=dict(t=44),
                      legend=dict(orientation="h", y=1.04))
    return fig


def scene_and_series_gif(result, show_field=True, wind="off", duration=150):
    """Combined looping animation: the cloud scene (left) and the regime
    time-series (right) GROWING IN SYNC, frame by frame — the graphs build up at
    the same time as the cloud, on fixed axes. Memoized on the run payload so
    repeat reruns are instant."""
    store = result.setdefault("_gifcache", {})
    ckey = (show_field, wind)
    if ckey in store:
        return store[ckey]
    meta = result["meta"]
    flow = flow_proxy(meta)
    r_max = _r_max(meta["scenario"])
    vmax = _vis_vmax(result)
    dt = meta["dt"]
    frames = result["frames"]
    t, panels = _twod_panels(result)
    n = len(panels)
    xr = (t[0], t[-1] if t[-1] > t[0] else t[0] + 1.0)
    yr = []                                   # per panel: (lo,hi) or [(lo,hi)/trace]
    for _title, traces, twin in panels:
        per = []
        for _n, ys, _c in traces:
            lo, hi = (min(ys), max(ys)) if ys else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
            pad = 0.06 * (hi - lo)
            per.append((lo - pad, hi + pad))
        yr.append(per if twin else
                  (min(p[0] for p in per), max(p[1] for p in per)))
    imgs = []
    for k, fr in enumerate(frames):
        fig = plt.figure(figsize=(12.2, max(4.7, 1.5 * n)))
        gs = fig.add_gridspec(n, 2, width_ratios=[1.7, 1.0], wspace=0.32, hspace=0.8)
        ax = fig.add_subplot(gs[:, 0])
        _draw_scene(ax, flow, fr, vmax, r_max, show_field, wind,
                    dt, meta["scenario"])
        for i, (title, traces, twin) in enumerate(panels):
            axp = fig.add_subplot(gs[i, 1])
            if twin and len(traces) == 2:
                (n1, y1, c1), (n2, y2, c2) = traces
                axp.plot(t[:k + 1], y1[:k + 1], color=c1, lw=1.7)
                axp.set_ylabel(n1, color=c1, fontsize=7)
                axp.tick_params(axis="y", labelcolor=c1, labelsize=7)
                axp.set_ylim(*yr[i][0])
                ax2 = axp.twinx()
                ax2.plot(t[:k + 1], y2[:k + 1], color=c2, lw=1.7)
                ax2.set_ylabel(n2, color=c2, fontsize=7)
                ax2.tick_params(axis="y", labelcolor=c2, labelsize=7)
                ax2.set_ylim(*yr[i][1])
            else:
                for name, ys, col in traces:
                    axp.plot(t[:k + 1], ys[:k + 1], color=col, lw=1.7, label=name)
                axp.set_ylim(*yr[i])
                if len(traces) > 1:
                    axp.legend(fontsize=6, loc="upper left")
            axp.set_xlim(*xr)
            axp.axvline(t[k], color="#AAB2C0", lw=0.7, ls=":")
            axp.set_title(title, fontsize=8)
            axp.tick_params(axis="x", labelsize=7)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        plt.close(fig)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB"))
    out = io.BytesIO()
    imgs[0].save(out, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0, optimize=True)
    store[ckey] = out.getvalue()
    return store[ckey]


def _climate_panels():
    """The climate-intervention metrics that actually matter for MCB (NOT q_c):
    cloud droplet number, cloud albedo, and the shortwave cloud radiative effect."""
    return [("Cloud droplet number N_d", "cm⁻³", "nc", "#2f9e44"),
            ("Cloud albedo", "", "albedo", "#2D6BE0"),
            ("Shortwave CRE", "W/m²", "cre", "#C0504D")]


def climate_timeseries(ts, ctrl=None):
    """Plotly of the MCB metrics over time — N_d, albedo, CRE. If ``ctrl`` (the
    unseeded twin's series) is given it is overlaid as a DOTTED baseline so the
    seeding effect is read directly."""
    panels = _climate_panels()
    fig = make_subplots(rows=len(panels), cols=1,
                        subplot_titles=[f"{t} ({u})" if u else t
                                        for t, u, _k, _c in panels])
    for i, (_title, _unit, k, col) in enumerate(panels):
        fig.add_trace(go.Scatter(x=ts["t"], y=ts[k], mode="lines",
                                 name="seeded" if ctrl else None,
                                 line=dict(color=col, width=2.4),
                                 showlegend=bool(ctrl) and i == 0), row=i + 1, col=1)
        if ctrl is not None:
            fig.add_trace(go.Scatter(x=ctrl["t"], y=ctrl[k], mode="lines",
                                     name="control (unseeded)",
                                     line=dict(color=col, width=1.8, dash="dot"),
                                     showlegend=i == 0), row=i + 1, col=1)
    fig.update_xaxes(title_text="time (s)")
    fig.update_layout(height=200 * len(panels), margin=dict(t=40),
                      legend=dict(orientation="h", y=1.06))
    return fig


def _draw_climate_scene(ax, flow, frame, vmax, seed_on):
    if seed_on:
        draw_frame_seeded(ax, flow, frame, vmax, r_max=60.0)
    else:
        draw_frame(ax, flow, frame, "qc", vmax=vmax, r_max=60.0)


def climate_scene_series_gif(result, ctrl_ts=None, duration=160):
    """Combined looping animation for the climate deck: the stratocumulus scene
    (left) and the MCB metrics N_d / albedo / CRE (right) growing IN SYNC. If
    ``ctrl_ts`` is given, the unseeded twin is drawn as a dotted baseline so the
    seeding effect builds up visibly. Memoized on the run payload."""
    store = result.setdefault("_gifcache", {})
    ck = ("synced", ctrl_ts is not None)
    if ck in store:
        return store[ck]
    meta = result["meta"]
    flow = flow_proxy(meta)
    frames = result["frames"]
    ts = result["ts"]
    seed_on = meta.get("seed_on", False)
    panels = _climate_panels()
    vmax = max(0.3, float(np.percentile([f["qc"].max() for f in frames], 90)))
    t = ts["t"]
    xr = (t[0], t[-1] if t[-1] > t[0] else t[0] + 1.0)
    yr = []
    for _ti, _u, k, _c in panels:
        vals = list(ts[k]) + (list(ctrl_ts[k]) if ctrl_ts else [])
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        pad = 0.08 * (hi - lo)
        yr.append((lo - pad, hi + pad))
    n = len(panels)
    imgs = []
    for kk, fr in enumerate(frames):
        fig = plt.figure(figsize=(13.5, max(5.4, 1.95 * n)))
        gs = fig.add_gridspec(n, 2, width_ratios=[1.7, 1.0], wspace=0.32, hspace=0.7)
        ax = fig.add_subplot(gs[:, 0])
        _draw_climate_scene(ax, flow, fr, vmax, seed_on)
        ax.set_title(f"stratocumulus deck   t = {t[kk]:.0f} s", fontsize=9)
        for i, (title, unit, k, col) in enumerate(panels):
            axp = fig.add_subplot(gs[i, 1])
            axp.plot(t[:kk + 1], ts[k][:kk + 1], color=col, lw=1.8,
                     label="seeded" if ctrl_ts else None)
            if ctrl_ts is not None:
                axp.plot(ctrl_ts["t"][:kk + 1], ctrl_ts[k][:kk + 1], color=col,
                         lw=1.5, ls=":", label="control")
            axp.set_xlim(*xr)
            axp.set_ylim(*yr[i])
            axp.axvline(t[kk], color="#AAB2C0", lw=0.7, ls=":")
            axp.set_title(f"{title} ({unit})" if unit else title, fontsize=8)
            axp.tick_params(labelsize=7)
            if ctrl_ts is not None and i == 0:
                axp.legend(fontsize=6, loc="upper left")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB"))
    out = io.BytesIO()
    imgs[0].save(out, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0, optimize=True)
    store[ck] = out.getvalue()
    return store[ck]


def bergeron_figure(result):
    """Plotly: domain liquid- and ice-water paths over time (the WBF hand-off)."""
    meta = result["meta"]
    fr = result["frames"]
    t = np.array([f["step"] for f in fr]) * meta["dt"]
    lwp = np.array([f["q_liquid"].sum() for f in fr])
    iwp = np.array([f["q_ice"].sum() for f in fr])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=lwp, mode="lines", name="liquid (Σ q_c)",
                             line=dict(color="#2D6BE0", width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=iwp, mode="lines", name="ice (Σ q_i)",
                             line=dict(color="#0FB5C4", width=2.5)))
    fig.update_xaxes(title_text="time (s)")
    fig.update_yaxes(title_text="domain water (g/kg, summed)")
    fig.update_layout(height=380, margin=dict(t=30),
                      legend=dict(orientation="h", y=1.08))
    return fig


# --- ice-habit gallery (adapted from examples/_habit_still.py) ------------- #
def habit_image(result):
    """Two-panel: where each habit sits in the storm (coloured by aspect ratio φ)
    and a sorted gallery of the actual predicted crystal shapes."""
    meta = result["meta"]
    flow = flow_proxy(meta)
    f = result["frames"][-1]
    ph = f["phase"]
    ice = (ph == 1) & (f["a_axis"] > 0) & (f["c_axis"] > 0)
    cmap = plt.cm.coolwarm_r
    norm = plt.Normalize(0.5, 1.5)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.4),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    if ice.sum() == 0:
        for ax in (axL, axR):
            ax.set_facecolor(NIGHT); ax.axis("off")
        axL.text(0.5, 0.5, "no ice crystals yet —\nrun longer or a colder scenario",
                 color="#cfd8e6", ha="center", va="center", transform=axL.transAxes)
        fig.patch.set_facecolor(NIGHT)
        return _png(fig, facecolor=NIGHT)

    a = f["a_axis"][ice]; c = f["c_axis"][ice]; phi = f["phi"][ice]
    x = f["x"][ice]; z = f["z"][ice]

    axL.set_facecolor(NIGHT)
    ssz = 8 + 60 * np.clip(a / a.max(), 0, 1)
    sca = axL.scatter(x, z, c=phi, cmap=cmap, norm=norm, s=ssz, alpha=0.9,
                      edgecolor="none")
    axL.set_xlim(0, flow.X); axL.set_ylim(0, flow.Z)
    axL.set_xlabel("x (m)", color="#cfd8e6"); axL.set_ylabel("z (m)", color="#cfd8e6")
    axL.tick_params(colors="#9fb0c6")
    axL.set_title("ice habit across the storm\n(colour = aspect ratio, size = crystal size)",
                  color="#e8e8e8", fontsize=12)
    cb = fig.colorbar(sca, ax=axL, fraction=0.04, pad=0.02)
    cb.set_label(r"aspect ratio $\phi=c/a$  (plate $\leftarrow$ 1 $\rightarrow$ column)")

    axR.set_facecolor(NIGHT)
    order = np.argsort(phi)
    sel = order[np.linspace(0, len(order) - 1, min(48, len(order))).astype(int)]
    ncol = 8
    for k, i in enumerate(sel):
        gx, gy = k % ncol, k // ncol
        ar = c[i] / a[i]
        w = 0.4 / np.sqrt(ar); h = 0.4 * np.sqrt(ar)
        axR.add_patch(Ellipse((gx, -gy), w, h, facecolor=cmap(norm(phi[i])),
                              edgecolor="w", lw=0.4))
        axR.text(gx, -gy - 0.42, r"$\phi$=%.2f" % phi[i], ha="center", va="top",
                 color="#bbb", fontsize=6)
    axR.set_xlim(-0.6, ncol - 0.4); axR.set_ylim(-(len(sel) // ncol) - 0.7, 0.7)
    axR.set_aspect("equal"); axR.axis("off")
    axR.set_title("gallery of predicted crystal shapes\n(plate $\\to$ sphere $\\to$ column)",
                  color="#e8e8e8", fontsize=12)
    n_plate = int((phi < 0.8).sum()); n_col = int((phi > 1.25).sum())
    fig.suptitle(f"ice habit (Chen–Lamb) — {int(ice.sum())} crystals: "
                 f"{n_plate} plates, {n_col} columns", color="white", fontsize=12)
    fig.patch.set_facecolor(NIGHT)
    fig.tight_layout()
    return _png(fig, facecolor=NIGHT)


# --- electrification dark-sky view (reuses draw_storm_electric) ------------- #
def _best_flash_index(frames):
    best, bi = -1, len(frames) - 1
    for i, f in enumerate(frames):
        for fl in f.get("flashes", []):
            n = len(fl.get("segments", []))
            if n > best:
                best, bi = n, i
    return bi


def electric_image(result):
    """Dark-sky charge + lightning still on the frame with the strongest flash
    (or the last frame, showing the charge dipole, if no flash fired)."""
    meta = result["meta"]
    flow = flow_proxy(meta)
    frames = result["frames"]
    xe = np.linspace(0, flow.X, flow.Nx + 1)
    ze = np.linspace(0, flow.Z, flow.Nz + 1)
    vmax_cloud = max(0.6, np.percentile(
        [(f["qc"] + f.get("q_ice", 0.0)).max() for f in frames], 92))
    cds = [np.abs(f["charge_density"])[np.abs(f["charge_density"]) > 0]
           for f in frames
           if f.get("charge_density") is not None and (f["charge_density"] != 0).any()]
    lim_charge = float(np.percentile(np.concatenate(cds), 97)) if cds else 1e-12
    f = frames[_best_flash_index(frames)]
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    fig.patch.set_facecolor(NIGHT)
    draw_storm_electric(ax, flow, f, xe, ze, vmax_cloud, lim_charge,
                        from_gaussian=0.7)
    return _png(fig, facecolor=NIGHT)


# ======================================================================== #
# Parcel mode figures (logic mirrored from app_streamlit.py — pure builders)
# ======================================================================== #
def _series(out, key):
    ts = sorted(out)
    return np.array(ts), np.array([out[t][key] for t in ts])


def _dsd_stack(out):
    ts = sorted(out)
    radii = out[ts[0]]["dsd_r"] * 1e6
    stack = np.array([out[t]["dsd_n"] for t in ts])
    rv = np.array([out[t]["rv"] for t in ts])
    return np.array(ts), radii, stack, rv


def _viridis_at(frac):
    import plotly.colors as pc
    return pc.sample_colorscale(VIRIDIS, [float(np.clip(frac, 0, 1))])[0]


def parcel_timeseries(runs, dt):
    """3×2 panel: RH, vapour, height, temperature, mixing ratios, number concs."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Relative humidity RH (%)", "Vapour q<sub>v</sub> (g/kg)",
                        "Height z (m)", "Temperature T (K)",
                        "Mixing ratios q<sub>x</sub> (g/kg)",
                        "Number conc. n<sub>x</sub> (cm⁻³)"))
    multi = len(runs) > 1
    dashes = ["solid", "dash", "dot", "dashdot"]
    for j, (label, out, *_rest) in enumerate(runs):
        dash = dashes[j % len(dashes)]
        pre = f"{label} " if multi else ""
        ts, _ = _series(out, "RH")
        tsec = ts * dt

        def add(key, row, col, color, name, scale=1.0):
            _, y = _series(out, key)
            y = y * scale
            if row == 3 and col == 2:
                y = np.where(y > 0, y, np.nan)
            fig.add_trace(go.Scatter(
                x=tsec, y=y, mode="lines", name=pre + name,
                line=dict(color=None if multi else color, dash=dash),
                legendgroup=label, showlegend=(row == 3)), row=row, col=col)
        add("RH", 1, 1, "#5aa9e6", "RH", 100.0)
        add("qv", 1, 2, "#2f9e44", "q_v")
        add("z", 2, 1, "#0c1626", "z")
        add("T_K", 2, 2, "#e8743b", "T")
        add("qa", 3, 1, "#2d6be0", "q_a (aerosol)")
        add("qc", 3, 1, "#f59f00", "q_c (cloud)")
        add("qr", 3, 1, "#2f9e44", "q_r (rain)")
        add("NA", 3, 2, "#2d6be0", "N_a (aerosol)")
        add("NC", 3, 2, "#f59f00", "N_c (cloud)")
        add("NR", 3, 2, "#2f9e44", "N_r (rain)")
    fig.update_yaxes(type="log", row=3, col=2)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_layout(height=820, legend=dict(orientation="h", y=-0.08),
                      margin=dict(t=40))
    return fig


def parcel_dsd_contour(out, dt):
    ts, radii, stack, rv = _dsd_stack(out)
    tsec = ts * dt
    z = stack.T.copy()
    z[z <= 0] = np.nan
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=tsec, y=radii, z=np.log10(z), colorscale=VIRIDIS,
        colorbar=dict(title="log₁₀ dN<br>(cm⁻³)"),
        hovertemplate="t=%{x:.0f}s<br>r=%{y:.2f}µm<br>log₁₀dN=%{z:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tsec, y=rv, mode="lines",
                             line=dict(color="#0c1626", width=2.5),
                             name="mean radius r̄ (µm)"))
    fig.update_yaxes(type="log", title_text="Radius r (µm)")
    fig.update_xaxes(title_text="Time (s)")
    fig.update_layout(height=520, title="DSD time evolution",
                      legend=dict(orientation="h", y=1.05))
    return fig


def parcel_dsd_spectra(out, dt):
    ts, radii, stack, _ = _dsd_stack(out)
    z = stack.copy()
    z[z <= 0] = np.nan
    n = len(ts)
    fig = go.Figure()
    idx = np.linspace(0, n - 1, min(12, n)).astype(int)
    for i in idx:
        frac = i / max(1, n - 1)
        fig.add_trace(go.Scatter(
            x=radii, y=z[i], mode="lines", line=dict(color=_viridis_at(frac)),
            name=f"t={ts[i] * dt:.0f}s", showlegend=False,
            hovertemplate="r=%{x:.2f}µm<br>dN=%{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale=VIRIDIS, cmin=0, cmax=ts[-1] * dt, color=[0],
                    colorbar=dict(title="Time (s)")), showlegend=False))
    fig.update_xaxes(type="log", title_text="Radius r (µm)")
    fig.update_yaxes(type="log", title_text="dN (cm⁻³)")
    fig.update_layout(height=520, title="DSD spectra coloured by time")
    return fig


def parcel_particles(M, A):
    m = A > 0
    r = np.zeros_like(M)
    r[m] = (M[m] / (A[m] * 4.0 / 3.0 * pi * rho_liq)) ** (1.0 / 3.0)
    r_um = r[m] * 1e6
    Aw = A[m]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if r_um.size:
        s = 4 + 16 * (np.log10(Aw + 1) / max(1.0, np.log10(Aw.max() + 1)))
        fig.add_trace(go.Scattergl(
            x=r_um, y=Aw, mode="markers",
            marker=dict(size=s, color=np.log10(Aw + 1), colorscale=VIRIDIS,
                        opacity=0.5, colorbar=dict(title="log₁₀ A")),
            name="super-droplets",
            hovertemplate="r=%{x:.3f}µm<br>A=%{y:.3g}<extra></extra>"),
            secondary_y=False)
        edges = np.logspace(np.log10(max(1e-3, r_um.min())),
                            np.log10(r_um.max() + 1e-9), 40)
        hist, e = np.histogram(r_um, bins=edges, weights=Aw)
        centers = np.sqrt(e[:-1] * e[1:])
        fig.add_trace(go.Bar(x=centers, y=hist, name="Σ A per bin", opacity=0.35,
                             marker_color="#5aa9e6"), secondary_y=True)
    fig.update_xaxes(type="log", title_text="Droplet radius r (µm)")
    fig.update_yaxes(type="log", title_text="multiplicity A", secondary_y=False)
    fig.update_yaxes(title_text="Σ A in bin", secondary_y=True)
    fig.update_layout(height=520, title="Final super-droplet distribution",
                      legend=dict(orientation="h", y=1.05))
    return fig


def parcel_profiles(runs):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Temperature T (°C) vs height",
                                        "LWC q_c+q_r (g/kg) vs height"))
    multi = len(runs) > 1
    dashes = ["solid", "dash", "dot", "dashdot"]
    for j, (label, out, *_rest) in enumerate(runs):
        dash = dashes[j % len(dashes)]
        _, z = _series(out, "z")
        _, T = _series(out, "T")
        _, qc = _series(out, "qc")
        _, qr = _series(out, "qr")
        nm = label if multi else None
        fig.add_trace(go.Scatter(x=T, y=z, mode="lines", line=dict(dash=dash),
                                 name=(nm or "T"), legendgroup=label,
                                 showlegend=multi), row=1, col=1)
        fig.add_trace(go.Scatter(x=qc + qr, y=z, mode="lines", line=dict(dash=dash),
                                 name=(nm or "LWC"), legendgroup=label,
                                 showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="Height z (m)", row=1, col=1)
    fig.update_xaxes(title_text="T (°C)", row=1, col=1)
    fig.update_xaxes(title_text="q_c + q_r (g/kg)", row=1, col=2)
    fig.update_layout(height=520)
    return fig
