"""Rendering for the 2D cumulus: a field background (cloud water / supersaturation)
with the Lagrangian super-droplets overlaid, coloured by radius. Used by the
animation script, the notebook, and the Streamlit page alike."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from droplab.parameters import p0, r_a, cp


_FIELDS = {
    "qc": dict(label="q$_c$ (g/kg)", cmap="Blues"),
    "q_liquid": dict(label="q$_{liquid}$ (g/kg)", cmap="Blues"),
    "q_ice": dict(label="q$_{ice}$ (g/kg)", cmap="BuPu"),
    "supersat": dict(label="supersaturation", cmap="RdBu_r"),
    "qv": dict(label="q$_v$ (kg/kg)", cmap="YlGn"),
    "theta": dict(label="θ (K)", cmap="RdYlBu_r"),
    "T": dict(label="T (K)", cmap="RdYlBu_r"),
}


def _draw_wind_overlay(ax, flow, frame, style="streamlines", quiver_perturb=True,
                       n_arrows=16):
    """Faint BACKGROUND hint of the circulation, drawn UNDER the droplets (low zorder)
    so the cloud stays the focus — this is a cloud model, not a wind model.

    Plots the PERTURBATION wind u' = u - Ū(z) (subtract the horizontal mean at each
    height — a Reynolds decomposition). A mean wind / shear can be ~50x the convective
    velocities and would swamp the field; the mean shear stays legible from the cloud's
    tilt. With no mean wind u' == u, so this is a no-op for those cases. The perturbation
    also has a huge vertical dynamic range (near the rigid surface w'->0 so inflow is
    strong & horizontal, the free troposphere is quiescent), which is why arrows are
    normalised to unit length rather than scaled by magnitude."""
    if "u" not in frame or "w" not in frame:
        return
    u, w = frame["u"], frame["w"]
    if quiver_perturb:
        u = u - u.mean(axis=0, keepdims=True)
        w = w - w.mean(axis=0, keepdims=True)
    xc = (np.arange(flow.Nx) + 0.5) * flow.dx
    zc = (np.arange(flow.Nz) + 0.5) * flow.dz
    if style == "streamlines":
        # thin grey streamlines trace the overturning cells; kept faint and low
        ax.streamplot(xc, zc, u.T, w.T, color="0.45", density=0.9, linewidth=0.6,
                      arrowsize=0.6, zorder=0.5)
    else:  # "arrows"
        sx = max(1, flow.Nx // n_arrows)
        sz = max(1, flow.Nz // n_arrows)
        XX, ZZ = np.meshgrid(xc[::sx], zc[::sz], indexing="ij")
        U, W = u[::sx, ::sz], w[::sx, ::sz]
        spd = np.hypot(U, W)
        ax.quiver(XX, ZZ, U / (spd + 1e-9), W / (spd + 1e-9), color="0.45",
                  scale=36, width=0.0026, pivot="mid", alpha=0.5, zorder=0.5)


def draw_frame(ax, flow, frame, field="qc", vmax=None, r_max=50.0,
               max_dots=7000, r_show=2.0, show_aerosol=True, quiver=False,
               n_arrows=16, quiver_perturb=True, quiver_style="streamlines",
               drop_cmap="viridis", vmin=None):
    """Render one snapshot onto a (cleared) axis. Returns the scatter handle.

    Unactivated aerosol/haze (r <= r_show µm) is drawn as faint black dots so the
    aerosol-laden air is visible; activated droplets (r > r_show) use the `viridis`
    colormap (perceptually uniform and colour-vision-safe) and grow in size with
    radius so rain stands out. Both populations are subsampled for legibility.

    quiver=True overlays a faint BACKGROUND hint of the circulation (drawn under the
    droplets); quiver_style is "streamlines" (default) or "arrows". Zero sim cost —
    u, w are already computed."""
    ax.clear()
    xe = np.linspace(0, flow.X, flow.Nx + 1)
    ze = np.linspace(0, flow.Z, flow.Nz + 1)
    fld = frame[field]
    spec = _FIELDS[field]
    if field == "supersat":
        lim = vmax or 0.01
        ax.pcolormesh(xe, ze, fld.T, cmap=spec["cmap"], vmin=-lim, vmax=lim,
                      shading="flat", zorder=0)
    elif vmin is not None:
        # background fields with a non-zero floor (T, theta): explicit vmin/vmax
        ax.pcolormesh(xe, ze, fld.T, cmap=spec["cmap"], vmin=vmin,
                      vmax=(vmax if vmax is not None else fld.max()),
                      shading="flat", zorder=0)
    else:
        ax.pcolormesh(xe, ze, fld.T, cmap=spec["cmap"], vmin=0.0,
                      vmax=(vmax or max(0.5, fld.max())), shading="flat", zorder=0)

    # wind overlay sits BETWEEN the field and the droplets (faint, background)
    if quiver:
        _draw_wind_overlay(ax, flow, frame, style=quiver_style,
                           quiver_perturb=quiver_perturb, n_arrows=n_arrows)

    rr = frame["r_um"]
    # aerosol / haze as faint black dots
    if show_aerosol:
        a = np.flatnonzero(rr <= r_show)
        if a.size > max_dots:
            a = a[np.linspace(0, a.size - 1, max_dots).astype(int)]
        ax.scatter(frame["x"][a], frame["z"][a], c="black", s=1.8,
                   alpha=0.22, edgecolors="none", zorder=1.5)
    # activated droplets, viridis (perceptually-uniform, colour-vision-safe), sized by radius
    idx = np.flatnonzero(rr > r_show)
    if idx.size > max_dots:
        idx = idx[np.linspace(0, idx.size - 1, max_dots).astype(int)]
    rsel = rr[idx]
    sizes = 3.0 + 32.0 * np.clip(rsel / r_max, 0.0, 1.0) ** 2   # big = rain
    sc = ax.scatter(frame["x"][idx], frame["z"][idx], c=rsel, cmap=drop_cmap,
                    s=sizes, vmin=r_show, vmax=r_max, edgecolors="none", alpha=0.85,
                    zorder=2)
    # lightning: draw each branched dielectric-breakdown channel recorded since the
    # previous frame (segments = (n,4) edges), with a glow under a bright core
    from matplotlib.collections import LineCollection
    for fl in frame.get("flashes", []):
        seg = fl.get("segments")
        if seg is None or len(seg) == 0:
            continue
        segs = seg.reshape(-1, 2, 2)
        # electric blue-white (NOT yellow -- yellow collides with the viridis rain drops)
        for lw, col, a in [(5.0, "#ff8c00", 0.35), (3.2, "#ffb347", 0.95), (1.4, "#fff1d6", 1.0)]:
            ax.add_collection(LineCollection(segs, colors=col, linewidths=lw, alpha=a,
                                             zorder=4, capstyle="round"))
    ax.set_xlim(0, flow.X)
    ax.set_ylim(0, flow.Z)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"step {frame['step']}")
    return sc


def draw_frame_seeded(ax, flow, frame, vmax=None, r_max=50.0, max_dots=7000,
                      seed_color="magenta"):
    """Like draw_frame, but OVERLAYS the seeded (tag>0) super-droplets in a bright
    colour so the injected aerosol is visible to the eye — you watch it appear at
    the injection, spread, and (MCB) stay as many small drops or (GCCN) grow into
    drizzle."""
    sc = draw_frame(ax, flow, frame, "qc", vmax, r_max, max_dots)
    tag = frame.get("tag")
    if tag is not None:
        s = np.flatnonzero(tag > 0)
        if s.size > max_dots:
            s = s[np.linspace(0, s.size - 1, max_dots).astype(int)]
        if s.size:
            rs = np.clip(frame["r_um"][s], 0.5, r_max)
            ax.scatter(frame["x"][s], frame["z"][s], s=3.0 + 30.0 * (rs / r_max) ** 2,
                       facecolors="none", edgecolors=seed_color, linewidths=0.6,
                       alpha=0.9, label="seeded")
    return sc


def animate_seeding_compare(base, seeded, fps=8, r_max=60.0, dt=1.0,
                            metric="precip", title="aerosol seeding"):
    """Side-by-side animation: unseeded (left) vs seeded (right, injected droplets
    ringed in magenta), with a time-series below tracking the difference the seeding
    makes. metric='precip' plots cumulative surface precipitation (the GCCN /
    drizzle response); metric='albedo' plots the domain-mean cloud albedo (Twomey)."""
    flow = seeded["flow"]
    fb, fs = base["frames"], seeded["frames"]
    n = min(len(fb), len(fs))
    steps = np.array([fs[k]["step"] for k in range(n)]) * dt
    qmax = np.percentile([f["qc"].max() for f in fs[:n]], 90)

    if metric == "albedo":
        from droplab.climate_diag import optics_from_frame
        yb = np.array([optics_from_frame(fb[k], flow)["albedo_mean"] for k in range(n)])
        ys = np.array([optics_from_frame(fs[k], flow)["albedo_mean"] for k in range(n)])
        ylabel = "cloud albedo"
    else:
        yb = np.array([fb[k].get("surf_precip", 0.0) for k in range(n)])
        ys = np.array([fs[k].get("surf_precip", 0.0) for k in range(n)])
        ylabel = "cumulative surface precip [kg]"

    fig = plt.figure(figsize=(12, 7))
    axL = fig.add_subplot(2, 2, 1)
    axR = fig.add_subplot(2, 2, 2)
    axT = fig.add_subplot(2, 1, 2)
    ymax = max(yb.max(), ys.max()) * 1.1 + 1e-12

    def update(k):
        draw_frame(axL, flow, fb[k], "qc", qmax, r_max); axL.set_title("unseeded")
        draw_frame_seeded(axR, flow, fs[k], qmax, r_max); axR.set_title(f"seeded — {title}")
        axT.clear()
        axT.plot(steps[:k + 1], yb[:k + 1], "C0", label="unseeded")
        axT.plot(steps[:k + 1], ys[:k + 1], "C3", label="seeded")
        axT.set_xlim(steps[0], steps[n - 1]); axT.set_ylim(0, ymax)
        axT.set_xlabel("time [s]"); axT.set_ylabel(ylabel)
        axT.legend(loc="upper left", fontsize=8)

    update(0)
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    return fig, anim


def animate_mcb(base, seeded, t_inject, dt=1.0, fps=8, r_max=40.0):
    """MCB animation: the seeded stratocumulus run (sea-salt droplets ringed) beside
    live cloud-albedo curves for the UNSEEDED and SEEDED decks and the ACCUMULATED
    top-of-atmosphere cooling computed from their difference.

    The baseline matters: a stratocumulus deck dims on its own as it drizzles, so the
    seeded run's albedo alone would fall and hide the intervention. Only the gap
    between the seeded and the same-meteorology unseeded deck isolates the Twomey
    brightening — that gap drives the accumulated cooling."""
    from droplab.climate_diag import optics_from_frame, toa_forcing
    flow = seeded["flow"]
    fb, fs = base["frames"], seeded["frames"]
    n = min(len(fb), len(fs))
    steps = np.array([fs[k]["step"] for k in range(n)]) * dt
    ab = np.array([optics_from_frame(fb[k], flow)["albedo_mean"] for k in range(n)])
    as_ = np.array([optics_from_frame(fs[k], flow)["albedo_mean"] for k in range(n)])
    dF = toa_forcing(as_ - ab)                        # W/m^2 vs same-meteorology deck
    frame_dt = (steps[1] - steps[0]) if n > 1 else dt
    cool = np.cumsum(-dF) * frame_dt / 1e6            # accumulated cooling [MJ/m^2]
    qmax = np.percentile([fs[k]["qc"].max() for k in range(n)], 90)
    amax = max(ab.max(), as_.max()) * 1.15

    fig = plt.figure(figsize=(12, 5.5))
    axS = fig.add_subplot(1, 2, 1)
    axA = fig.add_subplot(2, 2, 2)
    axF = fig.add_subplot(2, 2, 4)

    def update(k):
        draw_frame_seeded(axS, flow, fs[k], qmax, r_max)
        axS.set_title(f"MCB-seeded stratocumulus  (t={steps[k]:.0f} s)")
        axA.clear()
        axA.plot(steps[:k + 1], ab[:k + 1], "C0", label="unseeded")
        axA.plot(steps[:k + 1], as_[:k + 1], "C3", label="seeded")
        axA.axvline(t_inject, color="0.5", ls="--", lw=1)
        axA.set_xlim(steps[0], steps[-1]); axA.set_ylim(0, max(amax, 0.05))
        axA.set_ylabel("cloud albedo"); axA.legend(loc="upper left", fontsize=7)
        axF.clear()
        axF.plot(steps[:k + 1], cool[:k + 1], "C2")
        axF.axvline(t_inject, color="0.5", ls="--", lw=1)
        axF.set_xlim(steps[0], steps[-1])
        axF.set_ylim(min(cool.min() * 1.1, 0), max(cool.max() * 1.1, 1e-6))
        axF.set_xlabel("time [s]"); axF.set_ylabel("accumulated cooling\n[MJ/m²]")

    update(0)
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    return fig, anim


def _field_panel(ax, flow, data, title, cmap, vmin=None, vmax=None, sym=False):
    xe = np.linspace(0, flow.X, flow.Nx + 1)
    ze = np.linspace(0, flow.Z, flow.Nz + 1)
    if sym:
        lim = vmax or np.abs(data).max() or 1e-9
        im = ax.pcolormesh(xe, ze, data.T, cmap=cmap, vmin=-lim, vmax=lim, shading="flat")
    else:
        im = ax.pcolormesh(xe, ze, data.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    return im


def draw_panels(axs, flow, frame, P_col, ranges=None):
    """Four panels: cloud scene (q_c + droplets), vertical velocity w, temperature
    T, water vapour q_v. w is taken from the frame if present (dynamic) else from
    the static flow (kinematic)."""
    r = ranges or {}
    for ax in axs:
        ax.clear()
    draw_panels._sc = draw_frame(axs[0], flow, frame, field="qc",
                                 vmax=r.get("qc"), r_max=r.get("r_max", 50.0))
    axs[0].set_title(f"cloud (step {frame['step']})", fontsize=10)
    w = frame.get("w")
    if w is None:
        w = flow.cell_velocities()[1]
    _field_panel(axs[1], flow, w, "vertical velocity w (m/s)", "RdBu_r",
                 sym=True, vmax=r.get("w"))
    T = frame["theta"] * (P_col[None, :] / p0) ** (r_a / cp)
    _field_panel(axs[2], flow, T - 273.15, "temperature T (°C)", "inferno",
                 vmin=r.get("Tmin"), vmax=r.get("Tmax"))
    _field_panel(axs[3], flow, frame["qv"] * 1e3, "water vapour q_v (g/kg)",
                 "YlGnBu", vmin=0.0, vmax=r.get("qv"))


def animate_panels(result, fps=10, r_max=50.0):
    """2x2 animation: cloud scene + w + T + q_v."""
    flow, frames, P_col = result["flow"], result["frames"], result["P_col"]
    ranges = dict(
        r_max=r_max,
        qc=np.percentile([f["qc"].max() for f in frames], 90),
        w=np.percentile([np.abs(f.get("w", flow.cell_velocities()[1])).max()
                         for f in frames], 95),
        qv=np.percentile([f["qv"].max() for f in frames], 98) * 1e3,
    )
    Tall = np.concatenate([(f["theta"] * (P_col[None, :] / p0) ** (r_a / cp) - 273.15).ravel()
                           for f in frames[::max(1, len(frames) // 5)]])
    ranges["Tmin"], ranges["Tmax"] = np.percentile(Tall, [2, 98])

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    axs = axs.ravel()
    draw_panels(axs, flow, frames[0], P_col, ranges)
    fig.tight_layout()

    def update(k):
        draw_panels(axs, flow, frames[k], P_col, ranges)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / fps, blit=False)
    return fig, anim


def draw_storm_electric(ax, flow, frame, xe, ze, vmax_cloud, lim_charge,
                        from_gaussian=None):
    """Dark-sky electrification view: condensate shroud + charge dipole (red +/blue -) +
    the branched dielectric-breakdown discharge channels (bright white core). Designed so
    the LIGHTNING reads clearly, unlike the droplet-scatter view where bright rain drops
    and the bolt share the viridis yellow. `from_gaussian` is an optional smoothing sigma."""
    from matplotlib.collections import LineCollection
    ax.clear()
    ax.set_facecolor("#070b16")
    cloud = frame["qc"] + frame.get("q_ice", 0.0)
    if from_gaussian is not None:
        from scipy.ndimage import gaussian_filter
        cloud = gaussian_filter(cloud, from_gaussian)
    cloud = np.where(cloud < 0.02, np.nan, cloud)
    ax.pcolormesh(xe, ze, cloud.T, cmap="bone", vmin=0.0, vmax=vmax_cloud,
                  shading="flat", alpha=0.8, zorder=1)
    cd = frame.get("charge_density")
    if cd is not None and (cd != 0).any():
        ax.pcolormesh(xe, ze, np.where(cd == 0, np.nan, cd).T, cmap="bwr",
                      vmin=-lim_charge, vmax=lim_charge, shading="flat", alpha=0.6, zorder=2)
    for fl in frame.get("flashes", []):
        seg = fl.get("segments")
        if seg is None or len(seg) == 0:
            continue
        segs = seg.reshape(-1, 2, 2)
        # electric blue-white glow + white core (avoid yellow: collides with viridis)
        ax.add_collection(LineCollection(segs, colors="#ff8c00", linewidths=5.0, alpha=0.40, zorder=5))
        ax.add_collection(LineCollection(segs, colors="#ffb347", linewidths=3.0, alpha=0.95, zorder=5.1))
        ax.add_collection(LineCollection(segs, colors="#fff1d6", linewidths=1.2, alpha=1.0, zorder=5.2))
    ax.set_xlim(0, flow.X)
    ax.set_ylim(0, flow.Z)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    nfl = len(frame.get("flashes", []))
    ax.set_title(f"step {frame['step']}" + (f"  —  ⚡ {nfl}" if nfl else ""), color="#dddddd")


def animate_electric(result, fps=12, smooth=0.7):
    """FuncAnimation of the dark-sky electrification view (condensate + charge + lightning),
    with colour scales fixed across the run. Use for electrification=True runs."""
    flow, frames = result["flow"], result["frames"]
    xe = np.linspace(0, flow.X, flow.Nx + 1)
    ze = np.linspace(0, flow.Z, flow.Nz + 1)
    vmax_cloud = max(0.6, np.percentile([(f["qc"] + f.get("q_ice", 0.0)).max() for f in frames], 92))
    cds = [np.abs(f["charge_density"])[np.abs(f["charge_density"]) > 0] for f in frames
           if f.get("charge_density") is not None and (f["charge_density"] != 0).any()]
    lim_charge = np.percentile(np.concatenate(cds), 97) if cds else 1e-12
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    fig.patch.set_facecolor("#070b16")

    def update(k):
        draw_storm_electric(ax, flow, frames[k], xe, ze, vmax_cloud, lim_charge, smooth)

    update(0)
    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / fps, blit=False)
    return fig, anim


def animate(result, field="qc", fps=10, r_max=50.0):
    """Build a matplotlib FuncAnimation over the captured frames."""
    flow = result["flow"]
    frames = result["frames"]
    vmax = (np.percentile([f[field].max() for f in frames], 90)
            if field != "supersat" else
            np.percentile([np.abs(f[field]).max() for f in frames], 90))

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    sc = draw_frame(ax, flow, frames[0], field, vmax, r_max)
    cb = fig.colorbar(sc, ax=ax, label="droplet radius (µm)")

    def update(k):
        nonlocal cb
        draw_frame(ax, flow, frames[k], field, vmax, r_max)

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / fps, blit=False)
    return fig, anim


def animate_mixed_phase(out, dt=1.0, fps=8, inject_window=None):
    """Two-panel animation of a mixed-phase run: liquid q_c (left) and ice q_i
    (right) fields with their super-droplets, so glaciation is visible in space —
    the liquid deck and a growing ice region (e.g. an INP-seeding patch) side by
    side. Requires an ice=True run (frames carry q_liquid/q_ice/phase).
    inject_window=(t0,t1) annotates the title while INP is being injected."""
    flow, fr = out["flow"], out["frames"]
    if "q_ice" not in fr[0]:
        raise ValueError("animate_mixed_phase needs an ice=True run (q_ice frames)")
    xe = np.linspace(0, flow.X, flow.Nx + 1)
    ze = np.linspace(0, flow.Z, flow.Nz + 1)
    lmax = max(0.3, float(np.percentile([f["q_liquid"].max() for f in fr], 95)))
    imax = max(0.3, float(np.percentile([f["q_ice"].max() for f in fr], 98)))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 3.6))

    def upd(k):
        f = fr[k]; ph = f["phase"]
        for ax, fld, cm, vm, ttl, mask in (
                (axL, f["q_liquid"], "Blues", lmax, "liquid  q$_c$ (g/kg)", ph == 0),
                (axR, f["q_ice"], "BuPu", imax, "ice  q$_i$ (g/kg)", ph == 1)):
            ax.clear()
            ax.pcolormesh(xe, ze, fld.T, cmap=cm, vmin=0, vmax=vm, shading="flat")
            s = np.flatnonzero(mask & (f["r_um"] > 3.0))
            if s.size > 4000:
                s = s[np.linspace(0, s.size - 1, 4000).astype(int)]
            ax.scatter(f["x"][s], f["z"][s], s=2.5, c="0.15", alpha=0.25, edgecolors="none")
            ax.set_xlim(0, flow.X); ax.set_ylim(0, flow.Z)
            ax.set_title(ttl); ax.set_xlabel("x (m)")
        axL.set_ylabel("z (m)")
        t = f["step"] * dt
        note = "   <- INP injected" if inject_window and inject_window[0] <= t <= inject_window[1] else ""
        fig.suptitle("mixed-phase: liquid vs ice   t=%4.0f s%s" % (t, note))

    upd(0); fig.tight_layout()
    anim = animation.FuncAnimation(fig, upd, frames=len(fr), interval=1000 / fps, blit=False)
    return fig, anim


def animate_droplets_phase(out, dt=1.0, fps=8, r_show=2.0, r_max=60.0, max_dots=9000):
    """Droplet-only animation (NO q_c/q_i field contour): liquid super-droplets are
    round dots coloured by radius with the standard `viridis` map, and ICE
    super-droplets are drawn as small 6-armed snowflake markers, so you literally
    watch drops freeze in place and fall. Needs an ice=True run (carries `phase`)."""
    flow, fr = out["flow"], out["frames"]
    if "phase" not in fr[0]:
        raise ValueError("animate_droplets_phase needs an ice=True run (phase frames)")
    fig, ax = plt.subplots(figsize=(9, 4.2))

    def upd(k):
        f = fr[k]; ph = f["phase"]; rr = f["r_um"]
        ax.clear()
        ax.set_facecolor("#eef2f7")                       # light sky (viridis reads on it)
        liq = np.flatnonzero((ph == 0) & (rr > r_show))
        ice = np.flatnonzero(ph == 1)
        if liq.size > max_dots:
            liq = liq[np.linspace(0, liq.size - 1, max_dots).astype(int)]
        ax.scatter(f["x"][liq], f["z"][liq], s=3.0 + 30.0 * np.clip(rr[liq] / r_max, 0, 1) ** 2,
                   c=rr[liq], cmap="viridis", vmin=r_show, vmax=r_max, edgecolors="none", alpha=0.85)
        if ice.size > max_dots:
            ice = ice[np.linspace(0, ice.size - 1, max_dots).astype(int)]
        ax.scatter(f["x"][ice], f["z"][ice], marker=(6, 2, 0), c="#1f4e8c",
                   s=6.0 + 14.0 * np.clip(rr[ice] / r_max, 0, 1) ** 2,
                   linewidths=0.5, alpha=0.9)              # smaller 6-armed snowflakes
        ax.set_xlim(0, flow.X); ax.set_ylim(0, flow.Z)
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.set_title("liquid (viridis dots) vs ice (snowflakes)   t=%4.0f s   [ice SDs: %d]"
                     % (f["step"] * dt, int((ph == 1).sum())))

    upd(0); fig.tight_layout()
    anim = animation.FuncAnimation(fig, upd, frames=len(fr), interval=1000 / fps, blit=False)
    return fig, anim
