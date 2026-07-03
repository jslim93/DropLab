"""Lecture figure: cloud electrification in a simulated deep convective cell.

Resolves the causal chain of the non-inductive charging mechanism at the moment of
breakdown: (a) the charge density structure that rebounding graupel-crystal collisions
produce, (b) the electrostatic field it sets up via Gauss's law, (c) the dielectric-
breakdown discharge, and (d) the time series of the domain-maximum field against the
breakdown threshold. Intended for a university atmospheric-science course.

    python -m examples.lightning_lecture

Physical basis and its parameters (each within its measured range; stated, not hidden):
  - per-collision charge  delta_q = 5 fC   (Takahashi 1978; Saunders & Peck 1998: 1-100 fC)
  - rebound/charge-separating fraction  eps = 0.3   (laboratory 0.1-0.5)
  - charge-reversal temperature  T_rev = -10 C   (Saunders single-reversal dipole)
Caveat (stated on the figure): the field is solved in a 2-D grounded box, where its
ABSOLUTE magnitude is ~100x below a 3-D storm; the breakdown threshold is therefore
ILLUSTRATIVE (scaled to the 2-D field). The charging MECHANISM is physical; only the
absolute trigger level is geometry-limited. Full assessment: docs/ELECTRIFICATION_AUDIT.md.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter

from examples.cloud_cases import CASES
from droplab.parameters import p0, r_a, cp
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.flow2d_viz import animate_electric

FIG = os.path.expanduser("~/Desktop/droplab_lightning_lecture.png")
GIF = os.path.expanduser("~/Desktop/droplab_lightning.gif")
E_BREAKDOWN = 400.0           # illustrative 2-D trigger, set BELOW the active-phase field so it
                              # is hit repeatedly -> the charge-up / discharge sawtooth is visible


def main():
    cfg = dict(CASES["deep_convection"])
    # higher ice super-droplet resolution -> graupel populates many cells -> the charge
    # structure is spatially extended (n_super=42k gives only ~2-3 graupel SDs -> a 2-cell
    # artifact; 160k gives ~17 graupel SDs -> ~800 charged cells).
    cfg.update(n_super=160000, nt=1400, collect_every=6, electrification=True,
               charge_eff=0.3, E_breakdown=E_BREAKDOWN)
    print("running anelastic cumulonimbus (lecture) ...")
    o = run_flow2d_dynamic(**cfg)
    flow, frames = o["flow"], o["frames"]
    dt = o["dt"]
    hist = o["efield_history"]                         # (step, max|E|, n_flash) per step
    t_hist = hist[:, 0] * dt / 60.0
    E_hist = hist[:, 1]
    flash_t = t_hist[hist[:, 2] > 0]
    flash_E = E_hist[hist[:, 2] > 0]
    print(f"frames={len(frames)} flashes={int((hist[:,2]>0).sum())} charge residual="
          f"{float(o['charge'].sum()) + float(o['charge_to_ground']):.1e}")
    if flash_t.size == 0:
        print("no flashes -- lower E_breakdown"); return

    # the storm charges in one brief glaciation pulse, so instead of a time sequence we show
    # the CAUSAL CHAIN at the discharge moment: charge separates -> field grows -> PAK!
    totq = np.array([np.abs(f["charge"]).sum() for f in frames])
    nfl = np.array([len(f.get("flashes", [])) for f in frames])
    flash_idx = np.flatnonzero(nfl > 0)
    peak = max(flash_idx, key=lambda i: totq[i])      # discharge frame with the most charge
    fp = frames[peak]

    xe = np.linspace(0, flow.X, flow.Nx + 1)
    ze = np.linspace(0, flow.Z, flow.Nz + 1)
    xcen = 0.5 * (xe[:-1] + xe[1:]); zcen = 0.5 * (ze[:-1] + ze[1:])
    cloud = gaussian_filter(fp["qc"] + fp.get("q_ice", 0.0), 0.7)
    cloud_m = np.where(cloud < 0.02, np.nan, cloud)
    cd = fp["charge_density"]
    lim = np.percentile(np.abs(cd[cd != 0]), 97) if (cd != 0).any() else 1e-12
    from droplab import electrification as el
    phi = el.solve_potential(cd, flow.dx, flow.dz, True)
    _, _, Emag = el.efield(phi, flow.dx, flow.dz, True)
    # zoom to the charged region
    ii, jj = np.where(cd != 0)
    x0, x1 = max(xcen[ii].min() - 2500, 0), min(xcen[ii].max() + 2500, flow.X)
    z0, z1 = max(zcen[jj].min() - 2500, 0), min(zcen[jj].max() + 2500, flow.Z)

    # charge-reversal isotherm T = q_rev_T (-10 C): graupel charges - below it, + above
    kap = r_a / cp
    T = fp["theta"] * (o["P_col"][None, :] / p0) ** kap     # (Nx, Nz) [K]

    def _bg(ax):
        ax.set_facecolor("#0a0e18")
        ax.pcolormesh(xe, ze, cloud_m.T, cmap="bone", vmin=0, vmax=max(0.6, np.nanmax(cloud_m)),
                      shading="flat", alpha=0.45, zorder=1)
        ax.set_xlim(x0, x1); ax.set_ylim(z0, z1)
        ax.tick_params(colors="#aaa", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#444")

    fig = plt.figure(figsize=(15, 9.4))
    fig.patch.set_facecolor("#0a0e18")
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.30, wspace=0.26)
    lab = dict(color="#e8e8e8", fontsize=11.5)

    # (a) charge density structure
    ax = fig.add_subplot(gs[0, 0]); _bg(ax)
    im = ax.pcolormesh(xe, ze, np.where(cd == 0, np.nan, cd).T * 1e9, cmap="bwr",
                       vmin=-lim * 1e9, vmax=lim * 1e9, shading="flat", zorder=2)
    cr = ax.contour(xcen, zcen, T.T, levels=[263.15], colors="#33e0e0", linewidths=1.0,
                    linestyles="--")
    ax.clabel(cr, fmt="-10°C", fontsize=7, colors="#33e0e0")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(
        r"$\rho_q$ (nC m$^{-3}$)", color="#ccc", fontsize=9)
    ax.set_title("(a)  non-inductive charge separation", **lab)
    ax.set_ylabel("height z (m)", color="#ccc", fontsize=9)

    # (b) electrostatic field
    ax = fig.add_subplot(gs[0, 1]); _bg(ax)
    im = ax.pcolormesh(xe, ze, np.where(Emag <= 0, np.nan, Emag).T, cmap="inferno",
                       shading="flat", zorder=2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(
        r"$|\mathbf{E}|$ (V m$^{-1}$)", color="#ccc", fontsize=9)
    ax.set_title(r"(b)  field:  $\nabla^2\varphi=-\rho_q/\varepsilon_0,\ "
                 r"\mathbf{E}=-\nabla\varphi$", **lab)
    ax.set_xlabel("x (m)", color="#ccc", fontsize=9)

    # (c) dielectric-breakdown discharge
    ax = fig.add_subplot(gs[0, 2]); _bg(ax)
    ax.pcolormesh(xe, ze, np.where(cd == 0, np.nan, cd).T * 1e9, cmap="bwr",
                  vmin=-lim * 1e9, vmax=lim * 1e9, shading="flat", alpha=0.55, zorder=2)
    for fl in fp["flashes"]:
        seg = fl["segments"].reshape(-1, 2, 2)
        ax.add_collection(LineCollection(seg, colors="white", linewidths=2.6, zorder=5))
        ax.add_collection(LineCollection(seg, colors="#bcd8ff", linewidths=1.0, zorder=5.1))
    ax.set_title("(c)  dielectric-breakdown discharge", **lab)

    # (d) field time series vs threshold
    axt = fig.add_subplot(gs[1, :])
    axt.set_facecolor("#0c1322")
    axt.plot(t_hist, E_hist, color="#7fb0ff", lw=1.3, label=r"domain-max $|\mathbf{E}|$")
    axt.axhline(E_BREAKDOWN, color="#ff6b6b", ls="--", lw=1.2,
                label="breakdown threshold (illustrative, 2-D)")
    axt.plot(flash_t, flash_E, "o", color="white", ms=4, mec="#7fb0ff", label="discharge")
    axt.axvline(fp["step"] * dt / 60.0, color="#888", ls=":", lw=0.8)
    axt.set_xlabel("time (min)", color="#ccc", fontsize=10)
    axt.set_ylabel(r"max $|\mathbf{E}|$ (V m$^{-1}$)", color="#ccc", fontsize=10)
    axt.tick_params(colors="#aaa")
    for s in axt.spines.values():
        s.set_color("#444")
    axt.legend(loc="upper right", facecolor="#0c1322", edgecolor="#444",
               labelcolor="#ddd", fontsize=9)
    axt.set_title("(d)  field growth to breakdown", **lab)

    fig.suptitle("Cloud electrification in a simulated deep convective cell (DropLab)",
                 color="white", fontsize=14, y=0.985)
    fig.text(0.5, 0.005,
             "Non-inductive charging: rebounding graupel–crystal collisions, rate from the "
             "gravitational kernel × δq≈5 fC (Takahashi 1978; Saunders & Peck 1998); "
             "graupel/crystal by size (D>0.2 mm); single charge-reversal temperature (−10°C). "
             "Field by Gauss's law; discharge by the dielectric-breakdown model (Niemeyer et al. 1984). "
             "2-D: |E| magnitude and the breakdown threshold are illustrative (see ELECTRIFICATION_AUDIT.md).",
             color="#9aa4b4", fontsize=7.5, ha="center", va="bottom")
    fig.savefig(FIG, dpi=120, facecolor="#0a0e18", bbox_inches="tight")
    print("saved", FIG)

    figg, anim = animate_electric(o, fps=12)
    anim.save(GIF, writer="pillow", fps=12, dpi=90)
    print("saved", GIF)


if __name__ == "__main__":
    main()
