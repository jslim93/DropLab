"""Visualize the Linear Eddy Model droplet-spectrum broadening (SAM-LCM, ported to DropLab).
A deep warm congestus tower is run twice -- LEM off vs on -- and three panels show:
  (A) the in-cloud spectral width vs time: LEM broadening GROWS with cloud age (the cumulative
      r^2 supersaturation-fluctuation signature);
  (B) the final droplet size distribution, off vs on (the LEM tail is wider);
  (C) the per-super-droplet subgrid supersaturation anomaly eta' across the storm -- the
      turbulent fluctuation the LEM adds, which drives the broadening.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from droplab.flow2d_dynamic import run_flow2d_dynamic

OUT = "lem_broadening.png"


def width(f):
    r, A = f["r_um"], f["A"]
    c = (r > 2.0) & (r < 40.0)
    if c.sum() < 50:
        return np.nan, np.nan
    rc, w = r[c], A[c]
    m = np.average(rc, weights=w)
    return m, float(np.sqrt(np.average((rc - m) ** 2, weights=w)))


def main():
    base = dict(Nx=64, Nz=88, X=5000, Z=7000, nt=900, dt=2.0,
                n_super=64 * 88 * 50, collect_every=75, seed=2,
                dtheta_bubble=2.0, bubble_z=800.0, bubble_r=900.0,
                RH0=0.92, z_bl=800.0, collisions=False)
    off = run_flow2d_dynamic(**base, lem=False)
    on = run_flow2d_dynamic(**base, lem=True, lem_eps=1e-2)

    t = np.array([f["step"] for f in off["frames"]]) * base["dt"] / 60.0
    wo = np.array([width(f)[1] for f in off["frames"]])
    wn = np.array([width(f)[1] for f in on["frames"]])

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.42, wspace=0.28)

    # (A) spectral width vs time
    axA = fig.add_subplot(gs[:, 0])
    axA.plot(t, wo, "-o", color="#3b6fb6", lw=2, ms=4, label="LEM off")
    axA.plot(t, wn, "-o", color="#c0392b", lw=2, ms=4, label="LEM on (turbulent)")
    axA.set_xlabel("cloud age (min)"); axA.set_ylabel(r"spectral width $\sigma_r$ ($\mu$m)")
    axA.set_title("LEM broadening grows with cloud age\n(cumulative supersaturation-fluctuation effect)",
                  fontsize=11)
    axA.legend(frameon=False, loc="upper left")
    ax2 = axA.twinx()
    ratio = wn / wo
    ax2.plot(t, ratio, "--", color="#555", lw=1.3)
    ax2.axhline(1.0, color="#999", lw=0.8, ls=":")
    ax2.set_ylabel(r"broadening ratio $\sigma_{on}/\sigma_{off}$", color="#555")
    ax2.tick_params(axis="y", colors="#555")
    fin = ratio[np.isfinite(ratio)]
    if fin.size:
        axA.text(0.97, 0.05, f"final broadening  x{fin[-1]:.2f}", transform=axA.transAxes,
                 ha="right", va="bottom", fontsize=11, color="#c0392b",
                 bbox=dict(boxstyle="round", fc="#fbeae8", ec="#c0392b"))

    fo, fn = off["frames"][-1], on["frames"][-1]

    # (B) final size distributions
    axB = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 40, 41)
    for f, col, lab in ((fo, "#3b6fb6", "off"), (fn, "#c0392b", "on")):
        r, A = f["r_um"], f["A"]; c = (r > 2.0) & (r < 40.0)
        axB.hist(r[c], bins=bins, weights=A[c], histtype="step", color=col, lw=1.8,
                 density=True, label=lab)
    axB.set_xlabel(r"droplet radius ($\mu$m)"); axB.set_ylabel("norm. number")
    axB.set_title("final droplet spectrum", fontsize=11)
    axB.legend(frameon=False, fontsize=9)

    # (C) subgrid supersaturation anomaly across the storm
    axC = fig.add_subplot(gs[1, 1])
    r = fn["r_um"]; cloud = r > 2.0
    sc = axC.scatter(fn["x"][cloud], fn["z"][cloud], c=100.0 * fn["eta_anom"][cloud],
                     cmap="RdBu_r", s=5, vmin=-2, vmax=2, alpha=0.7, edgecolor="none")
    axC.set_xlabel("x (m)"); axC.set_ylabel("z (m)")
    axC.set_title("subgrid supersaturation anomaly", fontsize=11)
    cb = fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.02)
    cb.set_label(r"$s'$ (%)")

    fig.suptitle("DropLab Linear Eddy Model: turbulent broadening of the droplet spectrum "
                 "(SAM-LCM, ported)", fontsize=13, y=0.99)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print("saved", OUT)
    print("final broadening ratio:", f"{fin[-1]:.3f}" if fin.size else "n/a")


if __name__ == "__main__":
    main()
