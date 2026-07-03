"""Manuscript figure: super-droplet vs analytic Golovin spectrum.

Reproduces the canonical mass-density g(ln r) evolution under the additive
kernel (cf. Shima et al. 2009, Fig. 2). DropLab's collision solver (markers,
ensemble mean) is overlaid on the exact Golovin solution (lines).

    python -m validation.golovin_figure        # writes validation/golovin_spectrum.png
"""
import numpy as np

from validation.golovin_analytic import (
    B_GOLOVIN, radius_to_mass, g_lnr, number_conc,
)
from validation.golovin_box import run_golovin_box, spectrum_g_lnr

N0 = 2.0e8                           # 200 cm^-3
X0 = float(radius_to_mass(10.0e-6))   # mean droplet mass [kg] (r0 = 10 um)
M0 = N0 * X0
N_SD = 4096
DT = 1.0
N_ENS = 10                            # ensemble members (SDM spectra are noisy per-realisation)
TAUS = [0.0, 0.5, 1.0, 2.0]

# Radius grid for the spectrum [m]: 1 um .. 5 mm
R_EDGES = np.logspace(np.log10(1.0e-6), np.log10(5.0e-3), 41)
R_CTR = np.sqrt(R_EDGES[:-1] * R_EDGES[1:])


def compute():
    t_record = [tau / (B_GOLOVIN * M0) if tau > 0 else 0.0 for tau in TAUS]
    g_ens = np.zeros((N_ENS, len(TAUS), len(R_CTR)))
    for e in range(N_ENS):
        res = run_golovin_box(N0, X0, N_SD, DT, t_record, seed=100 + e)
        for j, pl in enumerate(res['plists']):
            g_ens[e, j] = spectrum_g_lnr(pl, R_EDGES)
    return t_record, g_ens.mean(axis=0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_record, g_num = compute()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(TAUS)))
    for j, (tau, t) in enumerate(zip(TAUS, t_record)):
        g_exact = g_lnr(R_CTR, t, N0, X0)
        ax.plot(R_CTR * 1e6, g_exact * 1e3, color=colors[j], lw=2,
                label=f"analytic  $\\tau$={tau:.1f}")
        ax.plot(R_CTR * 1e6, g_num[j] * 1e3, "o", color=colors[j], ms=3.5,
                mfc="none", label=f"DropLab  $\\tau$={tau:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel("droplet radius  [$\\mu$m]")
    ax.set_ylabel("mass density  $g(\\ln r)$  [g m$^{-3}$ per unit $\\ln r$]")
    ax.set_title("Golovin additive-kernel benchmark: DropLab SDM vs analytic")
    ax.set_xlim(2, 5000)
    ax.legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    out = "validation/golovin_spectrum.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    # numeric summary
    dlnr = np.diff(np.log(R_EDGES))
    print(f"{'tau':>5}   mass conservation in the binned spectrum")
    for j, (tau, t) in enumerate(zip(TAUS, t_record)):
        m_num = np.sum(g_num[j] * dlnr)           # integral of g(ln r) d ln r
        print(f"{tau:5.1f}   mass(g/m3) num={m_num*1e3:7.4f}  exact={M0*1e3:7.4f}")


if __name__ == "__main__":
    main()
