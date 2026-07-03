"""Aerosol-cloud-interaction (ACI) diagnostics demo on the 2D stratocumulus deck.

Runs a small full-length ensemble over background aerosol number N_a, fits the
standard sensitivities, and estimates the ERF from aerosol-cloud interaction
between a clean "pre-industrial" and a polluted "present-day" loading.

IMPORTANT — run length: precipitation susceptibility only appears once warm rain
has had time to form and fall (~15-25 min of cloud time). Short runs are
PRE-DRIZZLE and give a spurious sign. This demo uses nt=1200 (~30 min sim); each
member is ~15 s, the ensemble ~1-2 min.

    python examples/aci_diagnostics.py        # prints table, writes aci_diagnostics.png
"""
import numpy as np

from droplab.aci import (make_runner, aci_susceptibility, cloud_radiative_effect,
                       erfaci, erfaci_decomposition, diffusion_brightening)
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.soundings import DYCOMS, DYCOMS_RADIATION

# Non-precipitating Sc baseline (DYCOMS-II RF01), full drizzle-development length.
BASE = dict(
    Nx=40, Nz=40, X=4000.0, Z=1500.0, nt=1200, dt=1.5, n_super=18000,
    sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
    collisions=True, switch_TICE=True, eps=0.01, sediment=True,
    nu=6.0, nu_scalar=1.5, pert_amp=0.1, seed=0,
)
N_LIST = [50.0, 100.0, 200.0, 400.0]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runner = make_runner(depth=1.0, **BASE)
    # the N_LIST members are independent 2D runs — fan them across cores (n_jobs=-1).
    # Results are bit-identical to the serial sweep (n_jobs=1).
    res = aci_susceptibility(N_LIST, runner, n_jobs=-1)
    Na, Nd, reff, alb, prc, lwp = (res[k] for k in ("Na", "Nd", "reff", "albedo", "precip", "lwp"))

    cre = cloud_radiative_effect(alb)                       # W/m^2 (negative = cooling)
    erf = erfaci(alb[0], alb[-1])                           # clean PI vs polluted PD

    print("== ACI diagnostics (DYCOMS Sc, nt=1200) ==")
    print(f"{'N_a':>6} {'N_d':>7} {'reff_um':>8} {'LWP_g/m2':>9} {'albedo':>7} {'RWP_g/m2':>9} {'CRE_W/m2':>9}")
    for i, N in enumerate(Na):
        print(f"{N:6.0f} {Nd[i]:7.1f} {reff[i]*1e6:8.2f} {lwp[i]*1e3:9.2f} {alb[i]:7.3f} {prc[i]*1e3:9.3f} {cre[i]:9.1f}")
    print("-" * 66)
    print(f"ACI_N (dlnN_d/dlnN_a)  = {res['ACI_N']:.3f}  (r2={res['r2']['N']:.2f})")
    print(f"ACI_r (-dlnr/dlnN_a)   = {res['ACI_r']:.3f}  (r2={res['r2']['r']:.2f})  [Twomey ~1/3]")
    print(f"S_pop (-dlnRWP/dlnN_a) = {res['S_pop']:.3f}  (r2={res['r2']['precip']:.2f})  [>0 = suppression]")
    print(f"ACI_L (dlnLWP/dlnN_d)  = {res['ACI_L']:.3f}  (r2={res['r2']['lwp']:.2f})  [>0 ascending/inverted-V]")
    print(f"S_albedo (dlnA/dlnN_a) = {res['S_albedo']:.3f}  (r2={res['r2']['albedo']:.2f})  [brightening efficiency]")
    print(f"ERFaci (PI N={Na[0]:.0f} -> PD N={Na[-1]:.0f}) = {erf:.2f} W/m^2  [negative = cooling]")

    # ERFaci decomposition (Twomey vs LWP-adjustment) from two dedicated PI/PD runs
    pi_out = run_flow2d_dynamic(N_modes=(float(N_LIST[0]),), depth=1.0, **BASE)
    pd_out = run_flow2d_dynamic(N_modes=(float(N_LIST[-1]),), depth=1.0, **BASE)
    dec = erfaci_decomposition(pi_out, pd_out)
    print("-- ERFaci decomposition [W/m^2] --")
    print(f"  RFaci_Twomey      = {dec['RFaci_Twomey']:+.2f}  (droplet-number brightening)")
    print(f"  ERFaci_adjustment = {dec['ERFaci_adjustment']:+.2f}  (LWP / inverted-V response)")
    print(f"  ERFaci_total      = {dec['ERFaci_total']:+.2f}")

    # Gristey-2025 SAI diffusion-brightening on the polluted cloud field (reuse pd_out)
    db = diffusion_brightening(pd_out, delta_f_diff=0.2, mu0=0.7)
    print(f"-- SAI diffusion-brightening (Gristey 2025, df_diff=0.2, mu0=0.7) --")
    print(f"  d_albedo = {db['d_albedo']:+.4f}  ->  bonus CRE = {db['d_CRE']:+.2f} W/m^2  "
          f"[+cooling at high sun; reverses at low sun]")

    # log-log panels with fitted slopes (last panel: the LWP-N_d inverted-V)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, y, ylab, slope in (
        (ax[0], Nd, r"$N_d$ [cm$^{-3}$]", res["ACI_N"]),
        (ax[1], reff * 1e6, r"$r_{\rm eff}$ [$\mu$m]", -res["ACI_r"]),
        (ax[2], prc * 1e3, r"RWP [g m$^{-2}$]", -res["S_pop"]),
    ):
        a.loglog(Na, np.maximum(y, 1e-6), "o-")
        a.set_xlabel(r"$N_a$ [cm$^{-3}$]"); a.set_ylabel(ylab)
    ax[0].set_title(f"ACI_N = {res['ACI_N']:+.2f}")
    ax[1].set_title(f"ACI_r = {res['ACI_r']:+.2f} (Twomey)")
    ax[2].set_title(f"S_pop = {res['S_pop']:+.2f} (suppression)")
    # inverted-V: LWP vs N_d (this model shows the ascending/suppression branch)
    ax[3].plot(Nd, lwp * 1e3, "o-")
    ax[3].set_xscale("log")
    ax[3].set_xlabel(r"$N_d$ [cm$^{-3}$]"); ax[3].set_ylabel(r"in-cloud LWP [g m$^{-2}$]")
    ax[3].set_title(f"LWP-$N_d$: ACI_L = {res['ACI_L']:+.2f}")
    fig.suptitle(f"ACI diagnostics — DYCOMS Sc, ERFaci = {erf:.1f} W/m$^2$", y=1.02)
    fig.tight_layout()
    fig.savefig("examples/aci_diagnostics.png", dpi=150, bbox_inches="tight")
    print("wrote examples/aci_diagnostics.png")


if __name__ == "__main__":
    main()
