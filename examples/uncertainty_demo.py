"""Stochastic, uncertain microphysics -- a parcel-model demonstrator.

Runs the IDENTICAL warm-cloud parcel many times with different random seeds and
shows that warm-rain initiation is not deterministic: rain onset and rain amount
scatter widely. Then shows that this spread is partly numerical (it shrinks as the
super-droplet count grows) but stays large near the rain threshold (physical
sensitivity).

    python -m examples.uncertainty_demo     # writes examples/uncertainty_demo.png
"""
import numpy as np

from droplab.uncertainty import seed_ensemble, spread_vs_resolution

# A clean parcel tuned near the warm-rain threshold (maximally sensitive).
CFG = dict(RH=0.98, w=1.0, N_raw=(60.0,), mu_um=(0.08,), sig=(1.6,),
           kappa=0.6, collisions=True, dt=1.0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A) seed ensemble at fixed resolution
    ens = seed_ensemble(n_members=24, n_ptcl=1000, nt=1200, **CFG)
    print("== Stochastic microphysics (24 seeds, identical parcel, n_ptcl=1000) ==")
    print(f"  rain-onset height: mean depends on seed; spread (std) = {ens['onset_z_std']:.0f} m")
    print(f"  final rain water qr: mean = {ens['final_qr_mean']:.3f} g/kg, "
          f"CoV = {ens['final_qr_cov']:.2f}  (seed-to-seed!)")
    print(f"  fraction of seeds that rained = {ens['rain_fraction']:.0%}")

    # B) spread vs super-droplet count
    res = spread_vs_resolution([200, 500, 1000, 2000, 5000], n_members=12, nt=1200, **CFG)
    print("== Convergence: seed spread vs super-droplet count ==")
    for n, s, c in zip(res["n_ptcl"], res["onset_z_std"], res["final_qr_cov"]):
        print(f"  n_ptcl={n:5.0f}  onset-height std = {s:5.0f} m   final-qr CoV = {c:.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    # A1: qr vs height, one line per seed
    for m in ens["members"]:
        ax[0].plot(m["qr"], m["z"], lw=0.8, alpha=0.6)
    ax[0].set_xlabel("rain water $q_r$ [g/kg]"); ax[0].set_ylabel("height [m]")
    ax[0].set_title("24 seeds, same parcel\n(different rain each time)")
    # A2: histogram of rain-onset height
    onz = ens["onset_z"][np.isfinite(ens["onset_z"])]
    ax[1].hist(onz, bins=10)
    ax[1].set_xlabel("rain-onset height [m]"); ax[1].set_ylabel("# seeds")
    ax[1].set_title(f"onset scatter: std = {ens['onset_z_std']:.0f} m")
    # B: spread vs resolution
    ax[2].plot(res["n_ptcl"], res["onset_z_std"], "o-")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("super-droplets $n_{ptcl}$"); ax[2].set_ylabel("onset-height std [m]")
    ax[2].set_title("spread shrinks with resolution\n(numerical part converges)")
    fig.suptitle("Microphysics is stochastic & uncertain (warm-rain parcel)", y=1.03)
    fig.tight_layout()
    fig.savefig("examples/uncertainty_demo.png", dpi=150, bbox_inches="tight")
    print("wrote examples/uncertainty_demo.png")


if __name__ == "__main__":
    main()
