"""Super-droplet convergence study for the Golovin benchmark.

The super-droplet method is a Monte-Carlo solver, so its error in a single
realisation should fall like n_sd^(-1/2) as the number of super-droplets grows
(cf. Shima et al. 2009). We measure the RMS relative error of the total number
concentration at tau = 1 against the exact Golovin solution, over an ensemble of
independent realisations, for a sweep of n_sd, and fit the convergence slope.

    python -m validation.golovin_convergence    # table + validation/golovin_convergence.png
"""
import numpy as np

from validation.golovin_analytic import B_GOLOVIN, radius_to_mass, number_conc
from validation.golovin_box import run_golovin_box

N0 = 2.0e8
X0 = float(radius_to_mass(10.0e-6))
M0 = N0 * X0
DT = 1.0
TAU = 1.0
T_EVAL = TAU / (B_GOLOVIN * M0)


def rms_error_for(n_sd, n_ens, dt=DT, seed0=0):
    """RMS over the ensemble of |N(tau)/N0 - exact| for a given super-droplet count."""
    errs = []
    for e in range(n_ens):
        res = run_golovin_box(N0, X0, n_sd, dt, [0.0, T_EVAL], seed=seed0 + e)
        N0r, M0r = res['N'][0], res['M'][0]
        N_num = res['N'][-1]
        N_exact = N0r * np.exp(-B_GOLOVIN * M0r * T_EVAL)
        errs.append(abs(N_num - N_exact) / N_exact)
    return float(np.sqrt(np.mean(np.square(errs))))


def sweep(n_sd_list, n_ens, dt=DT):
    return np.array([rms_error_for(n, n_ens, dt) for n in n_sd_list])


def fit_slope(n_sd_list, errs):
    """Log-log slope of RMS error vs n_sd (ideal Monte-Carlo = -0.5)."""
    p = np.polyfit(np.log(n_sd_list), np.log(errs), 1)
    return float(p[0])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sd_list = [128, 256, 512, 1024, 2048, 4096]
    n_ens = 12
    errs = sweep(n_sd_list, n_ens)
    slope = fit_slope(n_sd_list, errs)

    print(f"{'n_sd':>6} {'RMS rel err':>12}")
    for n, e in zip(n_sd_list, errs):
        print(f"{n:6d} {e:12.4%}")
    print(f"log-log slope = {slope:.3f}  (ideal Monte-Carlo -0.5)")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(n_sd_list, errs, "o-", label=f"DropLab SDM (slope {slope:.2f})")
    ref = errs[0] * (np.array(n_sd_list) / n_sd_list[0]) ** -0.5
    ax.loglog(n_sd_list, ref, "k--", lw=1, label=r"$n_{sd}^{-1/2}$ reference")
    ax.set_xlabel("number of super-droplets  $n_{sd}$")
    ax.set_ylabel(r"RMS rel. error of $N(\tau{=}1)$")
    ax.set_title("Super-droplet convergence (Golovin benchmark)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("validation/golovin_convergence.png", dpi=150)
    print("wrote validation/golovin_convergence.png")


if __name__ == "__main__":
    main()
