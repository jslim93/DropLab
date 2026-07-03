"""Box-model driver for the Golovin benchmark.

Evolves a super-droplet population under *pure stochastic collection* using
DropLab's own collision routine (``droplab.collision.collection`` with the
``kernel_fn`` hook), so the test exercises the production coalescence + Linear
Sampling Method machinery — not a re-implementation. The additive Golovin kernel
is injected via ``kernel_fn``; no ascent, condensation, or sedimentation.

Run directly for a high-resolution check (optionally writes a spectrum plot):

    python -m validation.golovin_box
"""
import numpy as np

from droplab.micro_particle import particles
from droplab.collision import collection
from droplab.parameters import PARCEL_AIR_MASS
from validation.golovin_analytic import B_GOLOVIN, RHO_W, radius_to_mass, mass_to_radius

# Fixed box thermodynamics (only used to size the sampling volume; the analytic
# kernel makes radius/terminal-velocity/temperature irrelevant to the rate).
RHO_PARCEL = 1.0                       # [kg m^-3]  -> V_box = PARCEL_AIR_MASS / RHO_PARCEL
V_BOX = PARCEL_AIR_MASS / RHO_PARCEL    # [m^3]
P_ENV = 1.0e5                           # [Pa]
T_PARCEL = 283.0                        # [K]


def golovin_kernel(b=B_GOLOVIN):
    """Return K(m1, m2) = b (m1 + m2)  [m^3 s^-1], masses in kg."""
    return lambda m1, m2: b * (m1 + m2)


def init_population(N0, x0, n_sd, seed=0):
    """Constant-multiplicity super-droplet sampling of n(x,0) = (N0/x0) exp(-x/x0).

    Returns (particle_list, A) where A is the (equal) multiplicity per super-droplet.
    """
    rng = np.random.default_rng(seed)
    x = rng.exponential(scale=x0, size=n_sd)        # single-droplet masses [kg]
    n_real = N0 * V_BOX                              # total real droplets in the box
    A = n_real / n_sd                               # equal multiplicity
    plist = []
    for xi in x:
        p = particles(1)
        p.M = A * xi      # super-droplet water mass [kg]
        p.A = A           # multiplicity
        p.Ns = 1.0e-18
        p.kappa = 0.5
        p.z = 0.0
        plist.append(p)
    return plist, A


def number_and_mass_conc(plist):
    """Diagnose (N, M) concentrations [m^-3], [kg m^-3] from the super-droplet list."""
    A = np.array([p.A for p in plist])
    M = np.array([p.M for p in plist])
    return A.sum() / V_BOX, M.sum() / V_BOX


def spectrum_g_lnr(plist, r_edges):
    """Binned mass density g(ln r) [kg m^-3 per unit ln r] over the given r bin edges."""
    A = np.array([p.A for p in plist])
    M = np.array([p.M for p in plist])
    x = M / np.maximum(A, 1e-300)                   # single-droplet mass
    r = mass_to_radius(x)
    lnr_edges = np.log(r_edges)
    dlnr = np.diff(lnr_edges)
    mass_conc = M / V_BOX                            # kg m^-3 per super-droplet
    g, _ = np.histogram(np.log(r), bins=lnr_edges, weights=mass_conc)
    return g / dlnr


def run_golovin_box(N0, x0, n_sd, dt, t_record, b=B_GOLOVIN, seed=0):
    """Evolve the box and snapshot at the requested times.

    Returns dict: {'t': [...], 'N': [...], 'M': [...], 'plists': [list-at-each-t]}.
    """
    np.random.seed(seed)                            # collection() draws from np.random
    plist, _ = init_population(N0, x0, n_sd, seed=seed)
    kern = golovin_kernel(b)

    t_record = sorted(t_record)
    out = {'t': [], 'N': [], 'M': [], 'plists': []}
    t = 0.0
    nstep = int(round(max(t_record) / dt))
    rec_idx = 0
    # snapshot at t=0 if requested
    while rec_idx < len(t_record) and t_record[rec_idx] <= 1e-12:
        N, M = number_and_mass_conc(plist)
        out['t'].append(0.0); out['N'].append(N); out['M'].append(M)
        out['plists'].append([_clone(p) for p in plist])
        rec_idx += 1

    for step in range(nstep):
        plist, _, _, _ = collection(
            dt, plist, RHO_PARCEL, RHO_W, P_ENV, T_PARCEL,
            0.0, 0.0, 0.0, False, 0.0, 1.0e30, 0.0,
            kernel_fn=kern,
        )
        t = (step + 1) * dt
        while rec_idx < len(t_record) and t >= t_record[rec_idx] - 1e-9:
            N, M = number_and_mass_conc(plist)
            out['t'].append(t); out['N'].append(N); out['M'].append(M)
            out['plists'].append([_clone(p) for p in plist])
            rec_idx += 1
    return out


def _clone(p):
    q = particles(1)
    q.M, q.A = p.M, p.A
    return q


if __name__ == "__main__":
    # High-resolution standalone check.
    N0 = 2.0e8                       # 200 cm^-3
    r0 = 10.0e-6                     # characteristic initial radius [m]
    x0 = float(radius_to_mass(r0))   # mean droplet mass [kg]
    M0 = N0 * x0                     # ~0.84 g m^-3 LWC
    n_sd = 4096
    dt = 1.0
    # tau = b*M0*t ; pick t for tau = 0.5, 1, 2
    times = [t for t in (0.5, 1.0, 2.0)]
    t_record = [tau_t / (B_GOLOVIN * M0) for tau_t in times]

    from validation.golovin_analytic import number_conc, g_lnr
    res = run_golovin_box(N0, x0, n_sd, dt, t_record, seed=1)
    print(f"M0={M0*1e3:.3f} g/m^3,  x0={x0:.3e} kg,  n_sd={n_sd}")
    print(f"{'tau':>5} {'t[s]':>7} {'N/N0 num':>10} {'N/N0 exact':>11} {'M/M0':>8}")
    for tau_t, t, N, M in zip(times, res['t'], res['N'], res['M']):
        N_exact = number_conc(t, N0, x0)
        print(f"{tau_t:5.1f} {t:7.1f} {N/N0:10.4f} {N_exact/N0:11.4f} {M/M0:8.5f}")
