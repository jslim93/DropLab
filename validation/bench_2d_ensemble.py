"""Validate + benchmark the 2D parallel ensemble: members must be bit-identical to a
serial sweep, and the wall time should drop ~linearly with cores.

Run:  python -m validation.bench_2d_ensemble
"""
import time
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.soundings import DYCOMS, DYCOMS_RADIATION
from droplab.flow2d_ensemble import run_parallel

CFG = dict(nt=120, dt=1.0, Nx=64, Nz=40, X=3200.0, Z=1200.0, n_super=30000,
           sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True, pert_amp=0.1,
           nu=6, nu_scalar=1.5, collisions=True, switch_TICE=True, eps=0.01,
           sediment=True, collect_every=100000)


def member(seed):
    """One ensemble member: a 2D run, returning only a small summary."""
    out = run_flow2d_dynamic(N_modes=(200.,), seed=seed, **CFG)
    return (float(out["surf_precip"]), int(out["M"].size), float(out["theta"].mean()))


def main():
    seeds = list(range(8))
    run_flow2d_dynamic(nt=2, Nx=16, Nz=16, X=800.0, Z=800.0, n_super=2000,
                       sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, periodic_x=True,
                       collisions=True, switch_TICE=True, sediment=True,
                       collect_every=100000)  # warm numba
    t = time.time(); ser = [member(s) for s in seeds]; t_ser = time.time() - t
    t = time.time(); par = run_parallel(seeds, member, n_jobs=-1, threads_per_worker=1)
    t_par = time.time() - t
    print(f"{len(seeds)} members  serial {t_ser:.1f}s  parallel {t_par:.1f}s  "
          f"speedup {t_ser / t_par:.1f}x")
    print("bit-identical (parallel == serial):", ser == par)


if __name__ == "__main__":
    main()
