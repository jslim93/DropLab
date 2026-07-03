"""Parallel ensemble execution for the 2D model.

2D members (`run_flow2d_dynamic` runs differing only in seed or parameters) are
embarrassingly parallel, so fanning them across worker PROCESSES gives near-linear
throughput for the multi-run studies that dominate the 2D wall time: ACI
susceptibility sweeps over background aerosol, MCB seeded-vs-unseeded pairs, and
stochastic ensembles.

Nested-parallelism note: the single 2D run already uses numba `parallel=True`
(condensation/advection). Running members in parallel processes on TOP of that would
oversubscribe the cores (n_procs x n_threads >> cores). So each worker pins numba to
`threads_per_worker` threads (default 1) — the embarrassingly-parallel ensemble
scales better than the within-run (which has the serial dM_cell-scatter Amdahl tail),
so 1 thread/worker x n_jobs~cores is both faster and cleaner for ensembles.

Results are bit-identical to a serial sweep: each member's seed fixes its RNG, and the
parallel condensation/advection kernels are thread-order-independent, so the per-member
result does not depend on the thread count.
"""
import os
from joblib import Parallel, delayed


def run_parallel(items, member_fn, n_jobs=-1, threads_per_worker=1):
    """Run `member_fn(item)` for every item across worker processes (joblib/loky).

    Returns a list of results in input order. `member_fn` should run the 2D
    simulation AND extract a SMALL summary inside the worker (return the diagnostics
    you need, not the full output dict — the big arrays/frames are expensive to ship
    back between processes). `n_jobs=-1` uses all cores.

    `threads_per_worker` (default 1) pins each worker's numba thread count via the
    NUMBA_NUM_THREADS env var, set BEFORE the workers spawn so they read it when they
    import numba. This is essential: calling numba.set_num_threads inside the worker
    is too late (its threadpool is already up), and without the cap each of the
    n_jobs workers would launch a full numba threadpool -> n_jobs x n_threads
    oversubscription -> each member runs several times slower (measured: ~2x ensemble
    speedup uncapped vs ~6x with 1 thread/worker on a 10-core machine). The
    embarrassingly-parallel ensemble scales better across members than the within-run
    numba parallelism (which has the serial dM_cell-scatter Amdahl tail), so 1
    thread/worker is the right default for multi-run studies.
    """
    prev = os.environ.get("NUMBA_NUM_THREADS")
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, int(threads_per_worker)))
    try:
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(member_fn)(it) for it in items
        )
    finally:
        if prev is None:
            os.environ.pop("NUMBA_NUM_THREADS", None)
        else:
            os.environ["NUMBA_NUM_THREADS"] = prev
