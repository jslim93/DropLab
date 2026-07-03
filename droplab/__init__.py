import os as _os

# Silence the Intel OpenMP info line emitted by numba's parallel threading layer:
#   "OMP: Info #276: omp_set_nested routine deprecated, please use
#    omp_set_max_active_levels instead."
# It is a harmless runtime info (numba enables nested parallelism for our parallel=True
# kernels), but noisy. Must be set before libomp initialises — i.e. before any numba
# parallel kernel runs — so we set it here at package import.
_os.environ.setdefault("KMP_WARNINGS", "0")
