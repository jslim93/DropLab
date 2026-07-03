import os
import pathlib
import tempfile

# Isolate the sandbox's persistent disk cache from the developer's real
# ~/.droplab_cache during tests: point it at a fresh temp dir (set BEFORE
# app.ui.cache is imported so its module-level dir picks this up). A fresh dir
# keeps cache-dependent tests (e.g. "first run streams live") deterministic.
# mkdtemp (unique per session) rather than a fixed shared name: two CONCURRENT
# pytest sessions sharing one dir poison each other's "first run" assumptions.
_TEST_CACHE = pathlib.Path(tempfile.mkdtemp(prefix="droplab_cache_pytest_"))
os.environ["DROPLAB_CACHE_DIR"] = str(_TEST_CACHE)

import pytest
from droplab.micro_particle import particles


def make_particle(M, A, Ns=1e-18, kappa=0.5):
    p = particles(1)
    p.M, p.A, p.Ns, p.kappa = M, A, Ns, kappa
    return p
