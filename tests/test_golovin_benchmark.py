"""Golovin (additive-kernel) analytic benchmark for the collision-coalescence solver.

The stochastic collection equation has no analytic solution for a general kernel,
but for the Golovin additive kernel K(x,y)=b(x+y) with an exponential initial
spectrum it does (Golovin 1963; Scott 1968). This is the standard ground-truth
test for super-droplet schemes (Shima et al. 2009). Here it drives DropLab's
production ``collection`` routine (via the ``kernel_fn`` hook), so a pass means
the real coalescence + Linear Sampling Method machinery reproduces the exact
solution — not a re-implementation.

Verifies:
  1. mass is conserved exactly by coalescence;
  2. the total-number decay N(t) follows the analytic N0 exp(-b M0 t) within a
     few percent (single stochastic realisation, finite multiplicity).
"""
import numpy as np

from validation.golovin_analytic import B_GOLOVIN, radius_to_mass
from validation.golovin_box import run_golovin_box

# Modest, CI-fast configuration (the standalone __main__ runs higher resolution).
N0 = 2.0e8                          # 200 cm^-3
X0 = float(radius_to_mass(10.0e-6))  # mean droplet mass [kg] (r0 = 10 um)
N_SD = 2048
DT = 2.0
SEED = 1


def _run():
    M0_nom = N0 * X0
    taus = [0.5, 1.0, 1.5]
    t_record = [0.0] + [tau / (B_GOLOVIN * M0_nom) for tau in taus]
    res = run_golovin_box(N0, X0, N_SD, DT, t_record, seed=SEED)
    return res


def test_mass_is_conserved_exactly():
    res = _run()
    M0r = res['M'][0]
    for M in res['M']:
        assert abs(M / M0r - 1.0) < 1e-9, "coalescence must conserve total water mass"


def test_number_decay_matches_golovin_analytic():
    res = _run()
    # Compare against the analytic decay using the *realised* initial state
    # (the finite exponential draw fixes M0, N0 for this box).
    N0r, M0r = res['N'][0], res['M'][0]
    for t, N in zip(res['t'], res['N']):
        N_exact = N0r * np.exp(-B_GOLOVIN * M0r * t)
        rel = abs(N - N_exact) / N_exact
        assert rel < 0.06, f"N(t) off analytic by {rel:.1%} at t={t:.0f}s"


def test_kernel_fn_does_not_perturb_default_physics():
    """The hook must be inert when unused: a default collision call is unchanged."""
    from droplab.collision import determine_collision
    from droplab.micro_particle import particles

    def mk(M, A):
        p = particles(1); p.M, p.A, p.Ns, p.kappa = M, A, 1e-18, 0.5; return p

    np.random.seed(0)
    a = determine_collision(1.0, mk(2e-9, 100.0), mk(5e-9, 10.0), 1.0, 1000.0,
                            1e5, 283.0, 1, 2)
    np.random.seed(0)
    b = determine_collision(1.0, mk(2e-9, 100.0), mk(5e-9, 10.0), 1.0, 1000.0,
                            1e5, 283.0, 1, 2, kernel_fn=None)
    assert a == b
