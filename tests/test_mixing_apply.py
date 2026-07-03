import numpy as np
import matplotlib
matplotlib.use("Agg")
from droplab.micro_particle import particles
from droplab.parcel import create_env_profiles
from droplab.mixing import ParameterizedMixing
from droplab.condensation import esatw
from droplab.parameters import p0, r_a, cp, rv


def _env():
    """Analytic environment params the mixing now takes, recovered from the plotting
    helper (theta_init + lapse from the sounding it draws)."""
    _qv, th, z_env = create_env_profiles(290.0, 0.010, 0.0, 95000.0, "Stable")
    theta_init = float(th[0])
    lapse = float((th[1] - th[0]) / (z_env[1] - z_env[0]))
    return theta_init, lapse, 0.2, float(z_env[0]), float(z_env[-1])


def _drops(n=100):
    ps = []
    for _ in range(n):
        p = particles(1); p.M, p.A, p.Ns, p.kappa = 1.0e-9, 50.0, 1e-18, 0.5
        ps.append(p)
    return ps


def _q_env_inline(theta_init, lapse, rh_env, z, P, z_init, z_top):
    """The environmental q_v the mixing computes inline, for the expected-value check."""
    zc = min(max(z, z_init), z_top)
    T_env = (theta_init + lapse * zc) * (P / p0) ** (r_a / cp)
    es = esatw(T_env)
    return rh_env * (r_a / rv) * es / (P - es)


def test_redistribution_adds_no_net_water():
    # Entrainment legitimately exchanges water with the environment; the IHMD
    # redistribution (liquid -> vapor) must itself be conservative. So the TOTAL
    # water change must equal EXACTLY the bulk entrainment exchange and nothing more.
    ti, lap, rh, z0, zt = _env(); ps = _drops()
    air_mass = 100.0
    lam, w, dt = 5e-4, 1.0, 1.0
    P = 90000.0
    mix = ParameterizedMixing(lam, 0.5, ti, lap, rh, z_init=z0, z_top=zt)
    T, q = 288.0, 0.009
    total0 = q * air_mass + sum(p.M for p in ps)
    q_env = _q_env_inline(ti, lap, rh, 500.0, P, z0, zt)
    expected_change = (lam * w * dt) * (q_env - q) * air_mass   # bulk exchange only
    ps, T1, q1 = mix.apply(ps, T, q, P, 500.0, dt=dt, w=w, air_mass=air_mass)
    total1 = q1 * air_mass + sum(p.M for p in ps)
    assert np.isclose(total1 - total0, expected_change, atol=1e-12)
    assert np.isfinite(T1) and np.isfinite(q1)


def test_disabled_is_noop():
    ti, lap, rh, z0, zt = _env(); ps = _drops()
    mix = ParameterizedMixing(0.0, 0.5, ti, lap, rh, z_init=z0, z_top=zt)
    M_before = [p.M for p in ps]
    ps, T1, q1 = mix.apply(ps, 288.0, 0.009, 90000.0, 500.0, dt=1.0, w=1.0, air_mass=100.0)
    assert [p.M for p in ps] == M_before and (T1, q1) == (288.0, 0.009)


def test_apply_is_deterministic():
    ti, lap, rh, z0, zt = _env(); ps1 = _drops(); ps2 = _drops()
    m1 = ParameterizedMixing(8e-4, 0.7, ti, lap, rh, z_init=z0, z_top=zt)
    m2 = ParameterizedMixing(8e-4, 0.7, ti, lap, rh, z_init=z0, z_top=zt)
    r1 = m1.apply(ps1, 288.0, 0.009, 90000.0, 500.0, 1.0, 1.0, 100.0)
    r2 = m2.apply(ps2, 288.0, 0.009, 90000.0, 500.0, 1.0, 1.0, 100.0)
    assert [p.M for p in r1[0]] == [p.M for p in r2[0]] and r1[1:] == r2[1:]
