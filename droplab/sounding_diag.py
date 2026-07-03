"""Sounding diagnostics: CAPE / CIN / LFC / equilibrium level from parcel theory.

A surface parcel is lifted pseudo-adiabatically (dry-adiabatic lapse rate below the LCL, the
moist-adiabatic lapse rate above); CAPE is the integrated positive buoyancy from the level of
free convection (LFC) to the equilibrium level (EL). Used to BENCHMARK the deep-convection
core: a physically correct storm reaches the EL with updrafts ~0.3-0.5*sqrt(2*CAPE). Validated
against the Weisman-Klemp (1982) sounding (CAPE ~ 2120 J/kg, EL ~ 12 km).
"""
import numpy as np
from droplab.parameters import p0, r_a, cp, rv, l_v, g
from droplab.condensation import esatw

_kap = r_a / cp
_eps = r_a / rv


def _qsat(T, P):
    es = esatw(T)
    return _eps * es / np.maximum(P - es, 1.0)


def parcel_cape(sounding, N=1600, P0=1.0e5):
    """CAPE [J/kg], CIN [J/kg], LFC [m], EL [m] for a sounding dict (z [m], theta [K],
    qv [g/kg]). The profile is integrated to its top z; provide a sounding that includes the
    stratosphere (a strongly stable cap) or the EL will be pushed to the domain top."""
    zs = np.asarray(sounding["z"], float)
    Z = float(zs[-1])
    dz = Z / N
    z = (np.arange(N) + 0.5) * dz
    th_e = np.interp(z, zs, sounding["theta"])
    qv_e = np.interp(z, zs, np.asarray(sounding["qv"], float) * 1e-3)
    # hydrostatic Exner with virtual potential temperature
    inv = 1.0 / (th_e * (1.0 + 0.61 * qv_e))
    exner = (P0 / p0) ** _kap - g / cp * (np.cumsum(inv) * dz - 0.5 * inv * dz)
    P = p0 * exner ** (1.0 / _kap)
    T_e = th_e * exner
    Tv_e = T_e * (1.0 + 0.608 * qv_e)
    # parcel ascent: dry-adiabatic lapse below the LCL, moist-adiabatic above
    Tp = T_e[0]; qp = qv_e[0]
    Tv_p = np.empty(N)
    for k in range(N):
        if k > 0:
            qs = _qsat(Tp, P[k - 1])
            if qp < qs:                                # unsaturated -> dry adiabat
                Tp = Tp - (g / cp) * dz
            else:                                      # saturated -> moist-adiabatic lapse rate
                gam = g * (1.0 + l_v * qs / (r_a * Tp)) / (
                    cp + l_v ** 2 * qs * _eps / (r_a * Tp ** 2))
                Tp = Tp - gam * dz
                qp = _qsat(Tp, P[k])
        Tv_p[k] = Tp * (1.0 + 0.608 * min(qp, _qsat(Tp, P[k])))
    b = g * (Tv_p - Tv_e) / Tv_e
    pos = b > 0
    lfc = next((k for k in range(2, N) if pos[k] and not pos[k - 1]), None)
    if lfc is None:
        return 0.0, 0.0, None, None
    el = next((k for k in range(lfc + 1, N) if not pos[k]), N - 1)
    cape = float(np.sum(b[lfc:el][b[lfc:el] > 0]) * dz)
    cin = float(-np.sum(b[:lfc][b[:lfc] < 0]) * dz)
    return cape, cin, float(z[lfc]), float(z[el])
