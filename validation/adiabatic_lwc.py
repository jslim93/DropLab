"""Adiabatic liquid-water benchmark for the parcel + condensation core.

An ascending adiabatic parcel (condensation on, no collision, no entrainment)
must (i) conserve total water q_t = q_v + q_l, (ii) stay saturated in cloud, and
(iii) carry the adiabatic liquid water q_l(z) = q_t - q_s(T,P). This drives the
production ``ascend_parcel`` + ``condense_soa`` routines directly and records the
pressure, so q_s(T,P) can be evaluated independently.

    python -m validation.adiabatic_lwc
"""
import numpy as np

from droplab.parameters import p0, r_a, cp, rv, rho_aero, z_env
from droplab.aero_init import aero_init
from droplab.parcel import ascend_parcel, parcel_rho
from droplab.condensation import esatw
from droplab.condensation_fast import condense_soa


def q_sat(T, P):
    """Saturation (water) mixing ratio [kg/kg] from the model's own esatw."""
    e_s = esatw(T)
    return (r_a / rv) * e_s / (P - e_s)


def run_adiabatic_ascent(T0=283.15, P0=950e2, RH=0.98, w=1.0, nt=600, dt=1.0,
                         n_ptcl=1500, N_raw=(120.0,), mu_um=(0.04,), sig=(1.6,),
                         kappa=0.6, seed=0):
    """Adiabatic ascent (no collision/entrainment). Returns arrays by time."""
    mu = np.log(np.array(mu_um) * 1e-6)
    sg = np.log(np.array(sig))
    th = T0 * (p0 / P0) ** (r_a / cp) + 5e-3 * z_env
    q0 = RH * esatw(T0) / (P0 - RH * esatw(T0)) * r_a / rv
    k_aero = [kappa] * (len(N_raw) + 1)

    np.random.seed(seed)
    T, q, pl = aero_init("Random", n_ptcl, P0, 0.0, T0, q0,
                         np.array(N_raw) * 1e6, mu, sg, rho_aero, k_aero, False)
    M = np.array([p.M for p in pl], dtype=np.float64)
    A = np.array([p.A for p in pl], dtype=np.float64)
    Ns = np.array([p.Ns for p in pl], dtype=np.float64)
    ka = np.array([p.kappa for p in pl], dtype=np.float64)

    rec = {k: [] for k in ("z", "T", "P", "qv", "ql", "qt", "RH")}
    P, z = P0, 0.0
    for t in range(nt):
        z, T, P = ascend_parcel(z, T, P, w, dt, (t + 1) * dt, 3000.0, th, 1200.0, "linear")
        rho_p, _, air_mass = parcel_rho(P, T)
        T, q = condense_soa(M, A, Ns, ka, T, q, P, dt, air_mass, rho_aero,
                            switch_adaptive_dt=True)
        ql = M.sum() / air_mass
        e_a = q * P / (q + r_a / rv)
        rec["z"].append(z); rec["T"].append(T); rec["P"].append(P)
        rec["qv"].append(q); rec["ql"].append(ql); rec["qt"].append(q + ql)
        rec["RH"].append(e_a / esatw(T))
    return {k: np.array(v) for k, v in rec.items()}


def main():
    r = run_adiabatic_ascent()
    qt = r["qt"]
    drift = (qt.max() - qt.min()) / qt.mean()
    incloud = r["ql"] > 1e-5
    zc = r["z"][incloud]
    ql_ad = r["qt"][incloud] - q_sat(r["T"][incloud], r["P"][incloud])
    print(f"total-water conservation: drift = {drift:.2e} (of mean q_t)")
    print(f"cloud base ~ {zc.min():.0f} m,  top of run {r['z'][-1]:.0f} m")
    print(f"in-cloud RH range: {r['RH'][incloud].min():.4f} .. {r['RH'][incloud].max():.4f}")
    print(f"{'z[m]':>7} {'q_l model[g/kg]':>16} {'q_l adiabatic[g/kg]':>20}")
    sel = np.linspace(0, zc.size - 1, 6).astype(int)
    for i in sel:
        print(f"{zc[i]:7.0f} {r['ql'][incloud][i]*1e3:16.3f} {ql_ad[i]*1e3:20.3f}")
    # mean relative agreement well above base
    deep = zc > zc.min() + 50.0
    rel = np.abs(r["ql"][incloud][deep] - ql_ad[deep]) / np.maximum(ql_ad[deep], 1e-9)
    print(f"mean |q_l model - adiabatic| / adiabatic, >50 m above base: {rel.mean():.2%}")


if __name__ == "__main__":
    main()
