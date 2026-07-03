"""Adiabatic parcel benchmark: water conservation + adiabatic liquid water.

An ascending parcel with condensation only must conserve total water, stay
saturated in cloud, and carry the adiabatic liquid water q_t - q_s(T,P).
"""
import numpy as np

from validation.adiabatic_lwc import run_adiabatic_ascent, q_sat


def test_adiabatic_ascent_conserves_water_and_tracks_adiabat():
    r = run_adiabatic_ascent(nt=400, dt=1.0, seed=0)

    # (i) total water conserved to ~machine precision
    qt = r["qt"]
    drift = (qt.max() - qt.min()) / qt.mean()
    assert drift < 1e-9, f"total water not conserved: drift={drift:.2e}"

    incloud = r["ql"] > 1e-5
    assert incloud.sum() > 20, "parcel never formed a cloud"

    # (ii) parcel stays at (slightly super)saturation in cloud
    rh = r["RH"][incloud]
    assert rh.min() > 0.99 and rh.max() < 1.03, f"in-cloud RH out of range: {rh.min()}..{rh.max()}"

    # (iii) liquid water matches the adiabatic value q_t - q_s(T,P) above base
    zc = r["z"][incloud]
    ql_ad = r["qt"][incloud] - q_sat(r["T"][incloud], r["P"][incloud])
    deep = zc > zc.min() + 50.0
    rel = np.abs(r["ql"][incloud][deep] - ql_ad[deep]) / np.maximum(ql_ad[deep], 1e-9)
    assert rel.mean() < 0.06, f"LWC off adiabatic by {rel.mean():.1%}"
