"""Stage A: long-wave + net cloud radiative effect diagnostics."""
import numpy as np
from droplab.flow2d import Flow2D
from droplab.parameters import rho_liq, pi
from droplab.climate_diag import cloud_radiative_effect, lw_emissivity


def _cloud(flow, kz_lo, kz_hi, lwp_target_gm2, col_frac=1.0, r_um=10.0):
    """Build droplets that fill columns 0..col_frac*Nx between levels kz_lo..kz_hi
    to a target liquid water path (g/m^2). Returns M, A, x, z arrays."""
    ncol = int(col_frac * flow.Nx)
    levels = np.arange(kz_lo, kz_hi + 1)
    xs, zs = [], []
    for i in range(ncol):
        for k in levels:
            xs.append((i + 0.5) * flow.dx); zs.append((k + 0.5) * flow.dz)
    x = np.array(xs); z = np.array(zs)
    r = r_um * 1e-6
    # target: LWP = (sum M in a column)/(dx*depth)*1e3 g/m^2 ; per drop mass m1=(4/3)pi rho r^3
    n_per_col = len(levels)
    M_col = lwp_target_gm2 * 1e-3 * flow.dx                  # kg per column
    m1 = 4.0 / 3.0 * pi * rho_liq * r ** 3
    A_each = (M_col / n_per_col) / m1                        # multiplicity per super-droplet
    A = np.full(x.shape[0], A_each)
    M = np.full(x.shape[0], m1 * A_each)
    return M, A, x, z


def test_clear_sky_zero():
    flow = Flow2D(X=1000.0, Z=1000.0, Nx=10, Nz=10)
    T_col = np.linspace(290.0, 230.0, 10)
    M = np.zeros(0); A = np.zeros(0); x = np.zeros(0); z = np.zeros(0)
    d = cloud_radiative_effect(M, A, x, z, flow, T_col)
    assert d["swcre_mean"] == 0.0 and d["lwcre_mean"] == 0.0 and d["net_mean"] == 0.0


def test_emissivity_limits():
    assert abs(lw_emissivity(0.0, 0.0)) < 1e-12              # clear -> 0
    assert lw_emissivity(500.0, 0.0) > 0.99                  # thick -> ~1


def test_low_warm_deck_cools():
    """A thick, low (warm-top) stratocumulus deck: SW reflection wins -> net cooling."""
    flow = Flow2D(X=1000.0, Z=1000.0, Nx=10, Nz=10)
    T_col = np.linspace(290.0, 250.0, 10)
    M, A, x, z = _cloud(flow, 1, 2, lwp_target_gm2=120.0)    # top near k=2 -> warm
    d = cloud_radiative_effect(M, A, x, z, flow, T_col, mu0=0.6)
    assert d["swcre_mean"] < 0.0                              # reflects sunlight
    assert d["lwcre_mean"] > 0.0                              # some greenhouse
    assert d["net_mean"] < 0.0                                # SW dominates -> cools


def test_high_or_night_cloud_warms():
    """A high (cold-top) cloud, or any cloud in polar night (mu0=0): LW greenhouse wins."""
    flow = Flow2D(X=1000.0, Z=1000.0, Nx=10, Nz=10)
    T_col = np.linspace(290.0, 230.0, 10)
    M, A, x, z = _cloud(flow, 7, 9, lwp_target_gm2=80.0)     # top near k=9 -> very cold
    d_day = cloud_radiative_effect(M, A, x, z, flow, T_col, mu0=0.6)
    assert d_day["lwcre_mean"] > abs(d_day["swcre_mean"])    # LW beats SW for a cold cloud
    assert d_day["net_mean"] > 0.0
    d_night = cloud_radiative_effect(M, A, x, z, flow, T_col, mu0=0.0)
    assert d_night["swcre_mean"] == 0.0                      # polar night: no sunlight
    assert d_night["net_mean"] > 0.0                         # pure LW warming
