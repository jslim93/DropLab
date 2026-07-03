"""NetCDF output + selectable process diagnostics."""
import numpy as np
import pytest

pytest.importorskip("xarray", reason="NetCDF output needs xarray (pip install xarray netCDF4)")
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.output import save_run, gridded


def test_basic_partition_conserves_total_liquid(tmp_path):
    """The basic fields are derived by binning the super-droplets; haze + cloud + rain
    (qa+qc+qr) must exactly reconstruct the model's own total-liquid mixing ratio."""
    o = run_flow2d_dynamic(nt=150, Nx=32, Nz=32, n_super=8000, RH0=0.9, dtheta_bubble=3.0,
                           bubble_r=400., bubble_z=500., collisions=True, sediment=True,
                           collect_every=150, periodic_x=True, seed=1)
    ds = save_run(o, str(tmp_path / "b.nc"), fields="basic")
    f = o["frames"][-1]
    qc = ds["qc"].isel(time=-1).values.T
    qr = ds["qr"].isel(time=-1).values.T
    qa = ds["qa"].isel(time=-1).values.T
    assert np.abs((qc + qa + qr) - f["qc"]).max() < 1e-9


def test_diagnose_is_golden_safe_and_adds_rates():
    """diagnose= must not change the simulation (it only accumulates discarded amounts);
    when on, the frames carry the process-rate fields."""
    base = dict(nt=60, Nx=24, Nz=24, n_super=3000, ice=False, collect_every=60,
                periodic_x=True, seed=1)
    a = run_flow2d_dynamic(**base)
    b = run_flow2d_dynamic(diagnose="micro", **base)
    assert np.array_equal(a["frames"][-1]["qc"], b["frames"][-1]["qc"])   # unchanged
    assert "rates" in b["frames"][-1]
    assert {"cond", "evap", "freeze", "melt", "rime", "sip"} <= set(b["frames"][-1]["rates"])


def test_micro_rates_fire_in_a_mixed_phase_run(tmp_path):
    """A cold collisional deck should show non-zero deposition, freezing and riming rates."""
    o = run_flow2d_dynamic(nt=300, Nx=40, Nz=40, X=4000, Z=6000, dt=2.0, T0=258.0, RH0=0.999,
                           n_super=12000, collisions=True, ice=True, sediment=True,
                           freezing_mode="bigg", B_bigg=1.0e4, a_bigg=0.66, inp_n_cm3=10.0,
                           homogeneous=False, dtheta_bubble=2.0, bubble_r=600., bubble_z=700.,
                           collect_every=300, periodic_x=True, seed=3, diagnose="micro")
    ds = save_run(o, str(tmp_path / "m.nc"), fields="micro")
    assert {"cond", "dep", "freeze", "rime", "q_ice", "n_ice"} <= set(ds.data_vars)
    assert float(np.abs(ds["dep"]).max()) > 0.0       # deposition active
    assert float(np.abs(ds["freeze"]).max()) > 0.0    # freezing active


def test_radiation_cape_and_droplet_export(tmp_path):
    """fields='radiation' adds per-column albedo/lwp/iwp; droplets=True writes the Lagrangian
    super-droplet table; the sounding CAPE/EL go into attributes."""
    import xarray as xr
    from droplab.soundings import WEISMAN_KLEMP
    o = run_flow2d_dynamic(dynamics="anelastic", Nx=40, Nz=48, X=8000, Z=14000, dt=3.0, nt=300,
                           collect_every=150, n_super=9000, dtheta_bubble=4.5, bubble_r=1200.,
                           bubble_z=1200., periodic_x=True, seed=3, b_max=0.4, omega_max=0.18,
                           sponge_frac=0.16, sounding=WEISMAN_KLEMP, ice=True, homogeneous=True,
                           inp_n_cm3=0.5)
    p = str(tmp_path / "r.nc")
    save_run(o, p, fields="radiation", droplets=True)
    ds = xr.open_dataset(p)
    assert {"albedo", "lwp", "iwp"} <= set(ds.data_vars)
    assert ds["albedo"].dims == ("time", "x")             # per-column shape
    assert 0.0 <= float(ds["albedo"].max()) <= 1.0
    assert "CAPE_Jkg" in ds.attrs and ds.attrs["CAPE_Jkg"] > 500.0
    assert ds.sizes["record"] > 0 and {"sd_x", "sd_z", "sd_r_um", "sd_phase"} <= set(ds.variables)


def test_save_run_writes_loadable_netcdf(tmp_path):
    import xarray as xr
    o = run_flow2d_dynamic(nt=60, Nx=24, Nz=24, n_super=3000, collect_every=30,
                           periodic_x=True, seed=1)
    p = str(tmp_path / "o.nc")
    save_run(o, p, fields="basic")
    ds = xr.open_dataset(p)
    assert {"u", "w", "T", "qv", "qc", "qr", "nc", "nr", "na", "qa", "P"} <= set(ds.variables)
    assert ds.sizes["time"] == 2 and ds.sizes["z"] == 24 and ds.sizes["x"] == 24
    assert ds.attrs["model"] == "droplab 2D"


def test_sd_per_cell_and_sim_hours_conveniences():
    """Grid/dt-independent run sizing: sd_per_cell sets n_super = sd_per_cell*Nx*Nz, and
    sim_hours sets nt to cover that many hours at the current dt."""
    o = run_flow2d_dynamic(Nx=16, Nz=16, dt=2.0, sd_per_cell=40, nt=4, collect_every=4)
    assert o["A"].size == 40 * 16 * 16                 # 10240 super-droplets allocated
    # sim_hours -> nt: run two short steps' worth and confirm the step count via the field
    # history length is impractical here; assert the arithmetic the driver uses instead.
    o2 = run_flow2d_dynamic(Nx=12, Nz=12, dt=3.0, n_super=2000, sim_hours=0.01, collect_every=12)
    # 0.01 h at dt=3 s -> nt = round(36/3) = 12 steps -> one collected frame at step 12
    assert o2["frames"][-1]["step"] == 12
