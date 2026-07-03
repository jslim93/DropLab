"""NetCDF output + selectable diagnostics for the 2D model.

    from droplab.output import save_run
    out = run_flow2d_dynamic(..., diagnose="micro")   # process rates need the diagnose flag
    save_run(out, "storm.nc", fields="micro")          # -> self-describing NetCDF (xarray)

`fields` is a preset name or an explicit list, so a run only writes what is asked for (memory).
The "basic" fields are DERIVED post-hoc from the saved super-droplets, so they work on any run;
the process-rate fields ("micro") require diagnose= in run_flow2d_dynamic (see flow2d_dynamic).

Super-droplet size partitioning (radius r):
    haze / unactivated  r <  1 um      -> na, qa
    cloud               1 um <= r < 40 um -> nc, qc
    rain                r >= 40 um      -> nr, qr
"""
import numpy as np
from droplab.parameters import p0, r_a, cp, rho_liq, rho_ice, pi

R_HAZE = 1.0e-6        # m : below this a super-droplet is unactivated (aerosol/haze)
R_RAIN = 40.0e-6       # m : cloud / rain boundary
_kap = r_a / cp

# variable presets (choose a name, or pass an explicit list to save_run)
BASIC = ["u", "w", "T", "qv", "qc", "qr", "nc", "nr", "na", "qa"]
ICE = ["q_ice", "n_ice"]
RATES = ["cond", "evap", "dep", "sub", "freeze", "melt", "rime", "sip"]
RAD = ["albedo", "lwp", "iwp", "tau", "reff"]          # per-column (Nx,) -> (time, x)
PRESETS = {
    "basic": BASIC,
    "micro": BASIC + ICE + RATES,
    "radiation": BASIC + ICE + RAD,
    "full": BASIC + ICE + RATES + RAD,
}

_ATTRS = {
    "u": ("m/s", "horizontal velocity"), "w": ("m/s", "vertical velocity"),
    "T": ("K", "temperature"), "qv": ("g/kg", "water vapour mixing ratio"),
    "qc": ("g/kg", "cloud water mixing ratio"), "qr": ("g/kg", "rain water mixing ratio"),
    "qa": ("g/kg", "haze/unactivated water mixing ratio"),
    "q_ice": ("g/kg", "ice mixing ratio"),
    "nc": ("1/cm3", "cloud droplet number"), "nr": ("1/cm3", "rain drop number"),
    "na": ("1/cm3", "aerosol/haze number"), "n_ice": ("1/cm3", "ice crystal number"),
    "cond": ("g/kg/s", "condensation rate"), "evap": ("g/kg/s", "evaporation rate"),
    "dep": ("g/kg/s", "vapour deposition rate"), "sub": ("g/kg/s", "ice sublimation rate"),
    "freeze": ("g/kg/s", "freezing rate"), "melt": ("g/kg/s", "melting rate"),
    "rime": ("g/kg/s", "riming rate"), "sip": ("1/cm3/s", "secondary-ice (Hallett-Mossop) rate"),
    "albedo": ("1", "cloud short-wave albedo (per column)"),
    "lwp": ("g/m2", "liquid water path (per column)"), "iwp": ("g/m2", "ice water path (per column)"),
    "tau": ("1", "short-wave optical depth (per column)"),
    "reff": ("um", "cloud effective radius (per column)"),
}


def _radiation_columns(frame, flow, depth):
    """Per-x-column cloud radiative properties (length Nx): SW albedo / optical depth / effective
    radius / LWP from droplab.climate_diag, plus the ice-water path. Diagnostic optics (Twomey /
    two-stream); not interactive radiation."""
    from droplab.climate_diag import optics_from_frame
    o = optics_from_frame(frame, flow, depth)
    Nx, dx = flow.Nx, flow.dx
    x, r_um, A = frame["x"], frame["r_um"], frame["A"]
    phase = frame.get("phase", np.zeros(x.shape[0], dtype=np.int8))
    col = np.clip((x / dx).astype(np.int64), 0, Nx - 1)
    ice = phase == 1
    r = r_um * 1e-6
    iwp = np.zeros(Nx)                                  # ice water path per column (kg/m2)
    np.add.at(iwp, col[ice], A[ice] * (4.0 / 3.0 * pi * r[ice] ** 3 * rho_ice))
    iwp /= (dx * depth)
    return {"albedo": o["albedo"], "tau": o["tau"], "reff": o["reff"] * 1e6,
            "lwp": o["lwp"] * 1e3, "iwp": iwp * 1e3}


def _grid_basic(frame, flow, air_2d, depth):
    """Derive gridded basic fields (qc/qr/nc/nr/qa/na/q_ice/n_ice) by binning the saved
    super-droplets into the model grid by cell and radius/phase. air_2d is the EXACT per-cell
    air mass (kg) the model used (so the derived mixing ratios match the model's own)."""
    Nx, Nz, dx, dz = flow.Nx, flow.Nz, flow.dx, flow.dz
    x, z, r_um, A = frame["x"], frame["z"], frame["r_um"], frame["A"]
    phase = frame.get("phase", np.zeros(x.shape[0], dtype=np.int8))
    ix = np.clip((x / dx).astype(np.int64), 0, Nx - 1)
    iz = np.clip((z / dz).astype(np.int64), 0, Nz - 1)
    c = ix * Nz + iz
    r = r_um * 1e-6
    rho = np.where(phase == 1, rho_ice, rho_liq)
    Msd = A * (4.0 / 3.0 * pi * r ** 3 * rho)        # super-droplet water mass (kg)
    liq = phase == 0
    sel = {"qa": liq & (r < R_HAZE), "qc": liq & (r >= R_HAZE) & (r < R_RAIN),
           "qr": liq & (r >= R_RAIN), "q_ice": phase == 1}
    air = air_2d                                      # per-cell air mass (kg), (Nx,Nz)
    vol_cm3 = dx * dz * depth * 1e6                   # cell volume in cm^3

    def scat(mask, vals):
        out = np.zeros(Nx * Nz)
        np.add.at(out, c[mask], vals[mask])
        return out.reshape(Nx, Nz)

    g = {}
    for q, m in sel.items():
        g[q] = scat(m, Msd) / air * 1e3              # g/kg
    g["nc"] = scat(sel["qc"], A) / vol_cm3            # 1/cm3
    g["nr"] = scat(sel["qr"], A) / vol_cm3
    g["na"] = scat(sel["qa"], A) / vol_cm3
    g["n_ice"] = scat(sel["q_ice"], A) / vol_cm3
    return g


def gridded(result, fields="basic"):
    """Return {name: (time, Nz, Nx) array} for the requested fields, plus coords."""
    flow = result["flow"]; P_col = result["P_col"]; T_col = result["T_col"]
    depth = result.get("depth", 1.0); dt = result.get("dt", 1.0)
    Nx, Nz = flow.Nx, flow.Nz
    # the EXACT per-cell air mass the run used (scalar for Boussinesq, flat (Nx*Nz,) for
    # anelastic); fall back to surface density if an old result lacks it.
    amc = result.get("air_mass_cell", None)
    if amc is None:
        amc = (P_col[0] / (r_a * T_col[0])) * flow.dx * flow.dz * depth
    amc = np.asarray(amc)
    if amc.ndim == 0:                                 # scalar (Boussinesq)
        air_2d = np.full((Nx, Nz), float(amc))
    elif amc.size == Nx * Nz:                         # flat (Nx*Nz,) (anelastic)
        air_2d = amc.reshape(Nx, Nz)
    else:                                             # (Nz,) profile
        air_2d = np.broadcast_to(amc[None, :], (Nx, Nz))
    varlist = PRESETS.get(fields, None) if isinstance(fields, str) else list(fields)
    if varlist is None:
        varlist = list(fields) if not isinstance(fields, str) else BASIC
    frames = result["frames"]
    out = {v: [] for v in varlist}
    skipped = set()
    want_rad = bool(set(varlist) & set(RAD))
    for f in frames:
        g = _grid_basic(f, flow, air_2d, depth)
        T = f["theta"] * (P_col[None, :] / p0) ** _kap
        avail = dict(g, u=f["u"], w=f["w"], qv=f["qv"] * 1e3, T=T)
        if "rates" in f:                              # process rates (from diagnose=)
            avail.update(f["rates"])
        if want_rad:                                  # per-column radiative properties (Nx,)
            avail.update(_radiation_columns(f, flow, depth))
        for v in varlist:
            if v in avail:
                out[v].append(np.asarray(avail[v]))
            else:
                skipped.add(v)
    arrays = {v: np.array(out[v]) for v in varlist if out[v]}
    times = np.array([f["step"] * dt for f in frames], float)
    zc = (np.arange(flow.Nz) + 0.5) * flow.dz
    xc = (np.arange(flow.Nx) + 0.5) * flow.dx
    if skipped:
        miss = sorted(skipped)
        print("droplab.output: requested but unavailable (need diagnose= in the run?): %s" % miss)
    return arrays, times, zc, xc, P_col


def save_run(result, path, fields="basic", droplets=False):
    """Write the run to a self-describing NetCDF file. Gridded fields are (time, z, x);
    per-column radiation fields are (time, x); base-state pressure P(z) is 1-D. Sounding CAPE/EL
    and run params are stored as attributes. droplets=True also writes the Lagrangian
    super-droplet table (x, z, r, multiplicity, phase) as a ragged (record,) array per frame."""
    import xarray as xr
    arrays, times, zc, xc, P_col = gridded(result, fields)
    flow = result["flow"]
    data_vars = {}
    for v, a in arrays.items():
        units, lname = _ATTRS.get(v, ("", v))
        if a.ndim == 3:                                # gridded (time, Nx, Nz) -> (time, z, x)
            data_vars[v] = (("time", "z", "x"), a.transpose(0, 2, 1), {"units": units, "long_name": lname})
        else:                                          # per-column (time, Nx) -> (time, x)
            data_vars[v] = (("time", "x"), a, {"units": units, "long_name": lname})
    ds = xr.Dataset(data_vars, coords={"time": ("time", times, {"units": "s"}),
                                       "z": ("z", zc, {"units": "m"}),
                                       "x": ("x", xc, {"units": "m"})})
    ds["P"] = ("z", P_col, {"units": "Pa", "long_name": "base-state pressure"})
    ds.attrs.update(model="droplab 2D", ice=int(bool(result.get("ice", False))),
                    dt=float(result.get("dt", 1.0)), depth=float(result.get("depth", 1.0)),
                    nx=int(flow.Nx), nz=int(flow.Nz),
                    surf_precip_kg=float(result.get("surf_precip", 0.0)))
    # CAPE / equilibrium level of the sounding (parcel theory), if the sounding is available
    snd = result.get("sounding", None)
    if snd is not None:
        try:
            from droplab.sounding_diag import parcel_cape
            cape, cin, lfc, el = parcel_cape(snd)
            ds.attrs.update(CAPE_Jkg=round(float(cape), 1), CIN_Jkg=round(float(cin), 1),
                            LFC_m=None if lfc is None else round(float(lfc), 0),
                            EL_m=None if el is None else round(float(el), 0))
        except Exception:
            pass
    if droplets:                                       # Lagrangian super-droplet table (ragged)
        fr = result["frames"]
        rec_t = np.concatenate([np.full(f["x"].shape[0], f["step"] * result.get("dt", 1.0)) for f in fr])
        cat = lambda k: np.concatenate([np.asarray(f[k]) for f in fr])
        phase_cat = (np.concatenate([np.asarray(f.get("phase", np.zeros(f["x"].shape[0], np.int8))) for f in fr]))
        ds_d = xr.Dataset({"sd_time": ("record", rec_t), "sd_x": ("record", cat("x")),
                           "sd_z": ("record", cat("z")), "sd_r_um": ("record", cat("r_um")),
                           "sd_multiplicity": ("record", cat("A")), "sd_phase": ("record", phase_cat)})
        ds = xr.merge([ds, ds_d])
    ds.to_netcdf(path)
    return ds
