"""Cached run wrappers — the only place the app calls the physics engine.

Every wrapper is decorated with ``st.cache_data`` and keyed by its full argument
list (all hashable: scalars + tuples), so an identical configuration returns
instantly. The wrappers are pure consumers of the validated run functions
(``run_soa``, ``run_flow2d_dynamic``, ``climate_widget.simulate``); they add no
physics. They DO enforce the hard physics couplings so an invalid toggle combo
is impossible to actually run (the UI also disables them, but this is the
authoritative backstop).

The 2-D wrapper returns a trimmed, picklable payload: the collected frames (each
a dict of numpy arrays) plus a small ``meta`` of grid scalars, the derived
metrics, and a stability flag. We deliberately do NOT return the Flow2D object —
the renderer rebuilds a lightweight proxy from ``meta`` — so the cache entry
pickles cleanly.
"""
from __future__ import annotations

import hashlib
import io
import os
import pathlib
import pickle

import numpy as np
import streamlit as st

from droplab.timestep_soa import run_soa
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.climate_widget import figure, _seeding_spec, _BASE
from droplab.climate_diag import (column_optics, optics_from_frame, toa_forcing,
                               cre_from_frame)

from app.ui import presets


# ------------------------------------------------------------------------- #
# Persistent disk cache. Sits UNDER the in-process dicts (_TWOD_CACHE /
# _CLIM_CACHE): on a process-cache miss we check disk, and on a compute we write
# both. Keyed by the SAME config tuples the process caches use (sha1 of repr), so
# a warmed cache makes the five curated demos render instantly on first click and
# survives restarts. Volatile, rebuildable bytes (the in-payload "_gifcache" of
# rendered GIFs) are stripped before pickling. Size-capped with oldest-first
# (mtime) eviction. Location: $DROPLAB_CACHE_DIR or ~/.droplab_cache.
# ------------------------------------------------------------------------- #
_DISK_DIR = pathlib.Path(
    os.environ.get("DROPLAB_CACHE_DIR", pathlib.Path.home() / ".droplab_cache"))
# default 1 GB: the five warmed demos alone are ~240 MB, and a payload is
# 25-50 MB — a 512 MB cap would start evicting demos after a handful of user
# runs, silently breaking "first click is instant".
_DISK_CAP_BYTES = int(os.environ.get("DROPLAB_CACHE_MB", "1024")) * 1024 * 1024
_STRIP_KEYS = ("_gifcache",)          # rebuilt on demand → never persisted


def disk_cache_dir() -> pathlib.Path:
    """The directory used for the persistent cache (created on demand)."""
    return _DISK_DIR


def _disk_path(prefix: str, key) -> pathlib.Path:
    h = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()
    return _DISK_DIR / f"{prefix}_{h}.pkl"


def _disk_load(path: pathlib.Path):
    """Return the pickled payload, or None on any miss / corruption."""
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _disk_evict():
    """Trim the cache dir to the byte cap, deleting oldest (by mtime) first."""
    try:
        files = sorted(_DISK_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    except FileNotFoundError:
        return
    total = sum(p.stat().st_size for p in files)
    while total > _DISK_CAP_BYTES and files:
        victim = files.pop(0)
        try:
            total -= victim.stat().st_size
            victim.unlink()
        except OSError:
            pass


def _disk_store(prefix: str, key, payload: dict):
    """Persist a payload (minus volatile keys) atomically, then evict to cap."""
    try:
        _DISK_DIR.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in payload.items() if k not in _STRIP_KEYS}
        path = _disk_path(prefix, key)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(slim, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        _disk_evict()
    except Exception:
        pass          # a cache is an optimization; never let it break a run


# --- Parcel ---------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_parcel(seed, n_ptcl, nt, dt, T0, P0, RH, w, ascending_mode,
               N_raw, mu_um, sig, kappa, collisions, switch_TICE, eps,
               lambda_ent, ihmd, n_collect=120,
               rh_env=0.2, ent_start=0.0, ent_duration=None, sedi_removal=True):
    """Cached warm-parcel run. Tuple args stay tuples (hashable). Returns the
    dense time-sampled diagnostics dict plus the final (M, A) arrays.
    Entrainment: lambda_ent (strength), rh_env (how dry the entrained air is),
    ent_start/ent_duration in SECONDS (duration None = the whole ascent)."""
    step = max(1, nt // n_collect)
    collect = tuple(range(step, nt + 1, step))
    out, (M, A) = run_soa(
        seed=seed, n_ptcl=n_ptcl, nt=nt, dt=dt, T0=T0, P0=P0, RH=RH, w=w,
        N_raw=N_raw, mu_um=mu_um, sig=sig, kappa=kappa,
        ascending_mode=ascending_mode, collisions=collisions,
        switch_TICE=switch_TICE, eps=eps, lambda_ent=lambda_ent, ihmd=ihmd,
        collect=collect, rh_env=rh_env, ent_start=ent_start,
        ent_duration=ent_duration, sedi_removal=sedi_removal)
    return out, np.asarray(M), np.asarray(A)


# --- 2-D ------------------------------------------------------------------- #
def build_twod_config(scenario, resolution, *, collisions, ice, habit,
                      electrification, freezing_mode, homogeneous, melt,
                      hallett_mossop, N_modes, mu_um, sig, kappa,
                      seed_on, seed_kind, seed_N, seed_r, inject_min,
                      wind_shear, dtheta_bubble,
                      inp_n_cm3, inp_r_um, E_breakdown, charge_eff,
                      nt=None, dt=None):
    """Assemble the run_flow2d_dynamic kwargs, enforcing the hard couplings.

    ``nt`` / ``dt`` set the run length and time step (grid stays from the
    resolution). ``inject_min`` sets when mid-run seeding fires (simulated
    minutes). Kept separate from the cached runner so a smoke test can assert
    the couplings and the seeding timing without running the model.
    """
    cfg = presets.sized_config(scenario, resolution, nt=nt, dt=dt)

    # aerosol (always user-controlled)
    cfg["N_modes"] = tuple(float(x) for x in N_modes)
    cfg["mu_um"] = tuple(float(x) for x in mu_um)
    cfg["sig"] = tuple(float(x) for x in sig)
    cfg["kappa"] = tuple(float(x) for x in kappa)
    cfg["collisions"] = bool(collisions)

    # --- coupling 0 (ice needs a cold/deep scenario), 1 (habit⇒ice), 2 (elec⇒ice) --- #
    # authoritative backstop: a shallow-warm scenario can never run ice, even if a
    # stale disabled toggle passes ice=True (the UI is the first line; this is the wall).
    ice_capable = scenario in presets.ICE_CAPABLE
    allow_elec = electrification and scenario in presets.ELECTRIFY_SCENARIOS
    ice_on = bool((ice or habit or allow_elec) and ice_capable)
    cfg["ice"] = ice_on
    cfg["habit"] = bool(habit and ice_on)
    cfg["electrification"] = bool(allow_elec and ice_on)

    if ice_on:
        cfg["freezing_mode"] = freezing_mode
        cfg["homogeneous"] = bool(homogeneous)
        cfg["melt"] = bool(melt)
        cfg["hallett_mossop"] = bool(hallett_mossop)
        if inp_n_cm3 is not None:
            cfg["inp_n_cm3"] = float(inp_n_cm3)
        if inp_r_um is not None:
            cfg["inp_r_um"] = float(inp_r_um)

    if cfg["electrification"]:
        cfg["E_breakdown"] = float(E_breakdown)
        cfg["charge_eff"] = float(charge_eff)

    # --- coupling 3 (deep_convection ⇒ anelastic) ------------------------- #
    if scenario == "deep_convection":
        cfg["dynamics"] = "anelastic"

    # arctic: stronger initial noise so the (physically gentle, ~0.5-1 m/s) deck
    # circulation spins up within the first ~15 min instead of looking static.
    if scenario == "arctic":
        cfg["pert_amp"] = max(float(cfg.get("pert_amp", 0.1)), 0.25)

    # --- coupling 4 (wind_shear > 0 ⇒ periodic_x + CFL-safe dt) ----------- #
    # set explicitly (incl. 0) so a scenario's baked-in shear can be turned off.
    cfg["wind_shear"] = float(wind_shear or 0.0)
    if cfg["wind_shear"] > 0:
        cfg["periodic_x"] = True
        # CFL guard: the mean wind U(z)=shear*(z-Z/2) peaks at shear*Z/2, so the same
        # slider value that is mild on a 2-km domain is a ~40 m/s jet on a 12-km deep
        # domain — at the deep scenarios' large dt that violates CFL and the run goes
        # NaN. Cap dt so (U_max + convective-w margin)*dt <= 0.4*min(dx,dz), and
        # rescale nt to keep the same simulated time.
        dx = cfg["X"] / cfg["Nx"]
        dz = cfg["Z"] / cfg["Nz"]
        u_max = cfg["wind_shear"] * 0.5 * cfg["Z"]
        w_margin = 15.0 if cfg["Z"] > 8000.0 else 5.0
        dt_max = 0.4 * min(dx, dz) / (u_max + w_margin)
        if cfg["dt"] > dt_max:
            sim_time = cfg["nt"] * cfg["dt"]
            new_dt = max(0.5, int(dt_max * 4) / 4.0)   # tidy quarter-second steps
            cfg["dt"] = new_dt
            cfg["nt"] = max(1, int(round(sim_time / new_dt)))
    if dtheta_bubble is not None and "dtheta_bubble" in cfg:
        cfg["dtheta_bubble"] = float(dtheta_bubble)

    # mid-run seeding (MCB / GCCN / glaciogenic INP), else respect the scenario's own
    # seeding (none). _seeding_spec hardcodes t_inject; override it so the user
    # controls timing.
    if seed_on:
        if seed_kind == "Glaciogenic INP (ice)":
            # direct-ice injection into the cloud layer (phase='ice'), mirroring the
            # Climate page's _inp_seeding_spec — the liquid _seeding_spec would inject
            # CCN and BRIGHTEN the deck, the opposite sign of the glaciation lesson
            # this seed kind teaches.
            z_top = cfg["Z"]
            spec = _inp_seeding_spec(cfg["nt"], cfg["dt"], float(seed_N),
                                     z_lo=0.45 * z_top, z_hi=0.72 * z_top)
        else:
            spec = _seeding_spec(seed_kind, float(seed_N), float(seed_r),
                                 cfg["nt"], cfg["dt"])
        if inject_min is not None:
            spec["t_inject"] = float(inject_min) * 60.0
        cfg["seeding"] = spec
    else:
        cfg["seeding"] = None
    return cfg


# Process-level cache for 2-D runs (deliberately NOT st.cache_data): a plain dict
# lets the FIRST run stream frames live via on_frame, then serve the assembled
# payload instantly on repeats — "watch it compute, then loop" (the live UX).
_TWOD_CACHE = {}
_TWOD_ORDER = []
_TWOD_CAP = 16


# Bump when build_twod_config's mapping or the engine physics changes so stale disk
# entries (e.g. pre-CFL-guard NaN results) can never be served for the same widget args.
_CFG_VERSION = 10


def _twod_key(scenario, resolution, nt, dt, collisions, ice, habit,
              electrification, freezing_mode, homogeneous, melt, hallett_mossop,
              N_modes, mu_um, sig, kappa, seed_on, seed_kind, seed_N, seed_r,
              inject_min, wind_shear, dtheta_bubble, inp_n_cm3, inp_r_um,
              E_breakdown, charge_eff):
    """A hashable key over the full 2-D config (on_frame excluded)."""
    def _o(x):
        return None if x is None else round(float(x), 6)
    return (_CFG_VERSION, scenario, resolution, int(nt), round(float(dt), 6), bool(collisions),
            bool(ice), bool(habit), bool(electrification), freezing_mode,
            bool(homogeneous), bool(melt), bool(hallett_mossop),
            tuple(float(x) for x in N_modes), tuple(float(x) for x in mu_um),
            tuple(float(x) for x in sig), tuple(float(x) for x in kappa),
            bool(seed_on), seed_kind, _o(seed_N), _o(seed_r), _o(inject_min),
            _o(wind_shear), _o(dtheta_bubble), _o(inp_n_cm3), _o(inp_r_um),
            _o(E_breakdown), _o(charge_eff))


def twod_is_cached(*args):
    """True if this exact 2-D config is already available — in this process OR in
    the persistent disk cache (so warmed demos skip the live-run path after a
    restart)."""
    key = _twod_key(*args)
    return key in _TWOD_CACHE or _disk_path("twod", key).exists()


def _twod_store(key, payload):
    """Insert into the process cache with a simple FIFO cap; returns the payload."""
    _TWOD_CACHE[key] = payload
    _TWOD_ORDER.append(key)
    while len(_TWOD_ORDER) > _TWOD_CAP:
        _TWOD_CACHE.pop(_TWOD_ORDER.pop(0), None)
    return payload


def run_twod(scenario, resolution, nt, dt, collisions, ice, habit, electrification,
             freezing_mode, homogeneous, melt, hallett_mossop,
             N_modes, mu_um, sig, kappa,
             seed_on, seed_kind, seed_N, seed_r, inject_min,
             wind_shear, dtheta_bubble, inp_n_cm3, inp_r_um,
             E_breakdown, charge_eff, on_frame=None):
    """Run one 2-D config and return a trimmed, picklable payload.

    Process-cached by the full config (incl. nt/dt/inject_min) → an identical
    config returns instantly. ``on_frame(step, total, frame, flow)`` (optional)
    is forwarded to the engine so the UI can render frames live on the FIRST
    (uncached) run. Keys: ``frames`` (list of frame dicts), ``meta`` (grid +
    toggle scalars), ``metrics`` (reff_um, albedo, cloud_fraction),
    ``surf_precip``, ``qc_max``, ``n_flashes`` and ``unstable``.
    """
    key = _twod_key(scenario, resolution, nt, dt, collisions, ice, habit,
                    electrification, freezing_mode, homogeneous, melt,
                    hallett_mossop, N_modes, mu_um, sig, kappa, seed_on,
                    seed_kind, seed_N, seed_r, inject_min, wind_shear,
                    dtheta_bubble, inp_n_cm3, inp_r_um, E_breakdown, charge_eff)
    if key in _TWOD_CACHE:
        return _TWOD_CACHE[key]
    disk = _disk_load(_disk_path("twod", key))
    if disk is not None:
        return _twod_store(key, disk)

    cfg = build_twod_config(
        scenario, resolution, collisions=collisions, ice=ice, habit=habit,
        electrification=electrification, freezing_mode=freezing_mode,
        homogeneous=homogeneous, melt=melt, hallett_mossop=hallett_mossop,
        N_modes=N_modes, mu_um=mu_um, sig=sig, kappa=kappa,
        seed_on=seed_on, seed_kind=seed_kind, seed_N=seed_N, seed_r=seed_r,
        inject_min=inject_min, wind_shear=wind_shear, dtheta_bubble=dtheta_bubble,
        inp_n_cm3=inp_n_cm3, inp_r_um=inp_r_um,
        E_breakdown=E_breakdown, charge_eff=charge_eff, nt=nt, dt=dt)

    out = run_flow2d_dynamic(on_frame=on_frame, **cfg)

    if not (np.isfinite(out["theta"]).all() and np.isfinite(out["M"]).all()):
        bad = {"unstable": True, "meta": {"scenario": scenario}}
        # in-process only — an unstable result must NOT be persisted to disk, or a
        # later code/config fix would keep serving the stale NaN from the cache.
        return _twod_store(key, bad)

    flow = out["flow"]
    frames = out["frames"]
    qc_max = max(0.5, max(float(f["qc"].max()) for f in frames))
    o = column_optics(out["M"], out["A"], out["x"], out["z"], flow)
    n_flashes = sum(len(f.get("flashes", [])) for f in frames)

    meta = dict(X=flow.X, Z=flow.Z, Nx=flow.Nx, Nz=flow.Nz, dx=flow.dx,
                dz=flow.dz, depth=float(out.get("depth", 1.0)), dt=cfg["dt"],
                scenario=scenario,
                resolution=resolution, ice=cfg["ice"], habit=cfg["habit"],
                electrification=cfg["electrification"],
                anelastic=(cfg.get("dynamics") == "anelastic"),
                seed_on=bool(seed_on),
                P_col=out.get("P_col"))     # level pressure, for the T background field
    payload = {
        "unstable": False,
        "frames": frames,
        "meta": meta,
        "metrics": dict(reff_um=o["reff_mean"] * 1e6, albedo=o["albedo_mean"],
                        cloud_fraction=o["cloud_fraction"]),
        "surf_precip": float(out["surf_precip"]),
        "qc_max": qc_max,
        "n_flashes": int(n_flashes),
    }
    _disk_store("twod", key, payload)
    return _twod_store(key, payload)


# --- Climate --------------------------------------------------------------- #
_CLIM_X, _CLIM_Z = 3200.0, 1200.0


_CLIM_CACHE = {}
_CLIM_ORDER = []
_CLIM_CAP = 12


# Climate background decks. Each entry: engine kwargs describing the environment.
# dycoms = the classic marine Sc sunshade; bomex = shallow trade cumulus (surface-flux
# driven); arctic = MOSAiC mixed-phase deck (ice on) where GLACIOGENIC INP seeding
# demonstrates the cold-regime intervention (ice grows at the liquid's expense -> the
# deck dims: the opposite sign to MCB).
from examples.cloud_cases import CASES as _CASES
from droplab.soundings import BOMEX as _BOMEX_SND

_CLIM_BG = {
    # DYCOMS forcing + low scalar diffusion, UNIFIED with CASES['dycoms'] (single
    # source). The deck needs both to survive the 2-h Climate runs: nu_scalar 0.2
    # keeps the sharp inversion from diffusing away (the old 1.5 dried the BL out
    # by ~45 min), and DYCOMS_FORCING resupplies moisture as SUBCLOUD-DISTRIBUTED
    # qls/tls (+ subsidence) -- NOT bottom-cell H/LE, which pooled moisture at the
    # surface and turned the deck into surface-rooted cumulus (the 0be799a revert).
    "DYCOMS stratocumulus": dict(
        extra=dict(forcing=_CASES["dycoms"]["forcing"],
                   nu_scalar=_CASES["dycoms"]["nu_scalar"]),
        X=_CLIM_X, Z=_CLIM_Z, ice=False),
    "BOMEX cumulus": dict(
        extra=dict(sounding=_CASES["bomex"]["sounding"],
                   forcing=_CASES["bomex"]["forcing"], nu=14, nu_scalar=1.5),
        X=4800.0, Z=3000.0, ice=False, drop=("rad_cool",)),
    "Arctic mixed-phase": dict(
        extra=dict(sounding=_CASES["arctic"]["sounding"],
                   rad_cool=_CASES["arctic"]["rad_cool"], ice=True,
                   freezing_mode="abifm", inp_n_cm3=_CASES["arctic"]["inp_n_cm3"],
                   inp_r_um=_CASES["arctic"]["inp_r_um"],
                   inp_sigma=_CASES["arctic"]["inp_sigma"],
                   inp_frac=_CASES["arctic"]["inp_frac"], nu=6, nu_scalar=1.0),
        # drop the DYCOMS forcing _BASE now carries: MOSAiC runs unforced
        X=4800.0, Z=2600.0, ice=True, drop=("rad_cool", "forcing")),
}


def _inp_seeding_spec(nt, dt, seed_N, z_lo, z_hi):
    """Glaciogenic (direct-ice) injection: INP-born ice embryos into the cloud layer
    (engine spec phase='ice'; same canonical keys as tests/test_ice.py).

    The seed AMOUNT (N_cm3) scales the injected super-droplets' WEIGHT FACTOR
    (multiplicity), not their COUNT: _inject_aerosol splits N_cm3*V over a FIXED
    n_super, so a heavier seeding is a higher weight per SD, not more SDs — the run
    cost stays flat regardless of how much you seed. n_super is kept modest (2000) so
    the injection itself is cheap; the ice it makes is finely enough sampled at that
    weight for the intervention demo."""
    return dict(t_inject=max(50.0, 0.25 * nt * dt), x_frac=(0.0, 1.0),
                z_lo=z_lo, z_hi=z_hi, N_cm3=seed_N, r_um=2.0, r_wet_um=2.0,
                kappa=0.6, n_super=2000, phase="ice")


def _clim_key(background_N, ihmd, seed_on, seed_kind, seed_N, seed_r, inject_min,
              nt, Nx, Nz, n_super, dt=1.0, background="DYCOMS stratocumulus"):
    def _o(x):
        return None if x is None else round(float(x), 6)
    return (_CFG_VERSION, background, round(float(background_N), 4), round(float(ihmd), 4),
            bool(seed_on), seed_kind, _o(seed_N), _o(seed_r), _o(inject_min),
            int(nt), int(Nx), int(Nz), int(n_super), round(float(dt), 6))


def climate_is_cached(*args):
    """True if this exact climate config is already available — in this process OR
    in the persistent disk cache."""
    key = _clim_key(*args)
    return key in _CLIM_CACHE or _disk_path("clim", key).exists()


def _clim_remember(key, payload):
    """Insert into the climate process cache with a simple FIFO cap."""
    _CLIM_CACHE[key] = payload
    _CLIM_ORDER.append(key)
    while len(_CLIM_ORDER) > _CLIM_CAP:
        _CLIM_CACHE.pop(_CLIM_ORDER.pop(0), None)
    return payload


def run_climate(background_N, ihmd, seed_on, seed_kind, seed_N, seed_r,
                inject_min, nt, Nx, Nz, n_super, dt=1.0,
                background="DYCOMS stratocumulus", on_frame=None):
    """Stratocumulus run for the climate twin, process-cached so the FIRST run can
    stream live via ``on_frame`` and repeats are instant.

    Bypasses ``climate_widget.simulate`` (which sets collect_every=100000 → one
    frame) and calls ``run_flow2d_dynamic`` directly with a small collect_every so
    the deck can be ANIMATED. Identical physics — only the output cadence changes.
    Returns the final-deck PNG, the collected ``frames`` + grid ``meta`` (for the
    synced animation), and per-frame ``ts`` of the MCB metrics that matter —
    droplet number N_d (cm⁻³), cloud albedo and short-wave CRE (W/m²) — plus the
    headline scalars. (No q_c series: q_c is not the climate-relevant metric.)
    """
    key = _clim_key(background_N, ihmd, seed_on, seed_kind, seed_N, seed_r,
                    inject_min, nt, Nx, Nz, n_super, dt, background)
    if key in _CLIM_CACHE:
        return _CLIM_CACHE[key]
    disk = _disk_load(_disk_path("clim", key))
    if disk is not None:
        return _clim_remember(key, disk)

    bg = _CLIM_BG.get(background, _CLIM_BG["DYCOMS stratocumulus"])
    spec = None
    if seed_on:
        if seed_kind == "Glaciogenic INP (ice)":
            z_top = bg["Z"]
            spec = _inp_seeding_spec(nt, dt, float(seed_N),
                                     z_lo=0.45 * z_top, z_hi=0.72 * z_top)
        else:
            spec = _seeding_spec(seed_kind, float(seed_N), float(seed_r), nt, dt)
        if inject_min is not None:
            spec["t_inject"] = float(inject_min) * 60.0
    base = {**_BASE, "collect_every": max(2, nt // 30)}
    for k in bg.get("drop", ()):
        base.pop(k, None)
    base.update(bg["extra"])
    res = run_flow2d_dynamic(nt=nt, dt=dt, Nx=Nx, Nz=Nz, X=bg["X"], Z=bg["Z"],
                             n_super=n_super, N_modes=(float(background_N),),
                             ihmd=float(ihmd), seeding=spec, seed=3,
                             on_frame=on_frame, **base)
    # NaN guard, mirroring run_twod: a blown-up deck must surface as "unstable",
    # not render NaN plots/metrics silently.
    if not (np.isfinite(res["theta"]).all() and np.isfinite(res["M"]).all()):
        bad = {"unstable": True}
        # in-process only (see run_twod): never persist NaN results
        return _clim_remember(key, bad)
    flow = res["flow"]
    depth = float(res.get("depth", 1.0))
    o = column_optics(res["M"], res["A"], res["x"], res["z"], flow)
    summary = dict(reff_um=o["reff_mean"] * 1e6, albedo=o["albedo_mean"],
                   precip_kg=res["surf_precip"], droplet_number=float(res["A"].sum()))
    title = f"N0={background_N:.0f}/cc  IHMD={ihmd:.1f}" + (
        f"  +{seed_kind}" if seed_on else "  (no seeding)")

    import matplotlib.pyplot as plt
    fig = figure(res, summary, title=title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # per-frame metrics: N_d (cm^-3), albedo, r_eff, precip, and the full cloud
    # radiative effect split — SW (cooling), LW (warming), net — in W/m^2.
    fr = res["frames"]
    T_col = res.get("T_col")
    # daylight factor for the SW CRE: the Arctic MOSAiC deck is polar night, so SW is
    # near zero there and the LONG-WAVE effect dominates (dimming a polar liquid deck
    # lets long-wave escape -> cooling); marine decks get a representative daytime sun.
    mu0 = 0.1 if background == "Arctic mixed-phase" else 0.5
    cell_cm3 = flow.dx * flow.dz * depth * 1e6
    t, nc, alb, cre, reff, prc = [], [], [], [], [], []
    swc, lwc, netc = [], [], []
    for f in fr:
        of = optics_from_frame(f, flow)
        a = of["albedo_mean"]
        alb.append(a)
        cre.append(toa_forcing(a))
        reff.append(float(of["reff_mean"]) * 1e6)
        prc.append(float(f.get("surf_precip", 0.0)))
        rad = cre_from_frame(f, flow, T_col, mu0=mu0) if T_col is not None else None
        swc.append(rad["swcre_mean"] if rad else 0.0)
        lwc.append(rad["lwcre_mean"] if rad else 0.0)
        netc.append(rad["net_mean"] if rad else 0.0)
        cloudy = f["r_um"] > 1.0
        if "phase" in f:                     # mixed-phase runs: N_d counts DROPLETS,
            cloudy = cloudy & (f["phase"] == 0)   # not ice crystals (qc is liquid-only)
        ncell = max(1, int((f["qc"] > 0.01).sum()))
        nc.append(float(f["A"][cloudy].sum()) / (ncell * cell_cm3))
        t.append(float(f["step"]) * dt)
    ts = dict(t=t, nc=nc, albedo=alb, cre=cre, reff=reff, precip=prc,
              swcre=swc, lwcre=lwc, netcre=netc)
    meta = dict(X=flow.X, Z=flow.Z, Nx=flow.Nx, Nz=flow.Nz, dx=flow.dx,
                dz=flow.dz, depth=depth, dt=dt, seed_on=bool(seed_on))
    payload = {
        "png": buf.getvalue(),
        "frames": fr,
        "meta": meta,
        "n_frames": len(fr),
        "ts": ts,
        "reff_um": summary["reff_um"],
        "albedo": summary["albedo"],
        "precip_kg": summary["precip_kg"],
        "droplet_number": summary["droplet_number"],
        "albedo_mean": o["albedo_mean"],
    }
    _disk_store("clim", key, payload)
    return _clim_remember(key, payload)


def climate_forcing(d_albedo):
    """Idealized TOA shortwave forcing (W/m^2) from an albedo change — thin
    re-export so pages don't import climate_diag directly."""
    return toa_forcing(d_albedo)


# --- curated-demo argument reproduction ------------------------------------ #
# One source of truth for the EXACT run_twod / run_climate args a curated demo
# produces from the UI defaults, so scripts/warm_demo_cache.py warms the SAME
# cache keys the demo buttons hit. These MUST mirror the default widgets in
# modes.render_twod / render_climate; if those defaults change, update here.
def demo_twod_args(demo):
    """Positional run_twod args for a 2-D curated demo (mirrors render_twod)."""
    from app.ui import controls
    sc = demo["scenario"]
    m = presets.SCENARIOS[sc]
    base = presets.base_config(sc)
    t = demo["toggles"]
    ice = bool(t.get("ice", False))
    # mirror the UI COUPLINGS, not the raw toggles: microphysics_panel forces
    # habit = ice and electrification = ice & ELECTRIFY_SCENARIOS (controls.py) —
    # a demo key built from independent toggles warmed a key no button ever hits.
    habit = ice
    elec = bool(ice and sc in presets.ELECTRIFY_SCENARIOS)
    dt = float(m["dt_default"])
    nt = presets.default_nt(sc)
    seed = controls.SEED_DEFAULTS["MCB sea-salt"]
    bubble = {"idealized", "congestus", "deep_cold", "deep_convection"}
    dtheta = float(base.get("dtheta_bubble", 2.5)) if sc in bubble else None
    # UI defaults: the INP slider default is hardcoded 1.0 in controls.py; the
    # aerosol N default is the scenario's own CASES value (modes.py) — match both.
    inp_n = 1.0 if ice else None
    inp_r = float(base.get("inp_r_um", 3.0)) if ice else None
    _bN = float(base["N_modes"][0])
    return (sc, "quick", nt, dt, bool(t.get("collisions", True)), ice, habit, elec,
            "abifm", True, True, True,
            (_bN,), (0.08,), (2.0,), (0.6,),
            False, "MCB sea-salt", float(seed["N"]), float(seed["r"]),
            round(0.25 * m["default_min"], 1), 0.0, dtheta, inp_n, inp_r,
            400.0, 0.3)


def demo_climate_args(seed_on=True):
    """Positional run_climate args for the default MCB deck. Derives the run
    length and injection time from the presets constants render_climate uses, so
    the warmed key CANNOT drift from what the demo button actually runs."""
    from app.ui import controls
    seed = controls.SEED_DEFAULTS["MCB sea-salt"]
    nt = presets.CLIMATE_RUN_STEPS[presets.CLIMATE_RUN_DEFAULT]
    inject_min = round(presets.CLIMATE_INJECT_FRAC * (nt / 60.0), 1)
    return (200.0, 0.0, bool(seed_on), "MCB sea-salt", float(seed["N"]),
            float(seed["r"]), inject_min, nt, 64, 40, 30000)
