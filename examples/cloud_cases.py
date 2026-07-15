"""Gallery of 2D cloud regimes — run any case and render it.

    python -m examples.cloud_cases rico      # (or: bomex congestus dycoms fog diurnal)
    python -m examples.cloud_cases all       # render every case to /tmp

Each entry is the validated run_flow2d_dynamic config for that regime. The three new
ones:
  rico    — precipitating trade-wind cumulus (deeper/moister than BOMEX, drizzles)
  fog     — radiation fog: a SURFACE cloud from nocturnal ground cooling (base ~0 m)
  diurnal — continental cumulus diurnal cycle: clear morning -> afternoon cumulus ->
            evening decay (compressed 4-hour 'day')
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.flow2d_viz import draw_frame
from droplab.soundings import (BOMEX, CONGESTUS, DYCOMS, RICO, FOG, ISDAC, MOSAIC, CIRRUS, DEEP_COLD,
                             DEEP_CAPE, CUMULONIMBUS, BOMEX_FORCING, RICO_FORCING,
                             DYCOMS_RADIATION, DYCOMS_FORCING, DEEP_CONVECTION_FORCING)

_COMMON = dict(pert_amp=0.1, collect_every=400, seed=3)

CASES = {
    "bomex": dict(nt=1500, dt=2.0, Nx=96, Nz=72, X=4800, Z=3000, n_super=50000,
                  sounding=BOMEX, forcing=BOMEX_FORCING, N_modes=(200.,), nu=14,
                  nu_scalar=1.5, collisions=False, sediment=True, **_COMMON),
    "congestus": dict(nt=1800, dt=2.0, Nx=96, Nz=96, X=6000, Z=7000, n_super=60000,
                      sounding=CONGESTUS, forcing=BOMEX_FORCING, N_modes=(200.,), nu=16,
                      nu_scalar=1.5, collisions=True, switch_TICE=True, eps=0.01,
                      sediment=True, **_COMMON),
    # nu_scalar 0.2 (not the 1.5 the convective cases use): the deck lives or dies by
    # the SHARP 40-m inversion, and explicit scalar diffusion at 1.5 m^2/s erodes it at
    # ~8x the real entrainment rate -- the BL then dries out and the deck starves by
    # ~45-60 min. DYCOMS_FORCING (subcloud-distributed LHF/SHF + subsidence; see
    # droplab.soundings for why NOT bottom-cell fluxes) balances the budget so the deck
    # persists >= 2 h on the 2-D, quick and Climate grids (validated 2026-07-14).
    "dycoms": dict(nt=1500, dt=1.0, Nx=96, Nz=48, X=4800, Z=1200, n_super=60000,
                   sounding=DYCOMS, rad_cool=DYCOMS_RADIATION, forcing=DYCOMS_FORCING,
                   periodic_x=True, N_modes=(250.,), nu=6, nu_scalar=0.2, collisions=True,
                   switch_TICE=True, eps=0.01, sediment=True, **_COMMON),
    "rico": dict(nt=1200, dt=1.5, Nx=96, Nz=72, X=4800, Z=4000, n_super=60000,
                 sounding=RICO, forcing=RICO_FORCING, N_modes=(70.,), nu=12,
                 nu_scalar=1.5, collisions=True, switch_TICE=True, eps=0.01,
                 sediment=True, **_COMMON),
    "fog": dict(nt=1800, dt=1.0, Nx=64, Nz=48, X=2000, Z=600, n_super=40000,
                sounding=FOG, T0=283.0, RH0=0.99, surface_cool=-6.0e-3, periodic_x=True,
                N_modes=(200.,), nu=4, nu_scalar=0.2, collisions=False, sediment=False,
                b_max=0.05, omega_max=0.02, **_COMMON),
    "diurnal": dict(nt=4200, dt=2.0, Nx=96, Nz=72, X=4800, Z=3000, n_super=50000,
                    sounding=BOMEX, forcing=BOMEX_FORCING, diurnal_period=14400.0,
                    N_modes=(200.,), nu=14, nu_scalar=1.5, collisions=False,
                    sediment=True, **_COMMON),
    "shear": dict(nt=600, dt=2.0, Nx=120, Nz=72, X=6000, Z=3000, n_super=50000,
                  periodic_x=True, RH0=0.93, z_bl=600.0, z_inv=1900.0, dtheta_inv=4.0,
                  dtheta_bubble=2.5, bubble_z=500.0, bubble_r=500.0, wind_shear=2.5e-3,
                  N_modes=(200.,), nu=16, nu_scalar=1.5, collisions=False, sediment=True,
                  b_max=0.18, omega_max=0.05, **_COMMON),
    # Real MOSAiC 2019-11-01 Arctic PERSISTENT mixed-phase deck (polar-night cold,
    # ~-18 C surface / ~-24 C cloud top). ABIFM (Knopf & Alpert 2013) immersion freezing.
    # INP parameters are the SAM6-LCM MOSAIC/prm values verbatim: n_ice=1/L
    # (inp_n_cm3=0.001), rm_ice=0.37 um geometric-mean INP RADIUS, sigma_ice=2.55, and
    # frac_ice=0.5 (LES-style: INP spread over half the super-droplets at low weight,
    # inp_frac). ABIFM (c,m)=(-1.35, 22.62) natural dust, also from that prm.
    #
    # The small, broadly-distributed INP is what makes the deck PERSIST: freezing area
    # ~ r^2, so at rm=0.37 um only the large tail of the lognormal freezes, and slowly
    # (t_half ~ hours, not the ~2 min a 4 um core gives). The reservoir barely depletes
    # (~85% of INP still present after 3 h), so ice sustains at a low quasi-steady IWP
    # (~3-5 g/kg, verified to 6 h) while the liquid deck persists -- the real persistent
    # mixed-phase regime, with NO INP replenishment needed. A too-large INP core froze
    # everything in minutes (a burst that snowed out and killed the ice); a too-high
    # concentration (>=100/L, finely sampled) glaciates the whole deck. Deep domain holds
    # the ~1650 m BL.
    "arctic": dict(nt=1500, dt=1.0, Nx=96, Nz=64, X=4800, Z=2600, n_super=60000,
                   sounding=MOSAIC, rad_cool=DYCOMS_RADIATION, periodic_x=True,
                   N_modes=(60.,), nu=6, nu_scalar=1.0, collisions=True,
                   switch_TICE=True, eps=0.01, sediment=True, ice=True,
                   freezing_mode="abifm", inp_n_cm3=0.001, inp_r_um=0.37,
                   inp_sigma=2.55, inp_species="default", inp_frac=0.5, **_COMMON),
    # Idealized CIRRUS: the domain IS the upper-tropospheric layer (P0=250 hPa, ~10 km),
    # no surface boundary layer. An ice-supersaturated cold layer (T~228 K, below the
    # homogeneous-freezing threshold) glaciates by HOMOGENEOUS freezing with NO ice nuclei
    # (inp_n_cm3=0) -- in-situ cirrus, not surface convection (no bubble, gentle dynamics).
    "cirrus": dict(nt=900, dt=1.0, Nx=64, Nz=48, X=3000, Z=1800, n_super=40000,
                   sounding=CIRRUS, T0=228.0, P0=2.5e4, periodic_x=True,
                   z_bl=1800.0, RH_top=0.9, z_inv=2000.0, dtheta_inv=0.0, gamma_theta=0.0,
                   N_modes=(50.,), nu=8, nu_scalar=0.5, collisions=False,
                   switch_TICE=True, eps=0.01, sediment=True, ice=True,
                   freezing_mode="abifm", inp_n_cm3=0.0, homogeneous=True,
                   dtheta_bubble=0.0, b_max=0.02, omega_max=0.01, **_COMMON),
    # Deep COLD convective storm (snow): a bubble triggers deep convection in a cold,
    # conditionally-unstable column (~270 K surface, tops to ~7 km / -28 C). The whole cloud
    # is sub-freezing, so condensate glaciates aloft (immersion + homogeneous freezing) and
    # the ice grows by deposition and sediments as snow. This is the test bed for riming.
    "deep_cold": dict(nt=1800, dt=2.0, Nx=96, Nz=96, X=6000, Z=7000, n_super=60000,
                      sounding=DEEP_COLD, periodic_x=True, N_modes=(150.,), nu=16,
                      nu_scalar=1.5, collisions=True, switch_TICE=True, eps=0.01,
                      sediment=True, ice=True, freezing_mode="abifm", inp_n_cm3=0.5,
                      inp_r_um=3.0, inp_sigma=1.5, inp_species="default", homogeneous=True,
                      dtheta_bubble=2.5, bubble_z=600.0, bubble_r=600.0, **_COMMON),
    # DEEP CONVECTION -- a single-cell CUMULONIMBUS (needs the anelastic core). A capping
    # inversion + dry free troposphere confine convection to ONE plume triggered by a strong
    # bubble: it rises ~10 km as an isolated tower in clear air and glaciates into a spreading
    # ice anvil that snows out -- the iconic thunderstorm shape. The Boussinesq core caps
    # convection shallow (~2.6 km). (No surface forcing: the cap would suppress it anyway; the
    # bubble is the trigger. A wide domain leaves clear, subsiding air around the tower.)
    "deep_convection": dict(nt=2000, dt=4.0, Nx=140, Nz=128, X=21000, Z=16000, n_super=42000,
                            sounding=CUMULONIMBUS, dynamics="anelastic", periodic_x=True,
                            N_modes=(150.,), nu=16, nu_scalar=1.5, collisions=True,
                            switch_TICE=True, eps=0.01, sediment=True, ice=True,
                            freezing_mode="abifm", inp_n_cm3=0.5, inp_r_um=3.0, inp_sigma=1.5,
                            homogeneous=True, RH0=0.5, b_max=0.6, omega_max=0.15,
                            sponge_frac=0.28, sponge_tau=200.0,
                            dtheta_bubble=5.0, bubble_z=600.0, bubble_r=1200.0, **_COMMON),
}


def render(name, path=None, quiver=False, quiver_style="streamlines", **overrides):
    cfg = dict(CASES[name]); cfg.update(overrides)
    out = run_flow2d_dynamic(**cfg)
    fr = out["frames"]; flow = out["flow"]
    pick = [fr[len(fr) // 4], fr[len(fr) // 2], fr[-1]]
    r_max = 20.0 if name == "fog" else 200.0
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, f in zip(ax, pick):
        draw_frame(a, flow, f, "qc", vmax=max(0.5, f["qc"].max()), r_max=r_max,
                   show_aerosol=(name != "fog"), quiver=quiver,
                   quiver_style=quiver_style)
        a.set_title(f"{name}  t={f['step'] * CASES[name]['dt']:.0f} s")
    fig.suptitle(f"{name} — {out['frames'][-1]['qc'].max():.1f} g/kg, "
                 f"surf precip {out['surf_precip']:.1e} kg")
    fig.tight_layout()
    path = path or f"/tmp/case_{name}.png"
    fig.savefig(path, dpi=100)
    print(f"{name}: qc_max={out['frames'][-1]['qc'].max():.2f} g/kg  "
          f"surf_precip={out['surf_precip']:.2e} kg  ->  {path}")
    return out


def bergeron(name="arctic", path=None):
    """Liquid vs ice water path over time — the canonical mixed-phase lesson: the
    Bergeron hand-off as supercooled liquid converts to ice."""
    cfg = dict(CASES[name]); cfg["collect_every"] = max(5, cfg["nt"] // 40)
    out = run_flow2d_dynamic(**cfg); fr = out["frames"]
    t = np.array([f["step"] for f in fr]) * cfg["dt"]
    lwp = np.array([f["q_liquid"].sum() for f in fr])
    iwp = np.array([f["q_ice"].sum() for f in fr])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, lwp, "C0", label="liquid (q$_c$ sum)")
    ax.plot(t, iwp, "C3", label="ice (q$_i$ sum)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("column water (g/kg, domain sum)")
    ax.set_title(f"{name}: Bergeron hand-off"); ax.legend()
    path = path or f"/tmp/bergeron_{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"{name} Bergeron -> {path}  (liquid {lwp[0]:.1f}->{lwp[-1]:.1f}, "
          f"ice {iwp[0]:.1f}->{iwp[-1]:.1f})")
    return out


def animate(name, path=None, fps=8, quiver=False, quiver_style="streamlines",
            **overrides):
    """Run a case and save a GIF of the cloud evolving (good for shear/diurnal).
    quiver=True overlays a faint background wind hint (quiver_style "streamlines"
    or "arrows"). Pass overrides (e.g. wind_shear=2.5e-3, periodic_x=True) to
    compose effects onto a base case."""
    cfg = dict(CASES[name]); cfg.update(overrides)
    cfg["collect_every"] = max(5, cfg["nt"] // 40)          # ~40 frames
    out = run_flow2d_dynamic(**cfg)
    fr = out["frames"]; flow = out["flow"]
    vmax = max(0.5, float(np.percentile([f["qc"].max() for f in fr], 90)))
    r_max = 20.0 if name == "fog" else 80.0
    fig, ax = plt.subplots(figsize=(8, 4.2))

    def upd(k):
        draw_frame(ax, flow, fr[k], "qc", vmax=vmax, r_max=r_max,
                   show_aerosol=(name != "fog"), quiver=quiver,
                   quiver_style=quiver_style)
        ax.set_title(f"{name}   t={fr[k]['step'] * cfg['dt']:.0f} s")

    anim = animation.FuncAnimation(fig, upd, frames=len(fr), interval=1000 / fps)
    path = path or f"/tmp/case_{name}.gif"
    anim.save(path, writer="pillow", fps=fps)
    print(f"{name} animation ({len(fr)} frames){' +quiver' if quiver else ''} -> {path}")


def main():
    flags = {"--gif", "--quiver", "--arrows", "--streamlines"}
    args = [a for a in sys.argv[1:] if a not in flags]
    gif = "--gif" in sys.argv
    quiver = "--quiver" in sys.argv
    style = "arrows" if "--arrows" in sys.argv else "streamlines"
    arg = args[0] if args else "rico"
    names = list(CASES) if arg == "all" else [arg]
    for n in names:
        if gif:
            animate(n, quiver=quiver, quiver_style=style)
        else:
            render(n, quiver=quiver, quiver_style=style)


if __name__ == "__main__":
    main()
