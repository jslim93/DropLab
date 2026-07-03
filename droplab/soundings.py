"""Reference thermodynamic soundings (z, potential temperature, vapour).

Embedded here (not read from SAM) so the educational package is self-contained.
These are the standard GCSS/LES intercomparison initial profiles; the dynamic 2D
model interpolates theta(z) and q_v(z) from them, so the cloud is capped by a
REAL inversion instead of a hand-tuned one.

Each sounding: z [m], theta [K], qv [g/kg].
"""

# Shallow trade cumulus — Siebesma et al. (2003) BOMEX intercomparison.
# Trade inversion near 1500-2000 m caps the cumulus; q_v drops sharply above it.
BOMEX = dict(
    name="BOMEX (shallow cumulus)",
    z=[0.0, 520.0, 700.0, 1480.0, 2000.0, 4000.0],
    theta=[298.70, 298.70, 299.39, 302.40, 308.20, 315.50],
    qv=[17.00, 16.30, 14.98, 10.70, 4.20, 1.80],
)

# Cumulus congestus — deeper, weaker cap (a moister, less stable mid-troposphere
# than BOMEX so clouds reach ~4-5 km before the upper stable layer).
CONGESTUS = dict(
    name="congestus (deep cumulus)",
    z=[0.0, 600.0, 1000.0, 2000.0, 3500.0, 5000.0, 7000.0],
    theta=[298.5, 298.7, 300.0, 303.0, 307.5, 313.0, 322.0],
    qv=[17.5, 16.5, 14.5, 10.5, 6.0, 3.0, 1.2],
)

# Marine stratocumulus — DYCOMS-II RF01 (Stevens et al. 2005). A well-mixed
# cloud-topped boundary layer capped by a SHARP, strong inversion at ~840 m; the
# deck is driven by cloud-top longwave radiative cooling (see radiation below).
# This is the canonical marine-cloud-brightening (MCB) target.
DYCOMS = dict(
    name="DYCOMS-II RF01 (marine stratocumulus)",
    z=[0.0, 840.0, 880.0, 1200.0, 1500.0],
    theta=[289.0, 289.0, 297.5, 298.6, 299.3],     # sharp inversion jump at ~840 m
    qv=[9.0, 9.0, 1.5, 1.3, 1.1],                   # moist BL, dry free troposphere
)

# Precipitating trade-wind cumulus — representative of RICO (Rain In Cumulus over
# the Ocean; van Zanten et al. 2011). Like BOMEX but DEEPER and moister, with a
# higher, weaker trade inversion (~2-4 km), so the cumulus grow tall enough that
# collision-coalescence makes drizzle — the warm-rain counterpart to the
# non-precipitating BOMEX. Pair with LOW (maritime) aerosol in the run to let the
# big drops form. Values are a representative composite, not a bit-exact case spec.
RICO = dict(
    name="RICO (precipitating trade cumulus)",
    z=[0.0, 740.0, 2000.0, 3000.0, 4000.0],
    theta=[297.9, 297.9, 306.5, 311.5, 317.0],   # well-mixed sub-cloud, then warming
    qv=[16.0, 13.8, 4.4, 2.7, 1.8],              # moist BL, drying free troposphere
)

# Radiation fog — a STABLE, near-saturated nocturnal boundary layer. theta INCREASES
# with height (strongly stable -> no convection), and the surface air is ~97% RH at
# ~10 C. Run with surface_cool < 0 (ground radiating to the clear night sky): the
# lowest air chills to its dew point and fog condenses AT the surface, then grows
# upward. The mirror image of stratocumulus (which cools at cloud TOP). Use a shallow
# domain (Z~600 m) and T0=283 K to match.
FOG = dict(
    name="radiation fog (stable, near-saturated surface)",
    z=[0.0, 100.0, 300.0, 1000.0],
    theta=[283.0, 283.3, 285.0, 291.0],   # stable inversion from the ground up
    qv=[7.5, 7.3, 4.0, 2.0],              # ~97% RH at the surface, drying aloft
)

# Arctic mixed-phase stratocumulus, ISDAC (Barrow, Apr 2008; Ovchinnikov et al. 2014
# LES intercomparison). Authentic well-mixed BL theta (~265 K, cloud top ~820 m,
# inversion to ~269 K); the BL qv is moistened to near-saturation so a supercooled
# (~-9 C) cloud forms -> immersion freezing + WBF glaciation when ice=True.
ISDAC = dict(
    name="ISDAC Arctic mixed-phase stratocumulus",
    z=[0.0, 400.0, 820.0, 850.0, 1000.0, 1200.0, 2000.0],
    theta=[263.4, 265.0, 265.0, 268.6, 270.7, 271.9, 274.3],
    qv=[2.3, 2.2, 2.2, 0.9, 0.6, 0.5, 0.4],     # moist BL (cloud), dry above inversion
)

# MOSAiC 2019-11-01 (real radiosonde, from SAM6-LCM MOSAIC/OBS_files/20191101): a
# polar-night, deep (~1650 m), COLD mixed-phase boundary layer — surface ~255 K
# (-18 C), cloud top ~-24 C, inversion ~1800 m. theta is verbatim from the obs; qv is
# the observed profile scaled ~+15% so the thin cloud reliably forms in the idealised
# spin-up (the raw obs cloud layer is slightly sub-saturated, maintained in reality by
# processes the box doesn't fully resolve). The canonical PERSISTENT mixed-phase day:
# at realistic INP (~0.01-0.2 cm^-3) the liquid deck persists while a few large ice
# crystals snow out.
MOSAIC = dict(
    name="MOSAiC 2019-11-01 Arctic persistent mixed-phase stratocumulus",
    z=[0.0, 150.0, 410.0, 700.0, 1200.0, 1650.0, 1850.0, 2400.0],
    theta=[254.8, 254.9, 261.1, 262.5, 262.7, 263.4, 269.4, 273.7],
    qv=[0.83, 0.81, 1.07, 0.91, 0.86, 0.83, 0.39, 0.57],   # obs +15% (cloud layer near sat)
)

# Deep COLD convective storm (wintertime / lake-effect-like): a cold but conditionally
# unstable column. The surface is near freezing (~270 K) and the whole cloud is sub-freezing,
# so condensate glaciates aloft (immersion + homogeneous freezing) and the ice grows and
# falls as snow. Deep enough (to ~7 km) for cold tops.
DEEP_COLD = dict(
    name="deep cold convective storm (snow)",
    z=[0.0, 800.0, 1600.0, 3200.0, 5000.0, 7000.0],
    theta=[270.0, 271.0, 273.0, 281.0, 294.0, 312.0],
    qv=[3.6, 3.3, 2.5, 1.1, 0.4, 0.12],
)

# Idealized upper-tropospheric CIRRUS layer (~10 km, 250 hPa). Potential temperatures are
# high (theta ~ 345 K) but the low pressure makes the actual temperature ~225-232 K, i.e.
# below the ~235 K homogeneous-freezing threshold: lifted moist air forms droplets that
# freeze homogeneously (no ice nucleus needed) into cirrus ice. qv is small (cold-air
# moisture) but ice-supersaturated.
CIRRUS = dict(
    name="idealized cirrus (homogeneous freezing)",
    z=[0.0, 400.0, 800.0, 1200.0, 1800.0],
    theta=[344.0, 344.5, 346.0, 350.0, 356.0],
    qv=[0.42, 0.40, 0.30, 0.12, 0.05],
)

# High-CAPE tropical profile for DEEP CONVECTION (anelastic core). A warm moist base under a
# conditionally unstable mid-troposphere and a cold (~ -40 C) upper level: driven by sustained
# surface fluxes it builds a cumulonimbus that reaches the tropopause and glaciates into an ice
# anvil. (Needs dynamics="anelastic"; the Boussinesq core caps convection shallow.)
DEEP_CAPE = dict(
    name="deep convection (high-CAPE tropical)",
    z=[0.0, 1000.0, 2500.0, 4500.0, 6500.0, 9000.0, 10500.0],
    theta=[300.0, 301.0, 303.0, 308.0, 316.0, 330.0, 342.0],
    qv=[16.0, 13.0, 9.0, 5.0, 2.5, 0.6, 0.3],
)

# Single-cell CUMULONIMBUS profile: a moist boundary layer under a strong capping inversion
# (theta jumps 301->305 across ~1 km = CIN) and a DRY, conditionally-unstable free troposphere.
# The cap suppresses widespread convection, so a single strong trigger punches through as ONE
# isolated tower; the dry environment keeps cloud confined to the saturated updraft, so the
# textbook tower + spreading glaciated anvil stands against clear sky (pair with RH0~0.5, a
# strong narrow bubble, a wide domain, NO surface forcing, dynamics="anelastic").
CUMULONIMBUS = dict(
    name="single-cell cumulonimbus (capped CAPE)",
    z=[0.0, 800.0, 1200.0, 4000.0, 7000.0, 10000.0, 11500.0, 13500.0, 16000.0],
    theta=[300.0, 301.0, 305.0, 312.0, 322.0, 338.0, 350.0, 364.0, 382.0],
    qv=[15.0, 13.0, 5.0, 2.5, 0.9, 0.3, 0.10, 0.02, 0.005],   # dry, sub-ice-sat stratosphere
)

# Weisman-Klemp (1982) analytic sounding -- THE standard idealized deep-convection profile, used
# across the cloud-modelling literature for benchmarking. theta(z)=theta0+(theta_tr-theta0)
# (z/z_tr)^1.25 below the tropopause z_tr=12 km (theta_tr=343 K), an exponential stratosphere
# above; RH=1-0.75(z/z_tr)^1.25 capped at a well-mixed surface qv0=14 g/kg. These parameters
# give CAPE ~ 2120 J/kg and an equilibrium level ~ 12 km -- the reference the deep-convection
# validation checks the simulated storm against (tests/test_dcc_validation.py).
WEISMAN_KLEMP = dict(
    name="Weisman-Klemp 1982 (CAPE~2120 J/kg, EL~12 km)",
    z=[0.0, 500.0, 1000.0, 2000.0, 3500.0, 5500.0, 7500.0, 9500.0, 11500.0, 13000.0, 14000.0],
    theta=[300.0, 300.8, 301.9, 304.6, 309.2, 316.2, 323.9, 332.1, 340.8, 359.1, 375.9],
    qv=[14.0, 14.0, 14.0, 10.22, 5.45, 2.14, 0.73, 0.2, 0.04, 0.03, 0.04],
)

# LBA Amazon deep convection (subsampled from the SAM6.10.10 LBA/snd reference). Warm tropical
# base (+24 C) to a very cold tropopause; the textbook warm-rain -> mixed-phase -> glaciation
# profile. Pair with LBA_SFC_FORCING (SAM LBA surface fluxes) and dynamics="anelastic".
LBA = dict(
    name="LBA Amazon deep convection (SAM6 reference)",
    z=[0.0, 1000.0, 2086.0, 4197.0, 6001.0, 8170.0, 10878.0, 12000.0],
    theta=[297.6, 303.3, 308.5, 318.3, 327.7, 336.3, 344.7, 347.5],
    qv=[18.5, 14.6, 11.0, 6.7, 3.5, 1.0, 0.21, 0.1],
)

SOUNDINGS = {"bomex": BOMEX, "congestus": CONGESTUS, "dycoms": DYCOMS,
             "rico": RICO, "fog": FOG, "isdac": ISDAC, "mosaic": MOSAIC, "cirrus": CIRRUS,
             "deep_cold": DEEP_COLD, "deep_cape": DEEP_CAPE, "cumulonimbus": CUMULONIMBUS,
             "weisman_klemp": WEISMAN_KLEMP, "lba": LBA}


# Sustained surface fluxes that drive deep convection (no large-scale advective tendencies,
# like SAM's LBA: warm-ocean/land Bowen ratio). The continuous surface heating + moistening
# builds CAPE until a cumulonimbus fires and stays fed -- unlike a single decaying bubble.
DEEP_CONVECTION_FORCING = dict(
    H=250.0,           # surface sensible heat flux (W/m^2)
    LE=450.0,          # surface latent heat flux (W/m^2)
    z=[0.0, 12000.0],
    tls=[0.0, 0.0], qls=[0.0, 0.0], wls=[0.0, 0.0],   # no large-scale forcing
)


# DYCOMS-II RF01 cloud-top radiative-cooling parameters (Stevens et al. 2005):
# net LW flux F(z) = F0*exp(-kappa*LWP_above) + F1*exp(-kappa*LWP_below); the flux
# DIVERGENCE puts strong cooling in a thin layer at cloud top -> dense air sinks ->
# the Rayleigh-Benard-like cellular convection that keeps stratocumulus alive.
DYCOMS_RADIATION = dict(F0=70.0, F1=22.0, kappa=85.0)   # W/m^2, W/m^2, m^2/kg


# Large-scale + surface forcing that SUSTAINS BOMEX shallow cumulus (Siebesma
# et al. 2003). Without it a single triggered thermal is ~30x too vigorous; with
# it the convective velocity scale is the observed w* ~ 0.7 m/s, so cumulus stay
# gentle and the trade inversion caps them. Surface fluxes are fixed; the large-
# scale tendencies (radiative+advective cooling, drying, subsidence) are profiles.
BOMEX_FORCING = dict(
    H=9.46,            # surface sensible heat flux (W/m^2)
    LE=153.4,          # surface latent heat flux (W/m^2)
    z=[0.0, 300.0, 500.0, 1500.0, 2100.0, 2500.0, 4000.0],
    tls=[-2.315e-5, -2.315e-5, -2.315e-5, -2.315e-5, -1.389e-5, 0.0, 0.0],  # K/s
    qls=[-1.2e-8, -1.2e-8, 0.0, 0.0, 0.0, 0.0, 0.0],                         # kg/kg/s
    wls=[0.0, -0.0013, -0.0021, -0.0065, 0.0, 0.0, 0.0],                     # m/s (subsidence)
)


# Large-scale + surface forcing sustaining RICO precipitating trade cumulus (van
# Zanten et al. 2011). Small surface sensible / large latent heat flux (warm ocean),
# weak subsidence, ~2.5 K/day radiative cooling through the lower troposphere. Pair
# with the RICO sounding + low maritime aerosol to get drizzling cumulus.
RICO_FORCING = dict(
    H=5.2,             # surface sensible heat flux (W/m^2)
    LE=150.0,          # surface latent heat flux (W/m^2)
    z=[0.0, 740.0, 2000.0, 3000.0, 4000.0],
    tls=[-2.5 / 86400.0, -2.5 / 86400.0, -2.5 / 86400.0, -1.0 / 86400.0, 0.0],  # K/s (radiative cooling)
    qls=[0.0, 0.0, 0.0, 0.0, 0.0],                                              # kg/kg/s
    wls=[0.0, -0.0028, -0.0050, -0.0040, 0.0],                                  # m/s (subsidence)
)
