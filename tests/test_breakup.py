import math
import numpy as np
from droplab.collision import _inverse_erf


def test_inverse_erf_inverts_erf():
    for y in [-0.9, -0.3, 0.0, 0.25, 0.7, 0.95]:
        x = _inverse_erf(y)
        assert abs(math.erf(x) - y) < 1e-4, f"erf(ierf({y}))={math.erf(x)}"


def test_inverse_erf_endpoints_finite():
    assert _inverse_erf(0.0) == 0.0
    assert _inverse_erf(0.999) > 0 and np.isfinite(_inverse_erf(0.999))
    assert _inverse_erf(-0.999) < 0 and np.isfinite(_inverse_erf(-0.999))


def test_breakup_energy_matches_E_S09_weber():
    # We computed in _breakup_energy must match the We inside E_S09
    # (E_S09 = exp(-1.15*We)).
    from droplab.collision import _breakup_energy, E_S09
    from droplab.parameters import rho_liq
    R_m, R_n, v_r, T = 1.0e-3, 0.3e-3, 5.0, 283.0   # 1mm vs 0.3mm
    d_S, d_L, ga, CKE, We, CW = _breakup_energy(R_m, R_n, v_r, T)
    assert d_L == 2.0 * R_m and d_S == 2.0 * R_n
    assert abs(ga - d_L / d_S) < 1e-12
    es09 = E_S09(R_m, R_n, v_r, rho_liq, T)
    assert abs(math.exp(-1.15 * We) - es09) < 1e-9
    assert abs(CW - CKE * We) < 1e-30


def _mk(M, A, Ns=None, kappa=0.6):
    from droplab.micro_particle import particles
    p = particles(0)
    p.M, p.A = M, A
    p.Ns = (0.01 * M) if Ns is None else Ns
    p.kappa = kappa
    return p


def _total_M(parts):
    return sum(p.M for p in parts)


def test_breakup_conserves_water_and_solute():
    from droplab.collision import liquid_update_breakup
    from droplab.parameters import rho_liq
    import numpy as np
    np.random.seed(0)
    # a 1.4mm drop (A=10) hitting a 0.5mm drop (A=100): energetic -> breakup
    r_big, r_sml = 1.4e-3, 0.5e-3
    M_big = 4/3*np.pi*rho_liq*r_big**3 * 10
    M_sml = 4/3*np.pi*rho_liq*r_sml**3 * 100
    p_big, p_sml = _mk(M_big, 10), _mk(M_sml, 100)
    M0 = p_big.M + p_sml.M
    Ns0 = p_big.Ns + p_sml.Ns
    frags = liquid_update_breakup(p_sml, p_big, 283.0)   # (n=smaller A, m=larger A)
    after = [p_sml, p_big] + frags
    assert abs(_total_M(after) - M0) / M0 < 1e-10, "water not conserved"
    assert abs(sum(p.Ns for p in after) - Ns0) / Ns0 < 1e-9, "solute not conserved"


def test_breakup_caps_drop_size_and_makes_fragments():
    from droplab.collision import liquid_update_breakup
    from droplab.parameters import rho_liq
    import numpy as np
    np.random.seed(1)
    r_big, r_sml = 1.6e-3, 0.6e-3
    p_big = _mk(4/3*np.pi*rho_liq*r_big**3 * 5, 5)
    p_sml = _mk(4/3*np.pi*rho_liq*r_sml**3 * 50, 50)
    frags = liquid_update_breakup(p_sml, p_big, 283.0)
    assert len(frags) >= 1, "energetic collision must produce fragments"
    # every resulting super-droplet has A>0 and finite per-drop radius
    for p in [p_sml, p_big] + frags:
        if p.A > 0:
            r = (p.M / p.A / (4/3*np.pi*rho_liq))**(1/3)
            assert np.isfinite(r) and r > 0


def test_breakup_negative_residual_fallback(monkeypatch):
    """Force the negative-residual fallback branch (modes 1-3 over-claim the
    available water): modes 1&2 are dropped and exactly one ~d_S mode-3 drop
    results, with total water still conserved."""
    import droplab.collision as C
    from droplab.collision import liquid_update_breakup
    from droplab.parameters import rho_liq, pi
    import numpy as np
    # N draws: random()=0.0 makes (random < frac) true -> ceil(nr) (max N).
    monkeypatch.setattr(np.random, "random", lambda *a, **k: 0.0)
    # Inflate every fragment diameter so modes 1-3 over-claim particle_n's water.
    monkeypatch.setattr(C, "_frag_diameter_lognormal", lambda mu, s, u: 5.0e-3)
    monkeypatch.setattr(C, "_frag_diameter_normal", lambda mu, s, u: 5.0e-3)

    r_big, r_sml = 1.6e-3, 0.6e-3
    p_big = _mk(4/3*np.pi*rho_liq*r_big**3 * 5, 5)
    p_sml = _mk(4/3*np.pi*rho_liq*r_sml**3 * 50, 50)
    # d_S from the ORIGINAL pre-transfer per-drop radii (smaller of the two).
    R_n0 = (p_sml.M / p_sml.A / (4/3*pi*rho_liq))**(1/3)
    R_m0 = (p_big.M / p_big.A / (4/3*pi*rho_liq))**(1/3)
    d_S = 2.0 * min(R_n0, R_m0)
    M0 = p_big.M + p_sml.M

    frags = liquid_update_breakup(p_sml, p_big, 283.0)
    after = [p_sml, p_big] + frags

    # (a) water conserved across {p_n, p_m, fragments}
    assert abs(_total_M(after) - M0) / M0 < 1e-10, "water not conserved in fallback"
    # (b) modes 1&2 dropped: exactly one fragment remains, a single ~d_S drop
    assert len(frags) == 1, f"fallback must yield exactly one fragment, got {len(frags)}"
    frag = frags[0]
    assert frag.A == p_sml.A, "fallback mode-3 fragment has A = N3*A_n = A_n"
    # per-drop diameter of the fallback fragment equals d_S
    d_frag = (frag.M / frag.A / (pi / 6.0 * rho_liq))**(1/3)
    assert abs(d_frag - d_S) / d_S < 1e-9, f"fallback drop diameter {d_frag} != d_S {d_S}"


def test_fragments_merge_into_nearest_keeps_count_and_mass():
    """Breakup fragments are redistributed onto the existing super-droplets (no new
    ones), so the count is UNCHANGED and mass/number/solute are conserved."""
    from droplab.collision import _merge_fragments_into_nearest
    from droplab.parameters import rho_liq
    import numpy as np
    # a population spanning small..large per-drop sizes
    parts = [_mk(4/3*np.pi*rho_liq*r**3 * 100, 100) for r in
             (40e-6, 100e-6, 250e-6, 600e-6)]
    M0, A0, Ns0, n0 = (sum(p.M for p in parts), sum(p.A for p in parts),
                       sum(p.Ns for p in parts), len(parts))
    # two fragments: one ~100um-ish, one ~600um-ish
    frags = [_mk(110e-6, 50), _mk(580e-6, 30)]
    Mf, Af, Nsf = (sum(p.M for p in frags), sum(p.A for p in frags),
                   sum(p.Ns for p in frags))
    _merge_fragments_into_nearest(parts, frags)
    assert len(parts) == n0, "count must not change (no new super-droplets)"
    assert abs(sum(p.M for p in parts) - (M0 + Mf)) / (M0 + Mf) < 1e-12, "mass not conserved"
    assert abs(sum(p.A for p in parts) - (A0 + Af)) < 1e-6, "number not conserved"
    assert abs(sum(p.Ns for p in parts) - (Ns0 + Nsf)) / (Ns0 + Nsf) < 1e-12, "solute not conserved"


def test_fragment_lands_in_nearest_size_droplet():
    """A fragment is merged into the super-droplet whose per-drop size is closest."""
    from droplab.collision import _merge_fragments_into_nearest
    from droplab.parameters import rho_liq
    import numpy as np
    small = _mk(4/3*np.pi*rho_liq*(50e-6)**3 * 100, 100)
    large = _mk(4/3*np.pi*rho_liq*(600e-6)**3 * 100, 100)
    parts = [small, large]
    A_large_0 = large.A
    A_small_0 = small.A
    # a ~580um fragment must land in the large (600um) droplet, not the small one
    _merge_fragments_into_nearest(parts, [_mk(580e-6, 25)])
    assert large.A > A_large_0, "fragment did not merge into the nearest (large) droplet"
    assert small.A == A_small_0, "fragment wrongly merged into the far (small) droplet"


def test_breakup_keeps_superdroplet_count_bounded():
    """End-to-end: a collision box with breakup ON must not grow the super-droplet
    count (the nearest-merge redistribution keeps it fixed, no explosion)."""
    from droplab.collision import collection
    from droplab.parameters import rho_liq
    import numpy as np
    np.random.seed(5)
    rng = np.random.default_rng(5)
    r = np.exp(rng.uniform(np.log(300e-6), np.log(2000e-6), 400))
    A = rng.integers(500_000, 5_000_000, 400).astype(float)
    parts = [_mk(r[i], A[i]) for i in range(400)]
    n0 = len(parts)
    for _ in range(40):
        parts, *_ = collection(2.0, parts, 1.0, rho_liq, 1e5, 283.0, 0., 0., 0.,
                               False, 500., 1e9, 0., switch_breakup=True)
    assert len(parts) <= n0, f"count grew under breakup: {len(parts)} > {n0}"


# ---------------------------------------------------------------------------
# Task 5 tests: determine_collision returns 6-tuple with check_breakup
# ---------------------------------------------------------------------------

def test_determine_collision_returns_check_breakup():
    from droplab.collision import determine_collision
    from droplab.parameters import rho_liq
    import numpy as np
    p1 = _mk(4/3*np.pi*rho_liq*(0.5e-3)**3 * 100, 100)
    p2 = _mk(4/3*np.pi*rho_liq*(1.2e-3)**3 * 10, 10)
    out = determine_collision(1.0, p1, p2, 1.0, rho_liq, 1.0e5, 283.0, 1, 2,
                              switch_breakup=False)
    assert len(out) == 6, "determine_collision must now return 6 values"
    assert out[5] is False, "check_breakup must be False when switch off"


def test_breakup_fires_for_energetic_pair_when_on():
    from droplab.collision import determine_collision
    from droplab.parameters import rho_liq
    import numpy as np
    np.random.seed(2)
    fired = 0
    # Use half_length=1, nptcl=10000 so the LSM scaling factor gives p_break~0.5
    # (necessary to overcome PARCEL_AIR_MASS=1e6 m^3 in a unit test)
    for _ in range(200):
        p1 = _mk(4/3*np.pi*rho_liq*(0.6e-3)**3 * 100, 100)
        p2 = _mk(4/3*np.pi*rho_liq*(1.5e-3)**3 * 10, 10)
        out = determine_collision(2.0, p1, p2, 1.0, rho_liq, 1.0e5, 283.0, 1, 10000,
                                  switch_breakup=True)
        if out[5]:
            fired += 1
    # per-call rate ~0.5 over 200 trials -> expect ~100; >20 guards against a
    # future regression that silently drops the breakup rate.
    assert fired > 20, "breakup fired far less than expected for energetic pairs"


# ---------------------------------------------------------------------------
# Task 6 tests: collection wires breakup end-to-end
# ---------------------------------------------------------------------------

def _uniform_drops(n, r, A, rho_liq):
    import numpy as np
    return [_mk(4/3*np.pi*rho_liq*r**3 * A, A) for _ in range(n)]


def test_collection_breakup_off_is_noop():
    """switch_breakup=False must reproduce the coalescence-only result exactly."""
    from droplab.collision import collection
    from droplab.parameters import rho_liq
    import numpy as np
    args = dict(rho_parcel=1.0, rho_liq=rho_liq, p_env=1.0e5, T_parcel=283.0,
                acc_ts=0.0, aut_ts=0.0, precip_ts=0.0, sedi_removal=False,
                z_parcel=500.0, max_z=1000.0, w_parcel=0.0)
    np.random.seed(7)
    a = collection(2.0, _uniform_drops(200, 30e-6, 1000, rho_liq), **args)
    np.random.seed(7)
    b = collection(2.0, _uniform_drops(200, 30e-6, 1000, rho_liq), **args, switch_breakup=False)
    assert len(a[0]) == len(b[0])
    assert abs(sum(p.M for p in a[0]) - sum(p.M for p in b[0])) < 1e-20


def test_collection_breakup_conserves_water_and_caps_count():
    from droplab.collision import collection
    from droplab.parameters import rho_liq
    import numpy as np
    np.random.seed(3)
    # mix of big rain drops + small drops so breakup fires and adds fragments
    parts = (_uniform_drops(50, 1.3e-3, 10, rho_liq)
             + _uniform_drops(150, 0.4e-3, 200, rho_liq))
    M0 = sum(p.M for p in parts)
    n0 = len(parts)
    out = collection(2.0, parts, 1.0, rho_liq, 1.0e5, 283.0, 0.0, 0.0, 0.0,
                     False, 500.0, 1000.0, 0.0, switch_breakup=True)
    plist = out[0]
    assert abs(sum(p.M for p in plist) - M0) / M0 < 1e-9, "water not conserved"
    assert len(plist) <= int(1.1 * n0) + 1, f"count not capped: {len(plist)} vs {n0}"
