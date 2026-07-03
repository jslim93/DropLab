"""Toy electrification: pure-function unit tests (charging law, charge-conserving
deposition, Poisson sign convention, breakdown/discharge flash). Integration with the
dynamic model (golden-safety, dipole formation, flash firing) is in
tests/test_electrification_run.py."""
import numpy as np
import pytest

from droplab import electrification as el


def test_charge_reversal_sign():
    # below the reversal temperature graupel charges negative; above it, positive
    assert el.graupel_charge_sign(260.0, 263.15) == -1.0
    assert el.graupel_charge_sign(268.0, 263.15) == 1.0


def _column_setup():
    # a 1-wide, 4-tall column (Nx=1, Nz=4). Graupel (large) in cell z=2, crystals (small) in
    # cell z=1 just below -> the vertical sweep connects them (graupel falls through crystals).
    # cell index = ix*Nz + iz with Nx=1 -> cidx = iz.
    M = np.array([8e-9, 9e-9, 1e-10, 1.2e-10])         # 0,1 graupel ; 2,3 crystal
    A = np.array([1.0, 1.0, 1.0, 1.0])
    phase = np.array([1, 1, 1, 1], dtype=np.int8)
    cidx = np.array([2, 2, 1, 1])                      # graupel at z=2, crystals at z=1
    charge = np.zeros(4)
    Tf = np.array([260.0, 260.0, 260.0, 260.0])        # per-cell T (Nc=4), all in window
    qsc = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    return charge, M, A, phase, cidx, Tf, qsc


def test_deposit_conserves_charge_and_separates_by_size():
    charge, M, A, phase, cidx, Tf, qsc = _column_setup()
    el.deposit_charge(charge, M, A, phase, cidx, 4, Tf, qsc, q_sc_min=1e-5, q_rev_T=263.15,
                      dt=2.0, V_cell=1.0e6, Nx=1, Nz=4, dz=300.0, charge_eff=0.1)
    assert abs(charge.sum()) < 1e-12 * np.abs(charge).sum()   # net zero (sweep is balanced)
    # below reversal: graupel (idx 0,1) negative; crystals (idx 2,3) positive
    assert charge[0] < 0 and charge[1] < 0
    assert charge[2] > 0 and charge[3] > 0


def test_deposit_skips_inactive_cells():
    charge, M, A, phase, cidx, Tf, qsc = _column_setup()
    # no supercooled water -> no charging
    el.deposit_charge(charge, M, A, phase, cidx, 4, Tf, np.zeros(4), 1e-5, 263.15,
                      2.0, 1.0e6, 1, 4, 300.0)
    assert np.all(charge == 0.0)
    # water but too warm (outside the charging window) -> no charging
    el.deposit_charge(charge, M, A, phase, cidx, 4, np.full(4, 290.0), qsc, 1e-5, 263.15,
                      2.0, 1.0e6, 1, 4, 300.0)
    assert np.all(charge == 0.0)


def test_deposit_needs_both_graupel_and_crystal():
    # only crystals (all small) -> no graupel partner -> no charge
    charge = np.zeros(2)
    el.deposit_charge(charge, np.array([1e-10, 1.2e-10]), np.array([1.0, 1.0]),
                      np.array([1, 1], np.int8), np.array([1, 1]), 4, np.full(4, 260.0),
                      np.full(4, 1e-3), 1e-5, 263.15, 2.0, 1.0e6, 1, 4, 300.0)
    assert np.all(charge == 0.0)


def test_poisson_sign_convention():
    # a single positive charge cell -> potential is a POSITIVE maximum there, and the
    # field points radially AWAY from it (down the potential gradient).
    Nx = Nz = 16
    rho_q = np.zeros((Nx, Nz))
    rho_q[8, 8] = 1e-9
    phi = el.solve_potential(rho_q, dx=100.0, dz=100.0, periodic_x=False)
    assert phi[8, 8] == phi.max() and phi.max() > 0.0
    Ex, Ez, Emag = el.efield(phi, 100.0, 100.0, periodic_x=False)
    # just left of the charge the field points in -x (away, toward the wall)
    assert Ex[7, 8] < 0.0 and Ex[9, 8] > 0.0


def test_dbm_leader_propagates_along_potential_to_ground():
    # a potential that decreases monotonically toward the ground row (j=0): the leader
    # initiated up high must propagate DOWN the gradient and reach the ground.
    Nx, Nz = 12, 20
    j = np.arange(Nz)
    phi = np.tile(j.astype(float)[None, :], (Nx, 1))   # phi increases with height
    rng = np.random.default_rng(0)
    cells, edges, grounded = el.dbm_leader(phi, (6, Nz - 1), dx=100.0, dz=100.0, rng=rng,
                                           eta=1.0, max_cells=400)
    assert grounded                                    # reached the ground row
    assert any(j == 0 for (_, j) in cells)
    assert len(edges) == len(cells) - 1                # a tree (one edge per added cell)


def test_flash_below_threshold_is_none():
    charge = np.array([1e-12, -1e-12])
    cidx = np.array([10, 40])
    phi = np.zeros((8, 8)); Emag = np.zeros((8, 8))    # no field -> no breakdown
    rng = np.random.default_rng(0)
    assert el.flash(charge, cidx, 8, 8, phi, Emag, 1.0, 1.0, 100.0, 100.0, rng,
                    E_breakdown=1.5e5) is None


def test_flash_fires_neutralises_and_conserves():
    # a dipole on a small grid; force breakdown with a huge field everywhere
    Nx = Nz = 8
    charge = np.array([5.0, 5.0, -5.0, -5.0])
    cidx = np.array([3 * Nz + 6, 3 * Nz + 6, 3 * Nz + 1, 3 * Nz + 1])   # upper + / lower -
    phi = np.tile(np.arange(Nz).astype(float)[None, :], (Nx, 1))        # gradient to ground
    Emag = np.full((Nx, Nz), 1e7)                      # above threshold everywhere
    rng = np.random.default_rng(1)
    q0 = charge.sum()
    out = el.flash(charge, cidx, Nx, Nz, phi, Emag, 1.0, 1.0, 1000.0, 1000.0, rng,
                   E_breakdown=1.5e5, flash_neutralize=0.7, max_cells=200)
    assert out is not None and "segments" in out and "grounded" in out
    assert abs(charge.sum() - q0) < 1e-12             # discharge conserves net charge
    assert out["segments"].shape[1] == 4              # (x0,z0,x1,z1) edges
