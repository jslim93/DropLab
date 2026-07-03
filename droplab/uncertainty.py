"""Microphysics is stochastic and uncertain -- a demonstrator on the parcel model.

Warm-rain initiation is a probabilistic, sensitively-dependent process: which
droplets happen to collide changes whether and when rain forms. The super-droplet
method samples that stochastic collision process, so running the SAME parcel with
different random seeds gives DIFFERENT outcomes. Two distinct things show up, and
this module keeps them separate (conflating them teaches a misconception):

  * intrinsic spread       -- the genuine seed-to-seed variability of the
                              stochastic collision process at a fixed resolution;
  * numerical convergence  -- that spread shrinks as the number of super-droplets
                              n_ptcl grows (better sampling of the same process,
                              approaching the mean-field collection equation).

So `seed_ensemble` shows the spread; `spread_vs_resolution` shows it converging.
Near the warm-rain threshold the outcome stays highly sensitive even when well
resolved -- that sensitivity is physical, not a bug.

Conceptual basis: Dziekan & Pawlowska (2017), ACP -- super-droplet stochastic
coalescence vs the master equation. [CITE-UNVERIFIED: confirm before manuscript.]

Consumes the existing parcel engine only (droplab.timestep_soa.run_soa); adds no
physics. New file -> conflict-free with the physics lanes.
"""
import numpy as np

from droplab.timestep_soa import run_soa


def rain_onset(out, qr_thr=0.01):
    """First (step, height) at which rain water exceeds qr_thr [g/kg]; NaN if never."""
    for t in sorted(out):
        if out[t]["qr"] > qr_thr:
            return t, out[t]["z"]
    return np.nan, np.nan


def run_member(seed, qr_thr=0.01, collect_every=40, nt=1200, **kwargs):
    """One parcel ascent. Returns time series + scalar outcome metrics."""
    collect = tuple(range(collect_every, nt + 1, collect_every))
    out, _ = run_soa(seed=seed, nt=nt, collect=collect, **kwargs)
    ts = sorted(out)
    z = np.array([out[t]["z"] for t in ts])
    qr = np.array([out[t]["qr"] for t in ts])
    qc = np.array([out[t]["qc"] for t in ts])
    NR = np.array([out[t]["NR"] for t in ts])
    on_step, on_z = rain_onset(out, qr_thr)
    return dict(z=z, qr=qr, qc=qc, NR=NR,
                onset_z=on_z, onset_step=on_step,
                final_qr=qr[-1], final_qc=qc[-1], final_NR=NR[-1])


def seed_ensemble(n_members=24, qr_thr=0.01, nt=1200, **kwargs):
    """Run the identical parcel with seeds 0..n_members-1. Returns per-member series,
    scalar-outcome arrays, and spread statistics."""
    members = [run_member(s, qr_thr=qr_thr, nt=nt, **kwargs) for s in range(n_members)]
    onset_z = np.array([m["onset_z"] for m in members])
    final_qr = np.array([m["final_qr"] for m in members])
    final_NR = np.array([m["final_NR"] for m in members])
    fq_mean = float(np.nanmean(final_qr))
    return dict(
        members=members,
        onset_z=onset_z, final_qr=final_qr, final_NR=final_NR,
        onset_z_std=float(np.nanstd(onset_z)),
        final_qr_mean=fq_mean,
        final_qr_cov=float(np.nanstd(final_qr) / fq_mean) if fq_mean > 0 else float("nan"),
        rain_fraction=float(np.mean(np.isfinite(onset_z))),   # fraction of seeds that rained
    )


def spread_vs_resolution(n_ptcl_list, n_members=12, nt=1200, **kwargs):
    """How the seed-to-seed spread shrinks as super-droplet count grows.

    For each n_ptcl, run a seed ensemble and record the spread (rain-onset-height
    std and final-qr coefficient of variation). A decreasing trend is the numerical
    sampling component converging; a residual at large n_ptcl is the physical
    sensitivity near the rain threshold.
    """
    n_ptcl_list = list(n_ptcl_list)
    onset_std, qr_cov = [], []
    for n in n_ptcl_list:
        e = seed_ensemble(n_members=n_members, n_ptcl=n, nt=nt, **kwargs)
        onset_std.append(e["onset_z_std"])
        qr_cov.append(e["final_qr_cov"])
    return dict(n_ptcl=np.array(n_ptcl_list, float),
                onset_z_std=np.array(onset_std),
                final_qr_cov=np.array(qr_cov))
