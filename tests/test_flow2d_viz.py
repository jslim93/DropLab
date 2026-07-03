"""Smoke test: the 2D animation builds and a frame renders without error."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from droplab.flow2d_driver import run_flow2d
from droplab.flow2d_dynamic import run_flow2d_dynamic
from droplab.flow2d_viz import animate, draw_frame


def test_animation_builds():
    result = run_flow2d(nt=40, dt=2.0, Nx=16, Nz=16, n_super=4000,
                        collisions=False, collect_every=10)
    assert len(result["frames"]) >= 2
    fig, anim = animate(result, field="qc")
    assert anim is not None
    # render the last frame onto a fresh axis (exercises draw_frame end-to-end)
    ax = plt.subplots()[1]
    sc = draw_frame(ax, result["flow"], result["frames"][-1], field="supersat")
    assert sc is not None


def test_quiver_overlay_renders():
    """The wind-vector overlay draws on a dynamic frame (which carries u, w)."""
    out = run_flow2d_dynamic(nt=30, dt=2.0, Nx=16, Nz=16, n_super=3000,
                             collisions=False, collect_every=10, periodic_x=True,
                             wind_shear=2.5e-3)
    fr = out["frames"][-1]
    assert "u" in fr and "w" in fr
    ax = plt.subplots()[1]
    # both overlay styles, and the full-wind (no Reynolds decomposition) path, must render
    assert draw_frame(ax, out["flow"], fr, "qc", quiver=True,
                      quiver_style="streamlines") is not None
    assert draw_frame(ax, out["flow"], fr, "qc", quiver=True,
                      quiver_style="arrows") is not None
    assert draw_frame(ax, out["flow"], fr, "qc", quiver=True,
                      quiver_perturb=False) is not None
