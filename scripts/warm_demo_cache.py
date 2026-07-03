"""Precompute the five curated demos into the persistent disk cache.

Run once after install so the demo buttons render INSTANTLY on first click
(and survive restarts):

    python scripts/warm_demo_cache.py

It reproduces the EXACT configs the demo buttons launch (via
``cache.demo_twod_args`` / ``cache.demo_climate_args`` — the single source of
truth that mirrors the UI defaults), computes each, and writes it to the disk
cache at $DROPLAB_CACHE_DIR or ~/.droplab_cache. Re-running is cheap: already
warmed configs are served from disk.

Pure consumer of the droplab engine — no physics here.
"""
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.ui import cache, presets


def main():
    print(f"Warming DropLab demo cache → {cache.disk_cache_dir()}")
    t_all = time.time()
    for demo in presets.DEMOS:
        label = demo["title"]
        t0 = time.time()
        if demo["page"] == "2D":
            args = cache.demo_twod_args(demo)
            res = cache.run_twod(*args)
            note = ("unstable" if res.get("unstable")
                    else f"qc_max={res['qc_max']:.2f} frames={len(res['frames'])}")
        else:  # the climate (MCB) demo: warm the seeded deck + its unseeded twin
            cache.run_climate(*cache.demo_climate_args(seed_on=True))
            twin = cache.run_climate(*cache.demo_climate_args(seed_on=False))
            note = f"albedo={twin['albedo']:.2f} (+twin)"
        print(f"  ✓ {label:32s} {time.time() - t0:5.1f}s  {note}")
    print(f"Done in {time.time() - t_all:.1f}s. "
          f"Cache dir: {cache.disk_cache_dir()}")


if __name__ == "__main__":
    main()
