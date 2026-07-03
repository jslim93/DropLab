"""Visual identity for the DropLab sandbox — the "cirrus" design language.

One place owns the palette, the typography, and the small structural devices
(headers, the mode console cards, the "what am I looking at" caption) so every
mode looks like one app. Pure presentation: no physics, no run logic here.

The palette is atmospheric-optics, not a template default: storm-ink text on a
cirrus-white field, bolt-blue as the single primary accent, with ice-cyan and
ember used sparingly to encode phase (liquid vs ice) and warmth (rain). The
dark-sky lightning/habit panels in 2D mode are deliberate dramatic insets
against this light shell.
"""
from __future__ import annotations

import streamlit as st

# --- palette -------------------------------------------------------------- #
INK = "#0C1626"        # storm ink (text)
INK_SOFT = "#46566B"   # muted captions / eyebrows
PAPER = "#F6F8FB"      # cirrus white (page)
CARD = "#FFFFFF"       # raised surface
LINE = "#D8E0EC"       # hairline rules
BOLT = "#2D6BE0"       # primary accent — the lightning blue
ICE = "#0FB5C4"        # ice-cyan (frozen phase, cold scenarios)
EMBER = "#E8743B"      # warm-rain / climate ember
PLATE = "#3B6FB5"      # crystal habit: plate (oblate)
COLUMN = "#C0504D"     # crystal habit: column (prolate)
NIGHT = "#070B16"      # dark-sky inset background

# Each mode owns one accent so the console reads at a glance.
MODE_ACCENT = {
    "parcel": ICE,
    "twod": BOLT,
    "climate": EMBER,
    "lecture": "#6B5BD2",
}


def in_streamlit() -> bool:
    """True only inside a live Streamlit script run.

    Lets page files guard their render call so importing a page module (e.g. in
    a smoke test) does not execute the full Streamlit script. Uses the runtime
    context probe; any failure means "not running under Streamlit".
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"]  {{ font-family: 'Inter', system-ui, sans-serif; }}
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}

/* headings carry the personality */
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -0.01em; color: {INK}; }}

/* data reads as instrument numerals */
[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {INK}; }}
[data-testid="stMetricLabel"] p {{ color: {INK_SOFT}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}

/* the mode header band */
.dl-kicker {{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:0.18em;
             text-transform:uppercase; color:{INK_SOFT}; margin-bottom:0.15rem; }}
.dl-title {{ font-family:'Space Grotesk',sans-serif; font-size:2.1rem; font-weight:700;
            line-height:1.05; margin:0 0 0.25rem 0; color:{INK}; }}
.dl-lede {{ color:{INK_SOFT}; font-size:1.02rem; max-width:60ch; margin:0; }}
.dl-rule {{ height:3px; width:64px; border-radius:3px; margin:0.6rem 0 1.1rem 0; }}

/* the home console cards */
.dl-card {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:1.05rem 1.15rem 1.1rem;
           height:100%; box-shadow:0 1px 2px rgba(12,22,38,0.04); }}
.dl-card .top {{ height:4px; width:40px; border-radius:4px; margin-bottom:0.7rem; }}
.dl-card .eye {{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.16em;
               text-transform:uppercase; color:{INK_SOFT}; }}
.dl-card .name {{ font-family:'Space Grotesk',sans-serif; font-size:1.18rem; font-weight:600;
                 margin:0.1rem 0 0.35rem; color:{INK}; }}
.dl-card .job {{ color:{INK_SOFT}; font-size:0.92rem; line-height:1.35; }}

/* the "what am I looking at" caption */
.dl-look {{ background:{PAPER}; border-left:3px solid {BOLT}; border-radius:0 8px 8px 0;
           padding:0.55rem 0.85rem; color:{INK_SOFT}; font-size:0.9rem; margin:0.2rem 0 0.8rem; }}

/* tidy the auto sidebar nav */
[data-testid="stSidebarNav"] {{ background:transparent; }}
section[data-testid="stSidebar"] {{ border-right:1px solid {LINE}; }}

/* primary button = the bolt */
.stButton>button[kind="primary"] {{ background:{BOLT}; border:0; font-weight:600; }}
</style>
"""


def apply(page_title: str, icon: str = "⛅") -> None:
    """Set page config + inject the shared CSS. Call once at the top of a page."""
    st.set_page_config(page_title=f"{page_title} · DropLab", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


def header(kicker: str, title: str, lede: str, accent: str = BOLT) -> None:
    """The consistent mode-header band: a mono kicker, a display title, a one-line
    lede, and a short coloured rule in the mode's accent."""
    st.markdown(
        f"<div class='dl-kicker'>{kicker}</div>"
        f"<div class='dl-title'>{title}</div>"
        f"<p class='dl-lede'>{lede}</p>"
        f"<div class='dl-rule' style='background:{accent}'></div>",
        unsafe_allow_html=True,
    )


def whatami(text: str) -> None:
    """A short 'what am I looking at' caption rendered as an accented note."""
    st.markdown(f"<div class='dl-look'>{text}</div>", unsafe_allow_html=True)


def app_version() -> str:
    """The installed package version, or a fallback when running from source."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("droplab")
        except PackageNotFoundError:
            return "1.0.0 (source)"
    except Exception:
        return "1.0.0"


def footer() -> None:
    """A small, consistent version footer for the bottom of every page."""
    st.markdown(
        f"<div style='margin-top:2.2rem;padding-top:0.6rem;border-top:1px solid {LINE};"
        f"color:{INK_SOFT};font-family:JetBrains Mono,monospace;font-size:0.72rem;'>"
        f"DropLab v{app_version()} · a pure consumer of the validated droplab "
        f"engine — no new physics in the interface.</div>",
        unsafe_allow_html=True)


def about() -> None:
    """Help / About — what DropLab is, version, how to run, and the honest note
    on stopping a long run. Rendered as an expander (e.g. on the home page)."""
    with st.expander("ℹ️ About DropLab & help"):
        st.markdown(
            f"""
**DropLab** is a droplet-resolving open laboratory: a browser sandbox over a
validated Lagrangian super-droplet cloud model. Four instruments share one lab —
a **Parcel** microscope, a **2-D** cloud you can watch (ice, crystal habit,
electrification, deep convection), a **Climate**-intervention thermostat, and
guided **Lecture** lessons. The interface only *consumes* the physics engine; it
adds no new physics.

**Version:** `{app_version()}`

**Run it:** `streamlit run app/Home.py` (or `droplab-app` if installed with
`pip install -e .[app]`). For instant demos, warm the cache once with
`python scripts/warm_demo_cache.py`.

**Stopping a long run:** a live 2-D/Climate run streams frames as it computes.
Because Streamlit executes one script per session, the most reliable way to
abort mid-run is the app's built-in **Stop** control in the top-right toolbar
(or press **C**) — the live view yields to it at each frame. Changing any control
also supersedes an in-progress run on the next rerun. Finished runs are cached,
so re-viewing is instant.
""")
