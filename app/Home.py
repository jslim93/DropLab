"""DropLab sandbox — entrypoint.

Launch with:

    streamlit run app/Home.py

One app, four modes (Parcel / 2-D / Climate / Lecture) selectable from the
sidebar nav or the console cards on this landing page. A pure consumer of the
validated droplab physics engine — no new physics in the interface.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.ui import theme
from app.ui.modes import render_home

if theme.in_streamlit():
    render_home()
    theme.about()
    theme.footer()
