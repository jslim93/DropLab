"""Lecture mode page (thin glue — logic lives in app.ui.modes.render_lecture)."""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.ui import theme
from app.ui.modes import render_lecture

if theme.in_streamlit():
    render_lecture()
