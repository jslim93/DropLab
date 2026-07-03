"""Climate-intervention mode page (thin glue — logic in app.ui.modes.render_climate)."""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.ui import theme
from app.ui.modes import render_climate

if theme.in_streamlit():
    render_climate()
    theme.footer()
