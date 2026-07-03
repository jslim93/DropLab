"""End-to-end page tests: actually EXECUTE each page headlessly via Streamlit's
AppTest and assert no exception.

Unlike the import-only checks in test_sandbox_smoke.py, these run the full page
script through the Streamlit runtime — so they catch the widget-construction
crash class (e.g. a slider whose default sits outside its min/max) that
import-only tests miss. Pages that auto-run (Parcel) exercise a real run; the
button-gated pages (2-D, Climate) are asserted at their pre-run state (their run
paths are covered by the direct-wrapper tests in test_sandbox_smoke.py).
"""
import pathlib

import pytest

from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]

PAGES = [
    "app/Home.py",
    "app/pages/1_Parcel.py",
    "app/pages/2_TwoD.py",
    "app/pages/3_Climate.py",
    "app/pages/4_Lecture.py",
]


@pytest.mark.parametrize("rel", PAGES)
def test_page_runs_without_exception(rel):
    at = AppTest.from_file(str(ROOT / rel), default_timeout=180)
    at.run()
    assert not at.exception, f"{rel} raised: {at.exception}"


def test_twod_driven_run():
    """Drive one real run through the 2-D page UI: press Run and assert the page
    renders results without an exception (catches run-path UI wiring bugs)."""
    at = AppTest.from_file(str(ROOT / "app/pages/2_TwoD.py"), default_timeout=240)
    at.run()
    assert not at.exception
    # find and click the primary "Run cloud" button, then re-run the script
    buttons = [b for b in at.button if "Run" in (b.label or "")]
    assert buttons, "expected a Run button on the 2-D page"
    buttons[0].click()
    at.run()
    assert not at.exception, f"2-D driven run raised: {at.exception}"
