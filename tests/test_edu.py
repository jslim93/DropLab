"""Gates for the droplab.edu pedagogical scaffold (docs/EDU_FRAMEWORK.md).

A pure presentation layer: each step returns consistent markdown, the misconception
register is complete, and unknown codes fail loudly. No physics.
"""
import pytest

from droplab import edu


def test_misconception_register_is_complete():
    # warm-phase M1-M7 + synthesis-free, mixed-phase Mi1-Mi4
    assert {f"M{i}" for i in range(1, 8)} <= set(edu.MISCONCEPTIONS)
    assert {f"Mi{i}" for i in range(1, 5)} <= set(edu.MISCONCEPTIONS)
    for wrong, right in edu.MISCONCEPTIONS.values():
        assert wrong and right and wrong != right          # both halves present


def test_lesson_frames_outcome_and_names_misconception():
    md = edu.lesson("Marine cloud brightening", outcome="explain the Twomey effect",
                    misconception="M2")
    assert "Learning outcome" in md and "Twomey" in md
    assert "M2" in md and "bigger droplets" in md          # the wrong idea is NAMED


def test_six_steps_return_distinct_consistent_markup():
    p = edu.predict("bigger or smaller?")
    o = edu.observe()
    e = edu.explain("more, smaller drops; if you predicted bigger, that's M2.")
    r = edu.refine("hit albedo ~ 0.55 with one knob")
    c = edu.self_check("double the salt?", "albedo still rises")
    assert "Predict" in p and "compare with your prediction" in e
    assert "Refine" in r and "Observe" in o
    assert c.startswith("<details>") and "Check yourself" in c   # revealable formative check


def test_unknown_misconception_fails_loudly():
    with pytest.raises(KeyError):
        edu.misconception("M99")
    with pytest.raises(KeyError):
        edu.lesson("x", outcome="y", misconception="nope")
