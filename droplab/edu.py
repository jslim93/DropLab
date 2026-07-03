"""Reusable pedagogical scaffold for DropLab lessons — PURE presentation, no physics.

Emits the six-step Lesson Pattern (docs/EDU_FRAMEWORK.md) uniformly across notebooks
and apps:

    Frame (learning outcome + named misconception) -> Predict -> Observe ->
    Explain (tied to the prediction) -> Refine (completion task) -> Check (self-check).

In a Jupyter notebook each call renders Markdown/HTML; everywhere it also RETURNS the
markdown string, so the Streamlit apps and the instructor handbook reuse identical
wording. This is what makes "educational" a *uniform design property* rather than a
per-lesson improvisation.

    from droplab.edu import lesson, predict, observe, explain, refine, self_check
    lesson("Marine cloud brightening", outcome="explain the Twomey effect", misconception="M2")
    predict("Will droplets get bigger or smaller? Sign of the TOA forcing?")
    # ... run the model ...
    explain("more, smaller drops -> brighter; if you predicted bigger, that's M2.")
    refine("Using ONLY background N, hit albedo ~ 0.55.")
    self_check("Double the salt -- does cloud water change?", "Roughly no; albedo still rises.")
"""

# Misconception register (warm-phase M1-M7 from docs/PEDAGOGY_DESIGN.md §A;
# mixed-phase Mi1-Mi4 from docs/EDU_FRAMEWORK.md §5). (wrong intuition, correct idea).
# NOTE: instructor-experience candidates — validate before any research pre/post use.
MISCONCEPTIONS = {
    "M1": ("clouds need large supersaturation to form",
           "peak supersaturation is small (~0.1-1%) and is quickly consumed by the activating droplets"),
    "M2": ("more aerosol makes bigger droplets",
           "more aerosol -> more, SMALLER droplets (they compete for the same vapour)"),
    "M3": ("condensation makes rain",
           "condensation alone stalls near ~10 um; collision-coalescence bridges to rain"),
    "M4": ("all aerosol particles activate",
           "only particles above their critical (Koehler) size activate; the rest stay as haze"),
    "M5": ("droplet growth is smooth and uniform",
           "collection is stochastic and the spectrum broadens -- the reason a super-droplet method is used"),
    "M6": ("a few giant CCN are negligible",
           "a small number of giant CCN can seed the first rain embryos"),
    "M7": ("entrainment just dilutes everything uniformly",
           "homogeneous vs inhomogeneous mixing change droplet NUMBER vs SIZE differently"),
    "Mi1": ("below 0 C, clouds are made of ice",
            "supercooled LIQUID persists down to ~-38 C; mixed-phase clouds are common"),
    "Mi2": ("ice and liquid coexist in stable equilibrium",
            "WBF: ice grows at the liquid's expense (e_s,ice < e_s,liq), so mixed-phase is transient unless replenished"),
    "Mi3": ("more INP -> more ice -> more snow, always",
            "INP glaciate the cloud and can deplete the liquid; the precipitation response is non-monotonic"),
    "Mi4": ("drops freeze at 0 C",
            "homogeneous freezing is ~-38 C; heterogeneous freezing needs INP (Bigg/ABIFM); most drops supercool"),
}


def _emit(md):
    """Render inside a Jupyter KERNEL; always return the markdown string.

    Only auto-displays under a real notebook kernel (ZMQInteractiveShell) — in plain
    scripts and Streamlit it simply returns the string (so apps do `st.markdown(...)`
    with no display noise)."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell":
            from IPython.display import Markdown, display
            display(Markdown(md))
    except Exception:
        pass
    return md


def _mis(code):
    if code not in MISCONCEPTIONS:
        raise KeyError(f"unknown misconception {code!r}; known: {sorted(MISCONCEPTIONS)}")
    return MISCONCEPTIONS[code]


def misconception(code):
    """(②-flag) Name a common wrong intuition so it can be confronted, not bypassed."""
    wrong, right = _mis(code)
    return _emit(f"> ⚠️ **Common wrong intuition ({code}):** *“{wrong}.”* "
                 f"In fact: {right}.")


def lesson(title, outcome, misconception=None):
    """(① Frame) Learning outcome + (optionally) the targeted misconception."""
    md = f"## \U0001f9ea {title}\n\n> **Learning outcome:** {outcome}."
    if misconception:
        wrong, _ = _mis(misconception)
        md += f"\n>\n> ⚠️ **Targeted misconception ({misconception}):** *“{wrong}.”*"
    return _emit(md)


def predict(question):
    """(② Predict) Elicit a committed prediction BEFORE running."""
    return _emit(f"> ### \U0001f914 Predict (before you run)\n> {question}\n>\n"
                 f"> *Commit to an answer first — that is what makes this learning, not watching.*")


def observe(note=""):
    """(③ Observe) Read the result across the linked representations."""
    body = note or "Run the cell and read the result across the panels (field, droplets, diagnostics)."
    return _emit(f"> ### \U0001f52c Observe\n> {body}")


def explain(text):
    """(④ Explain) Reconcile the result AGAINST the student's own prediction."""
    return _emit(f"> ### \U0001f4d6 Explain — *compare with your prediction*\n> {text}")


def refine(task):
    """(⑤ Refine) A one-control completion task: change one knob to hit a target."""
    return _emit(f"> ### \U0001f3af Refine (one-knob challenge)\n> {task}")


def self_check(question, answer):
    """(⑥ Check) A revealable formative self-check mapped to the outcome."""
    html = ("<details><summary><b>✅ Check yourself</b> — click to reveal</summary>\n\n"
            f"> **Q:** {question}\n>\n> **A:** {answer}\n</details>")
    return _emit(html)
