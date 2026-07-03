#!/usr/bin/env bash
# Export a curated DropLab release snapshot for the public `droplab` repository.
#
#   ./scripts/make_release_snapshot.sh [DEST]     (default DEST=../droplab-release)
#
# INCLUDES: the package, tests, app, notebooks, examples, selected docs, packaging files.
# EXCLUDES: paper/ (manuscript lives in Overleaf), research scratch, legacy apps,
#           experiment images, .claude/, caches, git history (fresh repo = fresh history).
#
# After exporting:  see the printed next-steps (git init -> push -> tag v2.0.0).
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$SRC/../droplab-release}"

mkdir -p "$DEST"
# Plain --delete (NOT --delete-excluded): it removes dest files missing from the source but
# LEAVES excluded paths — crucially the dest's own .git/ — untouched, so re-running against a
# git clone never destroys its history. Excluded files are simply never copied in.
rsync -a --delete \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' \
  --exclude '.git/' --exclude '.claude/' --exclude '.streamlit/secrets*' \
  --exclude 'CLAUDE.md' --exclude 'assessment/' \
  --exclude 'paper/' --exclude 'manuscript_material/' \
  --exclude 'docs/superpowers/' \
  --exclude 'docs/*_DESIGN.md' --exclude 'docs/*_PLAN.md' \
  --exclude 'docs/SANDBOX_UIUX_HANDOFF.md' --exclude 'docs/LECTURE_MODE_CONTENT.md' \
  --exclude 'app_streamlit.py' \
  --exclude 'app/streamlit_sandbox.py' --exclude 'app/streamlit_climate.py' \
  --exclude 'tests/test_sandbox.py' --exclude 'tests/test_gui_smoke.py' \
  --exclude 'tests/test_climate_packaging.py' \
  --exclude 'README_STREAMLIT.md' \
  --exclude 'examples/_*.py' \
  --exclude '_*.png' --exclude 'lightning_*.png' --exclude 'lem_broadening.png' \
  --exclude 'comparison_*.png' \
  --exclude 'output/' --exclude '*.nc' --exclude '*.gif' \
  --exclude 'SENSITIVITY_TEST_HANDOFF.md' --exclude 'ensemble_comparison.py' \
  "$SRC/" "$DEST/"

# sanity: no personal paths / private references in the snapshot
if grep -rn "$HOME" "$DEST" \
     --include='*.py' --include='*.md' --include='*.toml' --include='*.yaml' \
     --include='*.yml' --include='*.cff' -l 2>/dev/null | grep -v ipynb; then
  echo "!! personal references found above — fix before publishing"; exit 1
fi
echo "clean: no personal paths in snapshot"

cd "$DEST"
python -m pytest tests/ -q -x --co -q >/dev/null 2>&1 && echo "tests collect OK" || echo "!! test collection failed — check excludes"

cat <<'EOS'

Snapshot ready. To publish:
  1. Create an EMPTY GitHub repo named `droplab` (no README/license — we bring our own).
  2. cd into the snapshot directory, then:
       git init -b main
       git add -A
       git commit -m "DropLab v2.0.0 — initial public release"
       git remote add origin git@github.com:jslim93/droplab.git
       git push -u origin main
       git tag v2.0.0 && git push origin v2.0.0
  3. Run the full test suite once in a fresh clone (pytest -q) before telling coauthors.
  4. (Recommended) Enable CI: the workflow in .github/ runs the suite on 3.11-3.13.
  5. (For the paper) Archive the release on Zenodo -> DOI, add the badge to README.
EOS
