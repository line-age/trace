#!/usr/bin/env bash
# The whole pipeline, in order, from a fresh $TRACE_WORK (default ../work).
#
#   ./run_all.sh        every step
#   ./run_all.sh 7      resume at step 7 (numbers as in the README)
#
# Prerequisites: setup/fetch_data.sh (sheet, county boundaries, rivers, gazetteer),
# spotter/setup.sh (text spotter), tesseract on PATH, and QGIS (for ./qpy). Whole-sheet text
# spotting (step 6) is skipped when $TRACE_WORK/text/sheet.json already exists.
# Roughly: prep 3 min, text spotting 30+ min on CPU, run_explain 25 min, the rest a few minutes.
set -euo pipefail
cd "$(dirname "$0")"
FROM="${1:-1}"
WORK="$(./qpy -c 'from common import S; print(S)')"
SPOTTER_PY="${TRACE_SPOTTER_PY:-$WORK/spotter/venv/bin/python}"

step() {
  local n="$1" what="$2"; shift 2
  [ "$n" -lt "$FROM" ] && return 0
  printf '\n== %s. %s  [%s]\n' "$n" "$what" "$(date +%H:%M:%S)"
  "$@"
}

spot_sheet() { [ -s "$WORK/text/sheet.json" ] && echo "(have text/sheet.json)" || "$SPOTTER_PY" -I spotter/spot_sheet.py; }

step 1  "georeference the sheet"          ./qpy setup/warp_sheet.py
step 2  "ink mask"                        ./qpy setup/make_ink.py
step 3  "1937 county boundaries"          ./qpy setup/make_borders.py
step 4  "grid, pools, heavy strokes"      ./qpy prep.py
step 5  "paint out grid for the spotter"  ./qpy paint_out.py
step 6  "text spotting, whole sheet"      spot_sheet
step 7  "letter-spaced names"             ./qpy extend_text.py
step 8  "text mask"                       ./qpy text_mask.py
step 9  "learn the sheet's dash styles"   ./qpy learn_styles.py
step 10 "the same masks, learned styles"  env TRACE_RUN=learned ./qpy prep.py
step 11 "georeferencing check and fix"    ./qpy border_fit.py
step 12 "read the legend"                 ./qpy legend.py
step 13 "remove, then explain"            env TRACE_RUN=learned ./qpy run_explain.py
step 14 "join each line's pieces"         ./qpy bridge.py
printf '\n== done: %s/out/explain_joined.gpkg, layer "corrected"\n' "$WORK"
