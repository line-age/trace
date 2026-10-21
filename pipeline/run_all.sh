#!/usr/bin/env bash
# The whole pipeline, in order, from a fresh $TRACE_WORK (default ../work).
#
# Prerequisites: setup/fetch_data.sh (sheet, county boundaries, rivers, gazetteer),
# spotter/setup.sh (text spotter), tesseract on PATH, and QGIS (for ./qpy).
# Comment steps out to resume part-way. Whole-sheet text spotting is skipped when
# $TRACE_WORK/text/sheet.json already exists.
# Roughly: warp 2 min, prep 3 min, text spotting 30+ min on CPU, the rest 25 min.
set -euo pipefail
cd "$(dirname "$0")"
WORK="$(./qpy -c 'from common import S; print(S)')"
SPOTTER_PY="${TRACE_SPOTTER_PY:-$WORK/spotter/venv/bin/python}"
say() { printf '\n== %s  [%s]\n' "$1" "$(date +%H:%M:%S)"; }

say "georeference the sheet";           ./qpy setup/warp_sheet.py
say "ink mask";                         ./qpy setup/make_ink.py
say "1937 county boundaries";           ./qpy setup/make_borders.py
say "grid, pools, heavy strokes";       ./qpy prep.py
say "paint out grid for the spotter";   ./qpy paint_out.py
if [ ! -s "$WORK/text/sheet.json" ]; then
  say "text spotting, whole sheet";     "$SPOTTER_PY" -I spotter/spot_sheet.py
fi
say "letter-spaced names";              ./qpy extend_text.py
say "text mask";                        ./qpy text_mask.py
say "learn the sheet's dash styles";    ./qpy learn_styles.py
say "the same masks, learned styles";   TRACE_RUN=learned ./qpy prep.py
say "georeferencing check and fix";     ./qpy border_fit.py
say "read the legend";                  ./qpy legend.py
say "remove, then explain";             TRACE_RUN=learned ./qpy run_explain.py
say "join each line's pieces";          ./qpy bridge.py
say "done: $WORK/out/explain_joined.gpkg, layer \"corrected\""
