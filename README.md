# trace

Lineage candidate line tracing from georeferenced maps.

Part of [Lineage](https://github.com/line-age/controlplane). What this component takes in,
what it puts out, and what it depends on are described in the controlplane's
[components.md](https://github.com/line-age/controlplane/blob/main/components.md#trace).

Work is tracked in the [Trace](https://github.com/orgs/line-age/projects/5) project. The
team lead is [@tisljaricleo](https://github.com/tisljaricleo). If that changes, the
[controlplane's table](https://github.com/line-age/controlplane#projects-and-leads) is updated first.

## Status

A working proof of concept on one sheet: *The Oil and Gas Journal's Illinois Basin Map*, 1937
(Indiana Historical Society `dc035/217`). It finds the oil pipe lines and writes them as one
clean line each: 33 lines today. The main trunks, parallel pairs and corners look right. Some
lines around Wood River are still missing, and there is no hand-traced ground truth yet, so
there are no real scores. How it works, in plain terms: [docs/how-it-works.md](docs/how-it-works.md).
Every rule, score and parameter, with where each number comes from:
[docs/algorithm.md](docs/algorithm.md).

Nothing here is tuned for production, and it has run on exactly one sheet.

## Quick start

You need macOS with [QGIS](https://qgis.org) 4.2 (its bundled Python has GDAL, numpy, scipy,
networkx and the rest), [uv](https://docs.astral.sh/uv/), git, curl and
[tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract uv`). About
3 GB of disk.

```
cd pipeline
./setup/fetch_data.sh        # the scan, 1937 county boundaries, rivers, county names
./spotter/setup.sh           # the text spotter (its own venv, torch, about 1.5 GB)
./run_all.sh                 # every step in order; about an hour, half of it text spotting
```

`./run_all.sh 7` resumes at step 7 (numbers as in the table below).

Everything goes in `work/` next to `pipeline/` (set `TRACE_WORK` to put it elsewhere). The
answer is `work/out/explain_joined.gpkg`, layer `corrected`. Open it in QGIS over
`work/sheet-warped.tif`.

To re-run only the last stages after changing the classifier or the join step (about a
minute):

```
TRACE_RUN=learned ./qpy run_explain.py --classify-only --reuse-symbols
./qpy bridge.py
```

## The steps

| # | Script | What it does | Writes (in `work/`) |
|---|---|---|---|
| 1 | `setup/warp_sheet.py` | Georeferences the scan from 22 hand-read control points | `sheet-warped.tif` |
| 2 | `setup/make_ink.py` | First ink mask | `ink.tif` |
| 3 | `setup/make_borders.py` | County boundaries as of 1937-07-01 | `out/borders_1937.gpkg` |
| 4 | `prep.py` | Grid, pools, heavy strokes, clean dashes | `out/masks.npz`, `out/dashes.json` |
| 5 | `paint_out.py` | Paints grid and chained dashes out, for the spotter | `out/sheet-painted.png` |
| 6 | `spotter/spot_sheet.py` | Reads all text, 1024 px tiles | `text/sheet.json` |
| 7 | `extend_text.py` | Finishes letter-spaced names, matches the gazetteer | `out/text_extended.json` |
| 8 | `text_mask.py` | Text as a mask | `out/text_mask.npz` |
| 9 | `learn_styles.py` | Learns the sheet's dash styles | `out/styles.json` |
| 10 | `prep.py` (learned) | Step 4 again, with those styles | `out/masks_learned.npz` |
| 11 | `border_fit.py` | Measures and fixes the georeferencing error | `out/georef_fix.json` |
| 12 | `legend.py` | Reads the legend into classes | `out/legend.json` |
| 13 | `run_explain.py` | Removes text, borders, rivers, symbols, grid; classifies what's left | `out/explain_paths.gpkg`, `out/explain_removed.tif`, `out/explain_remains.tif` |
| 14 | `bridge.py` | Joins each line's pieces into one line | `out/explain_joined.gpkg` |

Supporting modules: `common.py` (paths, shared geometry), `explain.py` (ink, paper colour,
the removal state), `stages.py` (the removal stages), `classify_residual.py` (pieces, links,
classes), `borders.py` (border detection, the georeferencing shift field).

Also here: `diagnose_breaks.py` and `tally_breaks.py` (why lines break), `debug/` (one-off
checks: duplicate lines, crops of the scan with lines drawn on), and `legacy/` (the earlier
dash-tracker approach this replaced; kept for reference, not maintained).

## Looking at the result

In QGIS, add `work/sheet-warped.tif`, then `work/out/explain_joined.gpkg` layer `corrected`.
Style it *Categorized* on `line` with random colours, so each pipeline is one colour, and
label it with `line`. Other layers worth a look:

- `explain_removed.tif`: what each stage removed (1 text, 2 boundary, 3 river, 4 legend, 5 grid)
- `explain_remains.tif`: the cleaned map the classifier saw
- `explain_joined.gpkg` layers `joins` (each bridged gap) and `ends` (each open end, with its cause)

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `TRACE_WORK` | `../work` | Working directory: inputs and outputs |
| `TRACE_SHEET` | `$TRACE_WORK/sheet-warped.tif` | The georeferenced scan |
| `TRACE_RUN` | empty | `learned` uses the learned dash styles (steps 10 and 13 need it) |
| `TRACE_SPOTTER_PY` | `$TRACE_WORK/spotter/venv/bin/python` | The spotter's Python |
| `TRACE_TESSERACT` | `tesseract` on PATH | Tesseract, for single letters |
| `QGIS_APP` | `/Applications/QGIS-final-4_2_2.app` | Where `./qpy` finds QGIS's Python |

## Rights and licences

- The scan is marked "All Rights Reserved" by the Indiana Historical Society, and the 1937 map
  may still be in copyright. `fetch_data.sh` downloads it for research use. Never commit it or
  anything rendered from it (`work/` is ignored).
- County boundaries: Newberry Atlas of Historical County Boundaries, CC0. Rivers: Natural
  Earth, public domain.
- The text spotter is built on AdelaiDet, licensed for academic use (contact the authors for
  commercial use), and its weights state no licence. Settle this before any production use.

## For AI agents

Start with [AGENTS.md](AGENTS.md).
