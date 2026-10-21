# Instructions for AI agents

You are working on a pipeline that reads a scanned, georeferenced historical map and extracts
one kind of line from it (here: oil pipe lines on the 1937 O&GJ Illinois Basin sheet). Read
these before changing anything:

1. [README.md](README.md): the steps, scripts, outputs and settings.
2. [docs/how-it-works.md](docs/how-it-works.md): the method in plain terms.
3. [docs/algorithm.md](docs/algorithm.md): every rule, score and parameter in run order, each
   number tagged with where it comes from (measured, legend, relative, sheet-specific, compute
   bound), and the output file formats. Reason from this; check against the code.
4. [.claude/skills/read-map/SKILL.md](.claude/skills/read-map/SKILL.md): the general
   procedure and the lessons learned, for any map. Claude Code loads it as the `read-map` skill.
5. [.claude/skills/read-map/prompts.yaml](.claude/skills/read-map/prompts.yaml): one seed
   prompt per step, the metrics each step moves, and which functions carry it out.

## Rules

- **Measure, don't tune.** Every number comes from the sheet (its legend, its histograms, its
  own measurements) or is a relative tolerance. The values tagged **S** in docs/algorithm.md
  are the exceptions still to fix; don't add more. Never add a constant that makes this one sheet
  look better. If a fix only works because of a number you picked by looking at this sheet,
  it's wrong. The goal is that a new sheet needs only a new measured profile.
- **Outside data is a prior, never an answer.** County boundaries and rivers say where to look;
  the ink says what is there. Never use data about the target itself (modern pipeline maps)
  as input. That's for scoring only.
- **Remove, and keep a mask.** A stage that explains ink erases it (paper colour) and records
  where in the removal mask. Only gap tests read the mask: removed ground means "covered,
  can't tell", untouched paper means "the line stops". Never treat removed ground as paper.
- **Diagnose with numbers before changing code.** When something is wrong, find which step and
  which rule caused it, count the cases, then change the method. `diagnose_breaks.py` is the
  model for this.
- **Look at the result.** Render the lines over the scan, one colour per line, at full zoom
  where it matters. Most bugs here were found by looking, not by counts: every line traced
  twice, a grid line passing as pipeline, corners cut across.
- **Report what you verified, and how.** If a run wasn't done or a check wasn't possible, say
  so. There is no ground truth yet, so "looks right" is a spot check, not a score.
- **Keep prompts.yaml current.** When you add or change a step, update its prompt, its
  `scores`/`feedback`, and its entry under `code_components`. The prompts and the code are
  both meant to be tuned later by a reflective optimizer (GEPA-style) against several metrics.

## Running

Use `pipeline/qpy` (QGIS's Python with GDAL and PROJ paths set) for everything except the text
spotter, which has its own venv (`$TRACE_WORK/spotter/venv/bin/python -I spotter/...`).

| What | Command (from `pipeline/`) | Time |
|---|---|---|
| Everything | `./run_all.sh` | about 1 hour |
| Classifier and join only | `TRACE_RUN=learned ./qpy run_explain.py --classify-only --reuse-symbols` then `./qpy bridge.py` | about 1 minute |
| Full removal and classification | `TRACE_RUN=learned ./qpy run_explain.py` | about 25 minutes |
| Symbol matching again | `TRACE_RUN=learned ./qpy run_explain.py --classify-only` | about 10 minutes |

Run long jobs as standalone processes. If you drive QGIS through an MCP server, its
`execute_code` backgrounds anything over about 120 s, and the job keeps running inside QGIS
even after you stop waiting, blocking later calls. Use QGIS only for viewing.

## Checking

- **Duplicates:** `./qpy debug/dbg_overlap2.py` lists pairs of output lines within 12 px of
  each other. Each pair is either a bug or a real parallel pair (about 11 px apart on this
  sheet). Look at each one: `./qpy debug/dbg_crop.py A B tag` writes
  `work/out/crop_tag.png`, the scan with both lines drawn on.
- **Breaks:** `./qpy diagnose_breaks.py` then `./qpy tally_breaks.py`: for every open end, the
  best continuation and which rule refused it.
- **Open ends:** `explain_joined.gpkg` layer `ends` gives each end's cause (junction, paper
  ahead, into removed text/legend/river, unlinked ink ahead, other).

## Known problems

- Missing lines around Wood River (the Shell line going southwest, most of Ajax); the second
  line of some parallel pairs is only partly found.
- `setup/make_ink.py` uses a fixed grey threshold (110), and `stages.py`/`make_borders.py`
  carry this sheet's extent as a literal box. Both should come from the sheet.
- Dry-hole symbols match zero times: their legend template doesn't fit their map versions.
- No ground truth. The next real step is a hand trace of two patches, one to tune on and one
  held out, and a scorer for the metrics in prompts.yaml.

## Gotchas

- `legend.json` bboxes are `[row0, col0, row1, col1]`, not x/y.
- Pixel space throughout: x = column, y = row. Convert to lon/lat only on output
  (`common.to_lonlat`).
- Keep GDAL/OGR dataset objects alive while you use their layers or geometries.
- QGIS 4 needs Qt6 enum paths in PyQGIS code (`Qt.PenStyle.SolidLine`).
- Never commit anything in `work/`, the scan, or images rendered from it (see README, rights).
