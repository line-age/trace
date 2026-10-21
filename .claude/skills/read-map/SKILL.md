---
name: read-map
description: Procedure for extracting a target linear feature (pipelines, railroads, roads, canals, boundaries) from a scanned, georeferenced historical map of any style. Use when tracing candidate lines from a map sheet, when tuning or debugging a map-reading pipeline, or when starting on a new sheet. Covers measuring the sheet's own symbology, explaining away the other layers (grid, fills, text, borders, rivers, symbols) in cooperative passes, using outside data as priors, tracking through occlusion, and checking results.
---

# Reading a map

The goal is to find one kind of line (the *target*) on a scanned map. Everything else on the
sheet gets in the way: letters, survey grids, borders, rivers, filled areas, point symbols. This
procedure finds the target by explaining away everything else, carefully, in passes that help
each other.

It was worked out on one sheet (O&GJ Illinois Basin, 1937). Rules that held on that sheet but
are not general are marked "sheet-specific" and must be re-measured on every new sheet, never
copied.

## Principles

1. **Read the map's parameters off the map.** Ink colours, dash lengths, gaps, stroke
   weights, letter heights and line tilts are measured on each sheet. The legend *confirms* a style; it does
   not define it. Draftsmen take liberties (shorter dashes in crowded areas, borders nudged off
   other features), so hard-coded legend values fail. No pixel constant belongs in code without
   a measurement behind it; write measured values to a per-sheet profile (below).
2. **Explain away, cooperatively.** Each non-target layer is found and *owned*, and each
   detector looks at the sheet minus what the other layers own. Iterate: removing borders makes
   text easier to read, removing text makes lines easier to follow, and finding lines helps
   separate letters fused to them. Recompute each layer from the original every round
   (don't erase cumulatively), so an early wrong claim can be handed back.
3. **Remove, and keep a mask.** A stage that explains ink erases it (filled with the local
   paper colour) and records where, by stage, in a removal mask, so later stages see a simpler
   map. Erased ground is *covered* ("can't tell"), never blank paper: only gap tests read the
   mask, and without it every crossing becomes a false gap. Layers cross (a pipeline crosses a
   border, a label sits on its line), so remove only ink that matches the stage's own measured
   style, and leave target-weight ink inside a border strip alone. Check what each removal
   costs: deleting dash-shaped ink once erased the diagonals of W and N.
4. **Three-valued evidence.** Every test returns *supports*, *contradicts* (blank paper where
   the line should be) or *uninformative* (covered). Only "contradicts" stops a line.
5. **A style is whatever the legend varies to tell symbols apart, and no single measurement
   decides it.** Find out per sheet which channels carry the distinction: **colour** (often the
   strongest and cheapest, where the map has it), stroke weight, dash pattern, casing or double
   lines, ornament (ticks, crosses, dots). On a monochrome sheet pattern and weight carry it; on
   a coloured one, colour may settle most of it outright. Use every channel the sheet uses
   together, and never one bar's length alone: dashes get clipped by crossings and shortened by
   the draftsman, so let a bar join any style it could be a clipped version of, and decide by
   the chain (its colour, spacing, weight).
6. **Outside data is a prior, never an answer.** Known geography says where to look; the ink
   says what is there. Buffer every prior by the measured georeferencing error, and require ink
   evidence inside the buffer. Never feed in data about the target itself (e.g. modern pipeline
   maps): that biases the very thing being measured. Keep it for validation.
7. **Never end on a guess.** A line may coast through covered ground, but its ends must sit on
   real evidence. Trim trailing coasted steps, including when it stops at a fill.
8. **Look at it, one colour at a time.** Judge results on the map, with one layer visible in one
   colour (solid = evidence, dashed = coasted). Status-coloured multi-layer views are
   unreadable. Re-check fixed windows after every change: fixes regress each other.

## Procedure

### 0. Setup

- Check the georeferencing against modern data at 2-3 well-separated places (river mouths,
  towns) before anything else.
- Ink: on a monochrome sheet, a threshold chosen from the grey histogram (not a fixed value).
  On a coloured sheet, separate the inks first: cluster ink pixels in a perceptual colour
  space (CIELAB), one mask per printed colour, plus black. Expect aged paper and fading to
  shift colours across the sheet, so fit the clusters on the sheet itself (and locally if the
  shift is strong), and expect colours where inks overprint or misregister to sit between
  clusters.
- Work in the scan's pixel frame; convert to lon/lat only on output.

### 1. Measure the sheet → sheet profile

Write `profile.json` for the sheet. Every number below comes from the sheet:

- **Which channels the legend uses** (colour, weight, pattern, ornament), from the legend
  swatches. This decides which of the measurements below matter.
- **Colours:** the printed ink palette (cluster centres) and which legend entries use which.
  When the target has its own colour, its colour mask is the first and best candidate filter,
  and most of the explain-away work below shrinks to what shares that colour.
- **Stroke widths:** histogram of 2 × distance transform over the ink. Expect separate peaks
  for hairlines (grid), target lines, heavy lines (borders) and fills.
- **Line tilt:** the angle spread of long straight strokes (warping leaves survey lines a few
  degrees off axis; an axis-only grid test leaves slivers that pass as dashes).
- **Dash styles:** thin elongated bars outside fills and text; pairs lying end to end on one
  line with similar lengths; histogram of dash length (and gap). Peaks are styles, valleys are
  the window bounds; floor each window at half its dash length. Wide gaps with a dot in them
  suggest dash-dot styles.
- **Letter height** and whether labels are letter-spaced (pitch > 1.5 × height).
- **Legend:** crop the swatches and match each to a measured style. A style with no legend
  entry is normal (the 1937 sheet's commonest dash was not in its legend).

### 2. Gather priors (dated to the map's year)

- **Boundaries:** US: Newberry *Atlas of Historical County Boundaries* (CC0, dated versions;
  select by `START_N <= yyyymmdd <= END_N`). Elsewhere: the nearest historical equivalent;
  modern OSM / census boundaries only where boundaries have not moved.
- **Survey grid:** PLSS (BLM) where it applies.
- **Names:** gazetteers for counties and places (census lists, GNIS) to complete and check text.
- **Water:** OSM waterways as a loose corridor only (rivers move; boundaries that follow rivers
  are unusable for fitting).

### 3. Tighten the georeferencing with the priors

Straight stretches of known boundaries become control measurements: at intervals, find the
printed line's offset across the boundary (ink peak in a window). Each gives the error along
one direction (n · shift = offset). Fit a **smooth, low-order** shift field (degree-2
polynomial with outlier down-weighting), not one that passes through every point: the
draftsman's deliberate nudges must not bend the target lines. **Hold out** a fifth of the
measurements and report error before/after on them. (1937 sheet: 7.0 → 3.7 px median.) Apply
the fix to the priors (to keep them aligned with the displayed scan) and to final output.

### 4. Explain away, in rounds

Order by certainty, cheapest first. Each detector sees ink minus every *other* layer's owned
ink. Repeat rounds until owned sets stop changing (a few % of pixels).

| Layer | How (shape-, prior- and evidence-based) | Pitfall seen |
|---|---|---|
| Grid | long straight thin strokes, at the measured tilts | axis-only stamps leave slivers |
| Fills / areas | ink a large square fits in, at a size above the heaviest *line* weight | bold letters pass a small square |
| Text | learned text spotter (see Tools), whole sheet in overlapping tiles with context; then letter-spaced extension; gazetteer completion | spotter misses spaced names; tile edges clip labels |
| Borders | prior strip (± georef error) + measured border weight / pattern inside it | border pattern is sheet-specific (1937: heavy long dashes, not clean dash-dot) |
| Rivers / water | heavy, curving, continuous strokes; loose water prior | rivers pass as dashes where broken by labels |
| Point symbols | small closed shapes from the legend | (not yet built) |
| Target | below | target is a layer too: owning it helps the others |

**Text specifics.** Text is occlusion, never "nothing here". A chain of dash-like bars made
only of in-text bars cannot seed a line (letters pass as dashes), but a line with at least two
members outside text keeps its in-text members: labels are printed *on* their own lines.
Letter-spaced names: walk along the baseline one letter pitch at a time, with the same three
values (letter / covered / blank-stops); find letters by tall columns (ink deep across the
band), so a line through the letter tops cannot widen or hide them; read the spotter's own
characters where its pieces line up, and match to the gazetteer. Mask spaced names letter by
letter: their gaps are paper, and lines run through them.

### 5. Explain what remains, then join each line

Code: `pipeline/classify_residual.py` and `pipeline/bridge.py` in this repo.

- **Pieces:** connected bits of leftover ink with centre, axis, length, stroke width (area /
  length) and colour. Keep elongated ones; don't drop short ones by a fixed length.
- **Links:** two ends that face each other get a log-odds score, target against chance
  lining-up: turn, sideways offset, similar length and width, the gap's evidence (blank paper
  counts against, removed ground is a small cost, ink counts for), and how crowded the spot
  is. A link may never pass over another piece lying along the same line: otherwise every
  dashed line comes out twice, as two interleaved chains of every other dash.
- **Global matching:** a maximum-weight matching over all ends, so each end gets its best
  partner against every alternative.
- **Classes compete:** each path is explained by the legend class that fits best, including
  chance. Judge stroke weight on the whole path (draftsmen vary dash length, not pen weight;
  a grid line's remnants line up perfectly and outvote a wrong weight per piece). The
  legend's own swatches are not map features.
- **Join each line's pieces:** across any gap that isn't a run of blank paper longer than the
  target's own gap, at any distance. Ends that nearly touch at an angle are a corner; two
  ends side by side pointing the same way are two lines. Join target paths only (a path that
  can't pass alone must not be glued on), and never count another line's ink as support for
  a join running beside it. Simplify each joined line to one polyline.
- **Check for duplicates:** two output lines within a stroke width or two of each other are a
  bug or a real parallel pair. Look at every one in the scan.
- **Known open problems:** lines that vanish into dense label clusters (Wood River), the
  second line of a parallel pair, lines ending inside fills.

### 6. Check

- Fixed comparison windows with distinct problems (dense field, river crossing, label-heavy).
  After any change, re-render all of them: on the 1937 sheet nearly every fix broke another window
  once.
- A line claimed earlier as a target may be a border (both looked like "true positives" until
  the boundary prior was overlaid). Overlay priors when judging.
- Ground truth: hand-traced patches, one for tuning and one held out. Score only on held-out.
- **Generality test:** run on a different sheet; count the steps that needed a new hand rule.
  The goal is that only `profile.json` changes.

## Tools

- **Text spotter:** MapTextPipeline (Rumsey fine-tune), via MapReader (MIT). Install in its
  own venv, never inside QGIS's Python: detectron2 + MapTextPipeline from source,
  `hf download rwood-97/MapTextPipeline_rumsey rumsey-finetune.pth`, config
  `configs/ViTAEv2_S/rumsey/final_rumsey.yaml`. Use `ViTAEPredictor` (pads to a multiple of 32;
  MapReader overwrites it with `DefaultPredictor` by mistake), feed BGR, CPU is fine
  (~1-4 s per 1024 px tile). Licence: built on AdelaiDet ("academic use"; contact for
  commercial) and the weights state none. Resolve before production use. mapKurator's own
  models are CC BY-NC / paid tiers.
- **Tesseract** is a weak fallback for map text (misses italics, reads dashed lines as text),
  useful only for single characters with a whitelist.
- **QGIS MCP:** `execute_code` runs in a fresh namespace; calls over ~120 s go to the
  background; QGIS 4 needs Qt6 enum paths (`Qt.PenStyle.SolidLine`,
  `Qgis.LabelPlacement.OverPoint`); keep GDAL feature objects alive while using their
  geometries; set the canvas extent before each render (the user may pan).

## Sheet profile template

```json
{
  "sheet": "<id>", "year": 1937,
  "ink_threshold": 0,
  "channels": ["colour", "weight", "pattern", "ornament"],
  "inks": {"<name>": {"lab": [0, 0, 0], "used_by": ["<legend entry>"]}},
  "georef_fix": {"deg": 2, "coef": [], "heldout_median_px": 0, "heldout_p90_px": 0},
  "stroke_px": {"grid": 0, "target": 0, "border": 0, "fill_square": 0},
  "grid_tilt_deg": 0,
  "letter_height_px": 0,
  "styles": {"<name>": {"dash": 0, "gap": 0, "window": [0, 0], "legend": "<swatch or null>"}},
  "border_pattern": "<what the printed boundaries look like>",
  "notes": []
}
```

The 1937 Illinois Basin values (for reference only, never as defaults): trunk dash 34 px / gap
6, medium dash 17.5 / gap 6 (not in the legend), target stroke ~4-6 px, county lines ~9-10 px
heavy long dashes, grid ~3 px with up to ~3° tilt, letter height 15-20 px (county names ~45 px,
letter-spaced), georeferencing error 7.0 px median before the fix.
