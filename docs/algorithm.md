# The algorithm, precisely

This is the reference for reasoning about what the pipeline does: every rule, score and
parameter, in run order, with where each number comes from. [how-it-works.md](how-it-works.md)
is the plain-terms version; the code in `pipeline/` is the ground truth, and where this
document and the code disagree, the code wins and this document has a bug.

Conventions: pixel space of `sheet-warped.tif` throughout (x = column, y = row). A legend
*period* P = dash + gap of the target swatch (34 + 5 = 39 px on this sheet). "sw" is stroke
width, measured as area / length. Log-odds are in nats.

## Where each number comes from

Every parameter below carries one of these tags. The aim is for everything to be **M**, **L**,
**R** or **C**; the **S** values are debts.

| Tag | Meaning |
|---|---|
| **M** | Measured on this sheet at run time |
| **L** | From the legend swatch, measured by `legend.py` |
| **R** | A relative tolerance or statistical choice, the same for any sheet |
| **S** | A fixed pixel or grey-level value that fits this sheet only and should become **M** |
| **C** | A compute bound: it limits search, not the answer (its effect is logged) |

---

## 1. Georeference (`setup/warp_sheet.py`)

22 hand-read control points (town *symbols*, not labels; extremities first) in
`setup/gcps.json`, degree-2 polynomial (chosen by leave-one-out cross-validation in the legend
spike), cubic resampling, EPSG:4326, alpha band for the margin. Output 9245 x 6413 px,
0.000451° per px.

## 2. Ink (`setup/make_ink.py`, `explain.ink_mask`)

Two masks exist; they are used by different steps.

| Mask | Rule | Used by |
|---|---|---|
| `ink.tif` | mean RGB < 110 **S**, inside alpha | prep, paint_out, extend_text, text_mask, learn_styles, border_fit |
| `explain.ink_mask` | paper L\* = mode of the L\* histogram **M**; ink if L\* < paper − 25 **R** or chroma > paper chroma + 25 **R** | legend, run_explain, bridge |

`explain.paper_background`: the local paper colour everywhere, a Gaussian blur (σ = 12 px
**S**) of non-ink pixels only (normalized convolution); where fewer than 5% of nearby pixels
are paper (inside a big pool), the sheet's median paper colour. Erased ink is filled from it.

## 3. County boundaries (`setup/make_borders.py`)

Newberry AHCB IL/IN/KY, counties with `START_N <= 19370701 <= END_N`, clipped to the sheet's
extent (a literal box **S**). 92 counties. Their outlines, unioned, are the boundary prior.

## 4. Shape layers (`prep.py`, `common.explain`)

On `ink.tif`, using `common.LEG` (dash 33, gap 6.5, thick 6.3: an early hand measurement of
the swatch **S**, superseded by `legend.json` but still used here):

- **pools:** opening with an 11 x 11 square **S**, dilated 3 px
- **grid:** openings with 1-px lines of length 3 x (33 + 6.5) = 118 px, horizontal and
  vertical, and tilted ±1, ±2, ±3° (the warp tilts survey lines) **S**; minus pools
- **thick:** distance transform > 6.3 / 2 + 1.5 **S**, measured after removing grid, dilated 5
- **rest:** ink minus all of those dilated 2

Step 13 removes the **grid** layer directly. Pools, thick strokes and chained dashes reach the
result indirectly, as "covered" evidence in the letter walk of step 7; the `dash` layer and
chains also feed `paint_out` and the legacy methods.

## 5. Paint out (`paint_out.py`)

Grid plus *chained* clean dashes (`common.chain` over `dashes.json`), dilated 2, painted
white on a copy of the sheet: `sheet-painted.png`. A lone dash-shaped bar may be a letter
stroke and is kept. Used only for finding letters in step 7.

## 6. Text spotting (`spotter/spot_sheet.py`)

MapTextPipeline, Rumsey fine-tune, `ViTAEPredictor`, CPU, native scale. 1024 px tiles, step
768 (256 overlap) **R**; tiles with < 0.2% pixels darker than 110 are skipped **S**. A reading
whose polygon comes within 6 px of an inner tile edge is marked `edge`. Merge, greedily: non-
edge first, then by score x sqrt(min(len, 8)); a reading is dropped if it overlaps a kept one
by more than half the smaller area. Output `text/sheet.json`: `{text, score, poly, edge}`,
802 readings on this sheet.

## 7. Letter-spaced names (`extend_text.py`)

For each reading at least 30 px tall (**S**: county lettering here is 40-55 px) whose letters
are spaced (pitch ≥ 1.5 x letter height **R**; one seed letter takes its pitch from the nearest
letter-sized blob within 6 heights):

1. Straighten a strip along its baseline (t along, b across), on `sheet-painted.png`.
2. Letters = runs of inked columns in |b| ≤ h/2. Strokes running along the strip longer than
   1.5h are removed first (a line through the letter tops). A letter must fill most of
   the band and not continue past it. Pool-sized ink (a 15 px square fits, **S**) is not a letter.
3. Walk outward one pitch at a time: *letter* (blob within pitch/3 of the prediction),
   *covered* (pool, grid or chained-dash ink: coast, at most 2 in a row **R**), *blank* (stop).
4. Read each letter alone with Tesseract (single character, capitals), match the string with
   `?` for coasted letters against county names of IL/IN/KY/MO/TN (difflib).

Output `text_extended.json`: per name, one box per letter (`cells`) and which readings it
absorbed. 37 of 57 names matched the gazetteer on this sheet.

## 8. Text mask (`text_mask.py`)

Spotter polygons plus extended-name letter cells (the spotter's own box for those names is
dropped: it spans paper the lines run through), grown 2 px. Used by `learn_styles.py` only.

## 9-10. Dash styles (`learn_styles.py`, then `prep.py` again)

Thin bars (width ≤ 9 **S**, length ≥ max(7, 2 x width)) outside grid, pools and text. Pairs:
centres within 70 px **S**, axes and the line between them within ~16° (|cos| ≥ 0.96) **R**,
lengths within x1.35 **R**, gap 2-12 px **S**. Histogram of pair dash length (6-51 px),
smoothed (σ = 2 bins); peaks ≥ 15% of the tallest are styles, valleys are window bounds, each
window floored at half its dash. The style nearest the legend dash is "trunk". On this sheet:
trunk 34 / gap 6.1 (window 24-51), medium 17.5 / gap 6 (window 9-24).

`TRACE_RUN=learned prep.py` rebuilds the masks with these styles (`masks_learned.npz`). Its
grid layer is identical; step 13 loads it for the grid.

## 11. Georeferencing fix (`border_fit.py`, `borders.shift_field`)

Along each straight stretch of a 1937 boundary (≤ 3 px from its chord **S**; river-following
boundaries left out), windows of 240 px every 120 px **S**: slide across the boundary's normal
±40 px **S** and take the offset where ink peaks. Each window measures n · shift = offset.
Fit a degree-2 polynomial shift field (dx, dy) by least squares with iterative down-weighting
(6 rounds, weight 1 / (1 + (r / 2.5s)²), s = 1.4826 x MAD) **R**. Every 5th measurement is
held out. This sheet: 295 measurements, held-out median 7.0 → 3.7 px, 90th percentile 17 →
9.6 px. Output `georef_fix.json`; applied to the priors (boundaries, rivers), never to the scan.

## 12. Legend (`legend.py`)

1. The reading "LEGEND" with the best score is the heading (height h). Entries: readings with
   letters from 2h left to 12h right of it and up to 14h below, merged into lines by centre
   height (within half a line), the block ending at the first gap > 2.5 line heights, and
   only lines whose left edge is within one line height of the median left edge.
2. Swatch: ink left of the label (label boxes blanked first), within 8h, ending at a blank run
   > 1.5h, from 0.4h above to 0.4h below the label line.
3. Type: **line** if wider than 2.5h and at most 0.8h tall (dash = median of the inner runs,
   gap = median gap, width = area / total dash length, colour = mean Lab); **solid area** if
   ≤ 2 pieces, > 60% filled, ≥ 0.5h on both sides; **stipple** if ≥ 10 pieces with median size
   < 0.05h²; otherwise **symbol** (binary template). The target is the entry whose label
   contains every word of `OIL PIPE LINE` (the one input naming what to find).

This sheet: OIL PIPE LINE line, dashed, dash 34, gap 5, width 6.9, Lab (18.2, 2.9, 4.3).
DRILLING WELL, DRY HOLE, REFINERY, NEW POOL symbols; OIL POOL solid area; GAS POOL stipple.
`bbox` is `[row0, col0, row1, col1]`.

## 13. Remove, then explain (`run_explain.py`, `stages.py`, `classify_residual.py`)

### The state

`explain.State`: a working RGB image, its ink, and a removal mask by stage. `remove(name, mask,
grow)`: the mask (dilated `grow` px) ∩ current ink is filled with paper colour, dropped from
ink, and recorded under `name`. Gap tests read the union of all removals as **covered**.

### Pass 1, pass 2, final

1. **Pass 1** (a borderless map for reading text): remove spotter polygons and extended
   letter cells, then boundaries, then rivers (below).
2. **Pass 2:** grow every reading's bound along its baseline (`grow_bound`: letters in a
   straightened strip, reaching 6h past each end **R**; accept the next letter if its gap is
   ≤ 1.3 x the word's pitch, or ≤ 0.8h for one character **R**). For each bound that grew by
   more than h/2, and for every letter-spaced name step 7 rebuilt, re-run the spotter inside it (padded to ≥ 384 px of context) on the sheet
   with pass 1's boundaries and rivers painted out. Keep readings whose centre lies inside
   their bound (± 0.3 x its height). 91 readings from 62 bounds on this sheet.
3. **Final**, from the original sheet, in this order:

| Stage | Rule | Params |
|---|---|---|
| text | all polygons (spotter, letter cells, pass 2), drawn with 2 px growth, removed with 1 more | **R** |
| boundary | windows of 160 px along each shifted 1937 boundary, strip ±12 px; per window the printed line's offset (ink peak), cover, stroke width (max 2 x distance transform across a ±4 px band) and colour. Windows with cover ≥ 0.4 are candidates. The border style = mode of candidate stroke widths (1 px bins, smoothed); accept windows within ±30% of it and ΔE ≤ 20 of their median colour. A thinner or differently coloured line inside the strip stays | strip 12 **S** (about the georef 90th percentile), 160/4 **S**, 0.4/30%/20 **R**. Measured style here: 9.5 px |
| river | strip ±45 px around each shifted Natural Earth river; a connected ink component is river if its extent *along* the river inside the strip is ≥ 4 x 45 px. Dashed lines are many short pieces and fail this | 45 **S**, 4x **R**. All 14 rivers on the sheet |
| legend | each symbol template at scales 0.85/1/1.15: score = mean ink under the template − mean ink in a 2 px ring around it; local maxima ≥ 0.6 are matches, removed with the template dilated 2. Solid areas: opening with a square of side 2 x target width. Stipple: components 0.3-3 x the swatch dot size with ≥ 6 such within 3 x the swatch dot spacing | 0.6 **R**, scales **R**. Matches here: 35 wells, 0 dry holes, 18 refineries, 1 new pool |
| grid | the grid layer from step 10, grown 1 | |

About 13% of the original ink remains.

### Pieces

Connected components of the remaining ink (8-connected), by PCA: centre, axis, length
(√(12·var) along the axis), sw = pixels / length, mean Lab from the original scan. Kept if
length ≥ max(6, 0.25 x legend dash) and length ≥ 2 sw (elongated). 1879 pieces here.

### Links

Candidate: two ends of different pieces within 3P (**C**, `LINK_PERIODS`). Refused if:

- either end points more than ~26° away from the other (cos < 0.9), or the axes differ > 25° **R**
- the straight gap passes over another kept piece whose axis is within 25° of the gap direction
  (that piece is the next link; without this rule every dashed line is traced twice, as two
  interleaved chains)
- the gap contains a run of blank paper (not ink, not removed) longer than 1.5P **R**, or more
  than 8P of removed ground (**C**)

Score, target against chance; each term is a density ratio over the same variable and range,
so the sum is a dimensionless log-odds:

| Term | Target | Chance |
|---|---|---|
| turn θ | half-normal σ = 8° | uniform 0-25° |
| sideways offset | half-normal σ = max(sw, 2) | uniform over the 25° wedge at distance d |
| length ratio (log) | normal σ = 0.35 | uniform over ±1.5 |
| sw ratio (log) | normal σ = 0.25 | uniform over ±1.5 |
| blank part of the gap | log-normal around the legend gap, σ = 0.6 | log-uniform up to 1.5P |
| removed ground | −0.3 per P | |
| existence | a next piece with p = 0.8 | 1 − exp(−λ·A), λ = density of piece ends within 300 px, A = the search wedge |

A "solid" score is computed alongside (a continuous line broken only by removals): allowed only
if ≤ 2 px of the gap is blank; turn σ = 15°, removed −0.1 per P. Link weight = max(target,
solid); kept if > 0.

### Matching

Maximum-weight matching (networkx) over all piece ends, per connected group; groups over 400
ends (**C**) are cut at their weakest 10% of links until small enough (0 cuts here: largest
group 6 once skip links were refused). Each end joins at most one other. Chains are walked
end to end into paths.

### Classes

Per path, target log-odds = Σ pieces [width term (normal σ = 0.25 x legend width) + colour term
(ΔE half-normal σ = 12) + log 2 − the sheet's own piece-width density (smoothed histogram)
− log(1/60)] + Σ links (target term) + dash term (log of median piece length / legend dash,
normal σ = 0.7: dash length varies by draftsman) + **path width term** (log of median sw /
legend width, normal σ = 0.2, against uniform ±1.5: pen weight does not vary, and a grid
line's remnants line up well enough to outvote a wrong width piece by piece). Solid log-odds =
Σ links (solid term). Class: **target** if target > 8 **R** (`MARGIN`) and > solid; else
**solid** if solid > 8; else **chance**.

Output `explain_paths.gpkg` layer `paths`: 174 paths, 76 target, on this run.

## 14. Join each line (`bridge.py`)

Input: target paths only (a path that can't pass as target alone is never glued onto one),
minus any lying entirely inside the legend's box (all entry bboxes ± 2P): the swatches are not
map features. Each path end has a point and an outward direction (its last dash).

Candidate joins: two ends of different paths within 20P (**C**). With d = distance,
sw = the thicker of the two:

- **close** if d < 2 sw (the direction between the ends is noise); **corner** if d ≤ P
- not a corner: both ends must point within 25° of the other end **R**
- a corner whose ends point the same way (cos > 0.5) is two lines side by side: refused
- **runs along another line:** sample the straight gap every 3 px; if more than 25% of samples
  lie within sw of another target path running within 25° of the gap direction, refused (the
  join would trace that line a second time)
- gap evidence across a band ±sw wide, where ink or removed ground anywhere across the band
  counts, except pixels within sw of another target line (already explained; they count as
  neither). A blank run longer than 1.5P: refused

Score (same form as links): turn = how far each end points away from the other end (the
larger; a line may bend inside the gap) half-normal σ = 8° vs uniform 25°; sideways offset
(the smaller seen from either end) σ = sw; sw ratio σ = 0.25; dash ratio σ = 0.35; paper beyond
what the dash pattern explains, −(blank − (gap/P)·(non-removed length)) / P; removed −0.3 per P;
existence as for links, λ = density of path ends within 300 px, A = the 25° wedge to
max(d, P), or π·max(d, sw)² for a corner. Corners get no turn or offset term. Joins with score > 0 go
to a maximum-weight matching over path ends; chains are walked into lines (a closed loop is
opened at one path). A line is kept if Σ path scores + Σ join scores > 8.

Then, per line end, a cause: **junction** if the end lies within 1.5 sw of another line's
segment crossing at more than 25°, or its 1.5P extension crosses one; **sheet edge** if within
2P of the edge; **into removed X** if more than half of the stretch from 3 px to 5P ahead is
removed ground (X = the stage that removed most of it); **unlinked ink ahead** if more than 30% is
leftover ink; **paper ahead** if it holds a blank run ≥ 3P; **other**.

Each line is simplified (Douglas-Peucker, tolerance = its median sw) into the `corrected`
layer. This sheet: 75 target paths → 33 lines from 42 joins.

## Output files

| File | Contents |
|---|---|
| `out/legend.json` | list of entries: `label`, `kind` (line/area/symbol), `bbox` [row0, col0, row1, col1], `colour_lab`, `width`; lines add `pattern`, `dash`, `gap`; areas `fill`; symbols `template`; `target` |
| `text/sheet.json` | list of readings: `text`, `score`, `poly` (sheet px), `edge` |
| `out/text_extended.json` | list of names: `cells` (one box per letter), `pieces` (absorbed reading indexes), `name`, `letters` |
| `out/georef_fix.json` | `deg`, `coef` (x then y), `W`, `H` |
| `out/explain_removed.tif` | 1 text, 2 boundary, 3 river, 4 legend symbols and areas, 5 grid, 0 not removed |
| `out/explain_remains.tif` | the cleaned RGB image the classifier saw |
| `out/explain_paths.gpkg` `paths` | every path: `n` pieces, `llr_target`, `llr_solid`, `cls` (target/solid/chance), `dash`, `sw`, `removed_gap` |
| `out/explain_joined.gpkg` `corrected` | the answer: one line per pipeline, `line`, `n_paths`, `members` (path fids) |
| `out/explain_joined.gpkg` `lines` | the same lines unsimplified, plus `score` |
| `out/explain_joined.gpkg` `joins` | each bridged gap: `a`, `b` (path fids), `gap_p` (in periods), `removed` (fraction), `score` |
| `out/explain_joined.gpkg` `ends` | each open end: `line`, `cause` |
| `out/explain_joined.gpkg` `junctions` | `line`, `meets` |

## Debts (the S values, and other known gaps)

- The 110 grey threshold, `common.LEG`, the 11 px pool square, the 118 px grid stamp, the
  pool-square and strip widths in text and borders, and the literal sheet extent should all
  come from the sheet (histograms, `legend.json`, the raster's own bounds).
- Two ink masks exist; one measured mask should serve every step.
- Dry-hole symbols match zero times: the legend template doesn't fit their map versions.
- No ground truth or scorer yet. The metrics a scorer should report are in
  `.claude/skills/read-map/prompts.yaml`.
