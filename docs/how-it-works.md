# How we got the 1937 pipeline map

tl;dr: we find the pipelines by removing everything on the sheet that *isn't* a pipeline, then explaining whatever ink is left. The result is 33 clean pipeline lines on the 1937 Oil & Gas Journal Illinois Basin map. Not done yet, but the main trunks and their corners look right.

## 1. Setup

- The scan was already georeferenced. We checked it against the dated 1937 county boundaries (Newberry Atlas of Historical County Boundaries): median error 7.0 px. A smooth degree-2 correction got that to 3.7 px on held-out points.
- Ink vs. paper: a pixel is ink if it's clearly darker or more saturated than the paper. The thresholds come from this sheet's own histogram, not from a constant.

## 2. Read the legend

We find the word LEGEND, then crop the swatch beside each label. The pipeline swatch gives us the target style: 34 px dashes, 5 px gaps, 6.9 px wide, near-black. The point symbols (wells, refineries) become templates for matching.

## 3. Remove everything that isn't pipeline

| Layer | How |
|---|---|
| Text | A map-trained text spotter (MapTextPipeline, Rumsey fine-tune) reads the sheet in tiles. County names are letter-spaced, so from each detection we walk the baseline one letter at a time and grow the box, then read again inside it on a copy with borders erased. We remove the letters only - pipelines run through the gaps between them. |
| County lines | The dated boundaries say where to look. We measure the printed border style inside those strips (heavy dashes, about 9.5 px) and remove only ink that matches. |
| Rivers | Natural Earth centrelines give a loose corridor; inside it we remove heavy, continuous, curving strokes. |
| Symbols | Legend templates matched across the map at three scales. |
| Survey grid | Long, thin, straight strokes, allowing a few degrees of tilt from the warp. |

## Occlusion (the important part)

Removed ink gets painted over with the local paper colour, and we keep a mask of *where* something was removed. When we later test a gap in a line, a masked pixel means "covered, can't tell" and untouched paper means "the line really stops here." Without that, every label sitting on a pipeline looks like a break.

## 4. Explain what's left (about 13% of the original ink)

- Each leftover blob becomes a piece: centre, direction, length, width, colour.
- Piece ends that face each other get a log-odds score: "pipeline" vs. "things that line up by accident." The score uses turn, sideways offset, similar length and width, and what's in the gap (paper, covered, or ink). Crowded areas raise the bar on their own.
- A global maximum-weight matching decides every link at once, so each end gets its best partner against all the alternatives.
- Each resulting path is labelled pipeline, solid line, or chance, whichever explains it best (stroke width and colour included).

## 5. Join each line's pieces

- Pieces of one pipeline are joined across any gap that isn't bare paper.
- Ends that nearly touch at an angle are a corner, so they get joined.
- We refuse two ends side by side pointing the same way (two lines, not a hairpin), joins that would run beside another line's ink, and partners much thinner than the pen (grid slivers).
- Each joined line is simplified into one clean polyline.

## Bugs we hit along the way

- Every dashed line was traced twice, as two interleaved chains taking every other dash. A link could hop over a dash and count that dash's ink as support. Fixed.
- A survey-grid line passed as pipeline. Its remnants lined up so well that alignment outvoted the wrong pen width (1.8 px vs. 6.9). Width now counts per path.
- Short "chance" paths got glued onto real lines and dragged in a town line. Stopped doing that.

## Where it stands

- 33 lines, no duplicates. Three close pairs remain and each one is a real pair of parallel pipelines about 11 px apart (checked in the scan).
- Still missing: some lines around Wood River (the Shell line heading southwest, most of Ajax), and the second line of some parallel pairs is only partly traced.
- No hand-traced ground truth yet, so no real scores - just our own spot checks. A hand trace of one or two patches would fix that.

## Where the code is

All in `pipeline/` (see the [README](../README.md) for the run order).

| Part | Code |
|---|---|
| Georeferencing check and fix | `setup/warp_sheet.py`, `border_fit.py`, `borders.py` |
| Ink and paper | `explain.py` (`ink_mask`, `paper_background`), `setup/make_ink.py` |
| Legend | `legend.py` |
| Text | `spotter/spot_sheet.py`, `spotter/spot_crops.py`, `extend_text.py`, `stages.py` (`text_bounds`, `redetect`) |
| County lines, rivers, symbols, grid | `stages.py` (`boundary_stage`, `river_stage`, `legend_stage`, `grid_stage`) |
| The removal mask | `explain.py` (`State.remove`) |
| Pieces, links, matching, classes | `classify_residual.py` |
| Joining each line, the corrected map | `bridge.py` |
| Running it all | `run_explain.py`, `run_all.sh` |
