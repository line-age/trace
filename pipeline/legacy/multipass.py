"""Multipass: explain the map away in rounds, each layer looking at what the others leave.

Computed once (pure shape, they depend on nothing else): grid (with the warp's tilt), fills,
heavy strokes, and the dash styles learned from the sheet. The text spotter's readings are
also fixed: it needs an image, and painting owned ink out of the image did not help it.

Then, each round, every layer is recomputed from the ORIGINAL ink minus what the OTHER layers
currently own (never erased cumulatively, so a wrong claim in one round can be handed back):

  text      letter-spaced extension, with last round's pipeline + border ink as the "cut"
            (cover, and cut from the spill test), instead of chained-dash pixels
  borders   1937 strips, with letters and last round's pipeline left out of the stroke reading
  dashes    bars from the residual with border ink taken out, so a dash fused to a border
            stroke comes free as a bar of its own
  pipeline  M2 on those dashes with text + borders owned; its tracks become owned pipeline ink.
            Only THIN ink (stroke <= MAX_DASH_WIDTH) is owned, so a track that wrongly ran
            along a county line can never take heavy border ink away from the border layer.

Rounds stop when every layer's owned set changes by less than SETTLED (share of its pixels).
Writes per round: out/multipass_r<N>.gpkg (m2_tracks, border_windows) and out/multipass_r<N>.npz
(owned masks); prints the change per layer.

Run with TRACE_RUN=text,border,learned (learned styles and their prep masks).
"""

import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import borders
import extend_text
import m2_tracker
import text_mask
from common import MAX_DASH_WIDTH, OUT, bar_stats, classify, load_ink, load_masks, write_lines

ROUNDS = 4
SETTLED = 0.02
PIPE_HALF_WIDTH = 4  # px either side of a tracked segment, then cut to thin ink


def dashes_from(rest):
    """Clean dashes in a residual, as prep would find them (learned styles)."""
    lab, bars = bar_stats(rest)
    ds = []
    for b in bars:
        b["profile"] = classify(b)
        if b["profile"]:
            ds.append(b)
    px = np.isin(lab, [d["id"] for d in ds])
    return [dict(c=list(map(float, d["c"])), ax=list(map(float, d["ax"])), length=d["length"], width=d["width"],
                 profile=d["profile"]) for d in ds], px


def pipeline_ink(feats, ink, thin):
    """Ink under tracked evidence (seed chains and dash steps; coasted steps own nothing)."""
    img = Image.new("1", (ink.shape[1], ink.shape[0]), 0)
    d = ImageDraw.Draw(img)
    for xy, a in feats:
        if a["status"] in ("seed", "dash"):
            d.line([tuple(map(float, p)) for p in xy], fill=1, width=2 * PIPE_HALF_WIDTH + 1)
    return np.asarray(img, bool) & ink & thin


def change(a, b):
    if b is None:
        return 1.0
    u = (a | b).sum()
    return float((a ^ b).sum() / u) if u else 0.0


def main():
    t0 = time.time()
    ink, gt, _ = load_ink()
    m = load_masks()
    thin = (2 * ndi.distance_transform_edt(ink) <= MAX_DASH_WIDTH) & ~m["grid"]
    prev = dict(text=None, border=None, pipe=None)
    for r in range(1, ROUNDS + 1):
        t = time.time()
        cut = None if prev["pipe"] is None else prev["pipe"] | prev["border"]
        extend_text.main(cut=cut, quiet=True)
        text, n_spot, n_cells = text_mask.build(ink.shape)
        exclude = text if prev["pipe"] is None else text | prev["pipe"]
        border, found, kinds, _ = borders.find_borders(ink, m, gt, exclude=exclude)
        rest = m["rest"] & ~border
        dashes, dash_px = dashes_from(rest)
        feats = m2_tracker.run(ink, dict(m, rest=rest, dash=dash_px), text | border, dashes, log=lambda s: None)
        pipe = pipeline_ink(feats, ink, thin)

        write_lines(OUT / f"multipass_r{r}.gpkg", "m2_tracks", gt, feats, m2_tracker.FIELDS)
        borders.write(found, gt, OUT / f"multipass_r{r}_borders.gpkg")
        np.savez_compressed(OUT / f"multipass_r{r}.npz", text=np.packbits(text), border=np.packbits(border),
                            pipe=np.packbits(pipe), shape=np.array(ink.shape))
        ch = {k: change(v, prev[k]) for k, v in dict(text=text, border=border, pipe=pipe).items()}
        st = [a["status"] for _, a in feats]
        print(f"round {r}: text {text.sum()} px ({n_cells} letter boxes), border {border.sum()} px "
              f"({kinds['heavy']} windows), dashes {len(dashes)}, tracks {len({a['track'] for _, a in feats})} "
              f"(dash steps {st.count('dash')}, coast {st.count('coast')}), pipeline ink {pipe.sum()} px | "
              f"changed: " + ", ".join(f"{k} {v:.1%}" for k, v in ch.items()) + f" | {time.time() - t:.0f}s", flush=True)
        prev = dict(text=text, border=border, pipe=pipe)
        if r > 1 and max(ch.values()) < SETTLED:
            print(f"settled after round {r}")
            break
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
