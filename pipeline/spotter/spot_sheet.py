"""Whole-sheet text spotting: 1024 px tiles at native scale, 256 px overlap, duplicates merged.

A label cut by a tile edge comes back clipped ("##NDALE"), so a reading touching an inner tile
edge loses to an overlapping reading of the same label that does not. Writes
$TRACE_WORK/text/sheet.json.

usage: spot_sheet.py [MIN_SIZE [IMAGE [OUT_NAME]]]
  MIN_SIZE  resize short side to this; default 1024, i.e. native
  IMAGE     default the warped sheet ($TRACE_SHEET); $TRACE_WORK/out/sheet-painted.png has grid and
            chained dashes painted out
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent))  # run with python -I, which drops it
import spot  # noqa: E402

TILE, STEP, EDGE = 1024, 768, 6
SHEET = os.environ.get("TRACE_SHEET", str(spot.WORK / "sheet-warped.tif"))


def tiles(w, h):
    xs = list(range(0, max(w - TILE, 0) + 1, STEP)) + [max(w - TILE, 0)]
    ys = list(range(0, max(h - TILE, 0) + 1, STEP)) + [max(h - TILE, 0)]
    return sorted({(x, y) for x in xs for y in ys})


def touches_edge(poly, x0, y0, w, h):
    """True if the polygon reaches a tile edge that is not also the sheet edge."""
    p = np.asarray(poly)
    return bool((x0 > 0 and p[:, 0].min() < x0 + EDGE) or (y0 > 0 and p[:, 1].min() < y0 + EDGE)
                or (x0 + TILE < w and p[:, 0].max() > x0 + TILE - EDGE)
                or (y0 + TILE < h and p[:, 1].max() > y0 + TILE - EDGE))


def merge(recs):
    """Greedy: whole readings first, then by score weighted toward longer strings."""
    def key(r):
        return (r["edge"], -r["score"] * min(len(r["text"].strip("#")), 8) ** 0.5)
    kept = []
    for r in sorted(recs, key=key):
        g = Polygon(r["poly"]).buffer(0)
        if g.area == 0 or any(g.intersection(k["g"]).area > 0.5 * min(g.area, k["g"].area) for k in kept):
            continue
        r["g"] = g
        kept.append(r)
    for k in kept:
        del k["g"]
    return kept


def main(min_size, image=SHEET, out_name="sheet"):
    Image.MAX_IMAGE_PIXELS = None
    rgba = np.asarray(Image.open(image).convert("RGBA"))
    rgb = np.where(rgba[:, :, 3:] == 0, 255, rgba[:, :, :3]).astype(np.uint8)  # warp margin to paper
    h, w = rgb.shape[:2]
    pred, labels = spot.predictor(min_size), spot.labels148()
    recs, t = [], time.time()
    tl = tiles(w, h)
    for i, (x0, y0) in enumerate(tl):
        crop = rgb[y0:y0 + TILE, x0:x0 + TILE]
        if (crop < 110).mean() < 0.002:  # blank paper
            continue
        for r in spot.spot(pred, labels, crop, 1, x0, y0):
            r["edge"] = touches_edge(r["poly"], x0, y0, w, h)
            recs.append(r)
        if i % 10 == 0:
            print(f"tile {i + 1}/{len(tl)}  raw {len(recs)}  {time.time() - t:.0f}s", flush=True)
    kept = merge(recs)
    spot.TEXT.mkdir(parents=True, exist_ok=True)
    json.dump(kept, open(spot.TEXT / f"{out_name}.json", "w"))
    print(f"tiles {len(tl)}  raw {len(recs)}  kept {len(kept)}  {time.time() - t:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 1024, *a[1:3])
