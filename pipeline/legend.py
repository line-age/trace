"""Read the legend into class models: every entry becomes a class, the target is one of them.

  1. Find the word LEGEND among the text readings; entries are the text lines below it.
  2. An entry's swatch is the ink just left of its label (until a blank run of > 1.5 x the
     label height), on the label's line.
  3. Each swatch is typed by shape:
       line    wider than 2.5 label heights, made of bars along it: dash length, gap,
               stroke width, colour (solid if one piece)
       area    a filled blob (a large square fits inside) or a stipple (many dots)
       symbol  anything else: kept as a binary template, matched later across the sheet

Writes out/legend.json. Nothing here knows what the map is about.
"""

import json
import re

import numpy as np
from osgeo import gdal
from scipy import ndimage as ndi

from common import OUT, SPOTS
from explain import SHEET, ink_mask, rgb_to_lab



def box(r):
    p = np.array(r["poly"])
    return p.min(0), p.max(0)


def entries(spots):
    leg = [r for r in spots if re.sub(r"[^A-Z]", "", r["text"].upper()) == "LEGEND"]
    if not leg:
        raise SystemExit("no LEGEND heading found")
    lo, hi = box(max(leg, key=lambda r: r["score"]))
    h = hi[1] - lo[1]
    below = []
    for r in spots:
        a, b = box(r)
        if lo[0] - 2 * h <= a[0] <= lo[0] + 12 * h and hi[1] < a[1] <= hi[1] + 14 * h and re.search(r"[A-Za-z]{2}", r["text"]):
            below.append((r, a, b))
    lines = []
    for r, a, b in sorted(below, key=lambda x: (x[1][1] + x[2][1]) / 2):
        yc = (a[1] + b[1]) / 2
        if lines and abs(lines[-1]["yc"] - yc) < 0.5 * (b[1] - a[1]):
            ln = lines[-1]
            ln["words"].append((a[0], r["text"]))
            ln["a"], ln["b"] = np.minimum(ln["a"], a), np.maximum(ln["b"], b)
        else:
            lines.append(dict(words=[(a[0], r["text"])], a=a, b=b, yc=yc))
    # the legend is one block: stop at the first vertical gap of more than 2.5 line heights
    block = lines[:1]
    for ln in lines[1:]:
        gap = ln["a"][1] - block[-1]["b"][1]
        if gap > 2.5 * (ln["b"][1] - ln["a"][1]):
            break
        block.append(ln)
    for ln in block:
        ln["words"] = [w for _, w in sorted(ln["words"])]
    # entry labels share one left edge (a list layout); a note set in another column is not an entry
    left = np.median([ln["a"][0] for ln in block])
    return [ln for ln in block if abs(ln["a"][0] - left) <= ln["b"][1] - ln["a"][1]]


def swatch(ink, ln):
    a, b = ln["a"], ln["b"]
    h = b[1] - a[1]
    y0, y1 = int(a[1] - 0.4 * h), int(b[1] + 0.4 * h)
    x1 = int(a[0]) - 2
    x0 = max(int(a[0] - 8 * h), 0)
    sub = ink[y0:y1, x0:x1]
    cols = sub.any(0)[::-1]  # walk left from the label
    end, blank = len(cols), 0
    started = False
    for k, c in enumerate(cols):
        if c:
            started, blank = True, 0
        elif started:
            blank += 1
            if blank > 1.5 * h:
                end = k - blank
                break
    if not started:
        return None
    left = x1 - end
    m = ink[y0:y1, left:x1]
    rows, colsx = np.nonzero(m)
    return (y0 + rows.min(), left + colsx.min(), y0 + rows.max() + 1, left + colsx.max() + 1), h


def describe(rgb, ink, bbox, h):
    y0, x0, y1, x1 = bbox
    m = ink[y0:y1, x0:x1]
    lab = rgb_to_lab(rgb[y0:y1, x0:x1][m]).mean(0)
    dt = ndi.distance_transform_edt(m)
    width = float(2 * np.percentile(dt[m], 90))
    w, hgt = x1 - x0, y1 - y0
    lbl, n = ndi.label(m, np.ones((3, 3)))
    base = dict(bbox=[int(v) for v in bbox], colour_lab=[round(float(v), 1) for v in lab], width=round(width, 1))
    if w >= 2.5 * h and hgt <= 0.8 * h:
        sl = ndi.find_objects(lbl)
        runs = sorted((s[1].start, s[1].stop) for s in sl if s is not None)
        dashes = [z - a for a, z in runs]
        gaps = [runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)]
        # the run touching the label end may be clipped by the label: report the inner ones
        inner = dashes[1:-1] if len(dashes) > 3 else dashes
        base["width"] = round(float(m.sum() / max(sum(dashes), 1)), 1)  # area / length, for a line
        return dict(base, kind="line", pattern="solid" if n == 1 else "dashed",
                    dash=round(float(np.median(inner)), 1), gap=round(float(np.median(gaps)), 1) if gaps else 0.0,
                    n_pieces=int(n))
    if n <= 2 and m.sum() / m.size > 0.6 and min(w, hgt) >= 0.5 * h:
        return dict(base, kind="area", fill="solid")
    sizes = ndi.sum(m, lbl, range(1, n + 1)) if n else []
    if n >= 10 and np.median(sizes) < 0.05 * h * h:
        return dict(base, kind="area", fill="stipple", dots=int(n))
    return dict(base, kind="symbol", template=m.astype(int).tolist())


def main(target="OIL PIPE LINE"):
    spots = json.load(open(SPOTS))
    ds = gdal.Open(SHEET)
    lns = entries(spots)
    ys = [int(min(l["a"][1] for l in lns)) - 60, int(max(l["b"][1] for l in lns)) + 60]
    xs = [int(min(l["a"][0] for l in lns)) - 400, int(max(l["b"][0] for l in lns)) + 60]
    arr = ds.ReadAsArray(xs[0], ys[0], xs[1] - xs[0], ys[1] - ys[0])
    rgb = arr[:3].transpose(1, 2, 0)
    ink, _ = ink_mask(rgb, arr[3] if arr.shape[0] > 3 else None)
    # labels are not swatches: blank their boxes before looking left of them
    for r in spots:
        a, b = box(r)
        if xs[0] <= a[0] < xs[1] and ys[0] <= a[1] < ys[1]:
            ink[int(a[1] - ys[0]):int(b[1] - ys[0]) + 1, int(a[0] - xs[0]):int(b[0] - xs[0]) + 1] = False
    out = []
    for ln in lns:
        lab = " ".join(ln["words"])
        local = dict(ln, a=ln["a"] - [xs[0], ys[0]], b=ln["b"] - [xs[0], ys[0]])
        sw = swatch(ink, local)
        if sw is None:
            continue
        bbox, h = sw
        d = describe(rgb, ink, bbox, h)
        d["bbox"] = [d["bbox"][0] + ys[0], d["bbox"][1] + xs[0], d["bbox"][2] + ys[0], d["bbox"][3] + xs[0]]
        d["label"] = lab
        d["target"] = all(w in lab.upper() for w in target.upper().split())
        out.append(d)
    json.dump(out, open(OUT / "legend.json", "w"))
    for d in out:
        extra = {k: d[k] for k in ("pattern", "dash", "gap", "fill", "dots") if k in d}
        print(f"{'*' if d['target'] else ' '} {d['label']:<28} {d['kind']:<7} width {d['width']:>4} px  "
              f"Lab {d['colour_lab']}  {extra}")
    return out


if __name__ == "__main__":
    main()
