"""Removal stages: each finds one kind of feature and removes it from the working State.

None of these knows what the map is about. Text comes from a learned spotter; boundaries and
rivers from outside data, matched with fuzziness and confirmed in the ink with a style measured
on the sheet; symbols and areas from the sheet's own legend; the grid from shape.
"""

import json
import os
import subprocess

import numpy as np
from osgeo import ogr, osr
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import oaconvolve
from scipy.spatial import cKDTree

from border_fit import boundary_px, chain_parts, resample
from borders import shift_field
from common import DATA, HERE, OUT, S
from explain import rgb_to_lab
from extend_text import geom, letters, strip

# the text spotter runs in its own venv (spotter/setup.sh); its code is in this repo
SPOTTER_PY = os.environ.get("TRACE_SPOTTER_PY", str(S / "spotter/venv/bin/python"))
SPOT_CROPS = HERE / "spotter/spot_crops.py"


def polys_mask(shape, polys, grow=2):
    img = Image.new("1", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        d.polygon([tuple(map(float, q)) for q in p], fill=1)
    m = np.asarray(img, bool)
    return ndi.binary_dilation(m, iterations=grow) if grow else m


# ---------------------------------------------------------------- text

def grow_bound(ink, r, extra=6.0):
    """Grow one detection along its baseline to the full extent of its character sequence.

    Letters are found in a straightened strip (tall inked columns: a line through the letter
    tops cannot hide them). From the detection's own letters, the next letter is accepted if
    its gap is no more than 1.3 x the word's letter pitch (or 0.8 x the height for a single
    character), so tight words and letter-spaced names grow by the same rule.
    """
    g = dict(geom(r))
    s = r["text"].strip("#")
    if len(s) <= 1:  # one character gives no baseline angle; assume the sheet's upright text
        c = g["o"] + g["u"] * g["L"] / 2
        g.update(u=np.array([1.0, 0.0]), n=np.array([0.0, 1.0]), o=c - [g["L"] / 2, 0])
    h = max(g["h"], 6)
    reach = extra * h + g["L"]
    (ink_s,), t, b = strip([ink], g, -reach, g["L"] + reach)
    blobs = letters(ink_s > 0.5, b, h)
    if not blobs:
        return None
    spans = [(t[a], t[z - 1]) for a, z in blobs]
    own = [k for k, (a, z) in enumerate(spans) if z >= -0.2 * h and a <= g["L"] + 0.2 * h]
    if not own:
        return None
    pitch = g["L"] / max(len(s), 1)
    max_gap = 1.3 * pitch if len(s) > 1 else 0.8 * h
    lo, hi = min(own), max(own)
    while hi + 1 < len(spans) and spans[hi + 1][0] - spans[hi][1] <= max_gap:
        hi += 1
    while lo - 1 >= 0 and spans[lo][0] - spans[lo - 1][1] <= max_gap:
        lo -= 1
    t0, t1 = min(spans[lo][0], 0), max(spans[hi][1], g["L"])
    o, u, n = g["o"], g["u"], g["n"]
    return [o + u * t0 - n * 0.6 * h, o + u * t1 - n * 0.6 * h, o + u * t1 + n * 0.6 * h, o + u * t0 + n * 0.6 * h], (t1 - t0) - g["L"] > 0.5 * h


def text_bounds(ink, spots, extended):
    """Bounds for every detection: the extension's span for letter-spaced names it rebuilt,
    the grown bound otherwise. Returns (bounds, which need re-detection)."""
    used = {int(k) for lb in extended for k in lb["pieces"].split(",")}
    bounds, redo = [], []
    for lb in extended:
        cells = np.array([q for c in lb["cells"] for q in c])
        if len(cells):
            a, b = cells.min(0), cells.max(0)
            bounds.append([a, [b[0], a[1]], b, [a[0], b[1]]])
            redo.append(True)
    for k, r in enumerate(spots):
        if k in used:
            continue
        gb = grow_bound(ink, r)
        if gb is None:
            bounds.append(np.array(r["poly"]))
            redo.append(False)
        else:
            bounds.append(np.array(gb[0]))
            redo.append(gb[1])
    return bounds, redo


def redetect(image_rgb, bounds, tag):
    """Run the spotter (its own venv) inside each bound, on the given working image."""
    img_path, crops_path, out_path = OUT / f"work_{tag}.png", OUT / f"crops_{tag}.json", OUT / f"redetect_{tag}.json"
    Image.fromarray(image_rgb).save(img_path)
    crops = []
    for p in bounds:
        p = np.asarray(p, float)
        a, b = p.min(0), p.max(0)
        crops.append([float(a[0]), float(a[1]), float(b[0] - a[0]), float(b[1] - a[1])])
    json.dump(crops, open(crops_path, "w"))
    r = subprocess.run([SPOTTER_PY, "-I", str(SPOT_CROPS), str(img_path), str(crops_path), str(out_path)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    found = json.load(open(out_path))
    keep = []
    for f in found:
        p = np.array(f["poly"])
        c = p.mean(0)
        a, b = np.asarray(bounds[f["crop"]], float).min(0), np.asarray(bounds[f["crop"]], float).max(0)
        pad = 0.3 * (b[1] - a[1])
        if a[0] - pad <= c[0] <= b[0] + pad and a[1] - pad <= c[1] <= b[1] + pad:  # inside its bound
            keep.append(f)
    return keep


# ---------------------------------------------------------------- outside features

def _windows(ink, rgb, lines, strip_px, win, half):
    """Along each prior line, per window: printed line offset, cover, stroke width, colour."""
    H, W = ink.shape
    width = 2 * ndi.distance_transform_edt(ink)
    offs = np.arange(-strip_px, strip_px + 1)
    out = []
    for line in lines:
        r = resample(line, 2.0)
        n = int(win / 2)
        for s in range(0, len(r) - n + 1, n):
            w = r[s:s + n]
            a, b = w[0], w[-1]
            u = (b - a) / (np.linalg.norm(b - a) + 1e-9)
            nrm = np.array([-u[1], u[0]])
            pts = w[:, None, :] + offs[None, :, None] * nrm
            x, y = np.round(pts[..., 0]).astype(int), np.round(pts[..., 1]).astype(int)
            if x.min() < 0 or y.min() < 0 or x.max() >= W or y.max() >= H:
                continue
            prof = ink[y, x].mean(0)
            k = int(np.argmax(prof))
            cols = slice(max(k - half, 0), k + half + 1)
            band = ink[y[:, cols], x[:, cols]]
            wb = width[y[:, cols], x[:, cols]].max(1)
            px = rgb[y[:, cols], x[:, cols]][band]
            out.append(dict(w=w, nrm=nrm, k=k, off=int(offs[k]), cover=float(prof[k]),
                            stroke=float(np.median(wb[wb > 0])) if (wb > 0).any() else 0.0,
                            lab=rgb_to_lab(px).mean(0) if len(px) else np.array([100.0, 0, 0]),
                            sel=(y[:, max(k - half - 2, 0):k + half + 3], x[:, max(k - half - 2, 0):k + half + 3])))
    return out


def boundary_stage(state, gt, strip_px=12, win=160, half=4, min_cover=0.4):
    """1937 boundaries: fuzzy strip, then the dominant printed line style inside the strips.

    The style is measured, not given: the mode of the stroke widths of the windows that hold a
    clear line, with the median colour of those windows. A window is boundary if its line
    matches that style (width within 30%, colour within delta E 20). A thinner or other-coloured
    line in the strip (a pipeline running along a county line) is left alone.
    """
    shift = shift_field(json.load(open(OUT / "georef_fix.json")))
    lines = [ln + shift(ln) for ln in chain_parts(boundary_px(gt))]
    wins = [w for w in _windows(state.ink, state.rgb, lines, strip_px, win, half) if w["cover"] >= min_cover]
    strokes = np.array([w["stroke"] for w in wins])
    hist = ndi.gaussian_filter1d(np.histogram(strokes, bins=np.arange(0, 30))[0].astype(float), 1)
    mode = float(np.argmax(hist) + 0.5)
    near = [w for w in wins if abs(w["stroke"] - mode) <= 0.3 * mode]
    lab0 = np.median([w["lab"] for w in near], axis=0)
    acc = [w for w in near if np.linalg.norm(w["lab"] - lab0) <= 20]
    m = np.zeros(state.ink.shape, bool)
    for w in acc:
        y, x = w["sel"]
        m[y, x] = True
    n = state.remove("boundary", m & state.ink)
    return dict(style_width=round(mode, 1), style_lab=[round(float(v), 1) for v in lab0], windows=len(wins), accepted=len(acc),
                removed_px=n, found=[(w["w"] + w["off"] * w["nrm"]) for w in acc])


def river_px(gt):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    sheet = ogr.CreateGeometryFromWkt("POLYGON((-91.12 39.64,-86.93 39.64,-86.93 36.73,-91.12 36.73,-91.12 39.64))")
    out = []
    for f in ("ne_10m_rivers_lake_centerlines", "ne_10m_rivers_north_america"):
        ds = ogr.Open(str(DATA / f"ne/{f}/{f}.shp"))
        lyr = ds.GetLayer()
        lyr.SetSpatialFilter(sheet)
        for feat in lyr:
            g = feat.GetGeometryRef()
            parts = [g.GetGeometryRef(i) for i in range(g.GetGeometryCount())] if g.GetGeometryCount() else [g]
            for p in parts:
                xy = np.array(p.GetPoints())[:, :2]
                out.append((feat.GetField("name"), np.c_[(xy[:, 0] - gt[0]) / gt[1], (xy[:, 1] - gt[3]) / gt[5]]))
    return out


def river_stage(state, gt, strip_px=45, min_along=4.0):
    """Rivers: a wide fuzzy strip around each Natural Earth river (rivers move, and the data is
    generalized), then the expected feature inside it: a long CONTINUOUS stroke following the
    river (its along-river extent inside the strip >= min_along x the strip half-width). A
    dashed line inside the strip is many short pieces and is not taken."""
    shift = shift_field(json.load(open(OUT / "georef_fix.json")))
    m = np.zeros(state.ink.shape, bool)
    lab, _ = ndi.label(state.ink, np.ones((3, 3)))
    found = []
    for name, line in river_px(gt):
        line = line + shift(line)
        r = resample(line, 2.0)
        if len(r) < 3:
            continue
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))]
        a, b = np.floor(r.min(0) - strip_px).astype(int), np.ceil(r.max(0) + strip_px).astype(int)
        a = np.maximum(a, 0)
        b = np.minimum(b, [state.ink.shape[1] - 1, state.ink.shape[0] - 1])
        sub = lab[a[1]:b[1] + 1, a[0]:b[0] + 1]
        ys, xs = np.nonzero(sub)
        if not len(ys):
            continue
        pts = np.c_[xs + a[0], ys + a[1]]
        dist, idx = cKDTree(r).query(pts)
        inside = dist <= strip_px
        pts, idx, ids = pts[inside], idx[inside], sub[ys[inside], xs[inside]]
        for comp in np.unique(ids):
            sel = ids == comp
            along = arc[idx[sel]]
            if along.max() - along.min() >= min_along * strip_px:
                m[pts[sel, 1], pts[sel, 0]] = True
                found.append(name)
    n = state.remove("river", m, grow=1)
    return dict(removed_px=n, rivers=sorted(set(found)))


# ---------------------------------------------------------------- legend symbols and areas

def legend_stage(state, legend, orig_ink, scales=(0.85, 1.0, 1.15), thr=0.6, kinds=("symbol", "area"), log=None):
    """Every non-target legend entry explains its own shapes: symbols by template, solid areas
    by a square opening sized from the target line's weight, stipple by clusters of dots the
    size of the swatch's dots."""
    report = {}
    line_w = max(e["width"] for e in legend if e["kind"] == "line")
    for e in legend:
        if e["target"] or e["kind"] not in kinds:
            continue
        if e["kind"] == "symbol":
            T0 = np.array(e["template"], bool)
            m = np.zeros(state.ink.shape, bool)
            hits = 0
            for sc in scales:
                T = np.asarray(Image.fromarray(T0.astype(np.uint8) * 255).resize(
                    (max(3, int(T0.shape[1] * sc)), max(3, int(T0.shape[0] * sc))), Image.NEAREST)) > 127
                bg = ndi.binary_dilation(T, iterations=2) & ~T
                f = state.ink.astype(np.float32)
                # overlap-add: far faster than one sheet-sized FFT for a small template
                on = oaconvolve(f, T[::-1, ::-1].astype(np.float32), mode="same") / T.sum()
                off = oaconvolve(f, bg[::-1, ::-1].astype(np.float32), mode="same") / max(bg.sum(), 1)
                score = on - off
                peaks = (score >= thr) & (score == ndi.maximum_filter(score, size=T.shape))
                ys, xs = np.nonzero(peaks)
                hits += len(ys)
                hy, hx = T.shape[0] // 2, T.shape[1] // 2
                for y, x in zip(ys, xs):
                    y0, x0 = y - hy, x - hx
                    if y0 < 0 or x0 < 0 or y0 + T.shape[0] > m.shape[0] or x0 + T.shape[1] > m.shape[1]:
                        continue
                    m[y0:y0 + T.shape[0], x0:x0 + T.shape[1]] |= ndi.binary_dilation(T, iterations=2)
            report[e["label"]] = dict(kind="symbol", matches=hits, removed_px=state.remove(f"legend:{e['label']}", m, grow=0))
            if log:
                log(f"legend {e['label']}: {hits} matches")
        elif e["kind"] == "area" and e.get("fill") == "solid":
            side = int(round(2 * line_w)) | 1
            m = ndi.binary_dilation(ndi.binary_opening(state.ink, np.ones((side, side), bool)), iterations=3)
            report[e["label"]] = dict(kind="area", square=side, removed_px=state.remove(f"legend:{e['label']}", m, grow=0))
        elif e["kind"] == "area" and e.get("fill") == "stipple":
            y0, x0, y1, x1 = e["bbox"]
            sw = orig_ink[y0:y1, x0:x1]
            lb, n = ndi.label(sw)
            if n < 3:
                continue
            dot = float(np.median(ndi.sum(sw, lb, range(1, n + 1))))
            cs = np.array(ndi.center_of_mass(sw, lb, range(1, n + 1)))
            spacing = float(np.median(cKDTree(cs).query(cs, k=2)[0][:, 1]))
            lab, nn = ndi.label(state.ink, np.ones((3, 3)))
            sizes = ndi.sum(state.ink, lab, range(1, nn + 1))
            small = np.nonzero((sizes >= 0.3 * dot) & (sizes <= 3 * dot))[0] + 1
            cent = np.array(ndi.center_of_mass(state.ink, lab, small))
            cnt = np.array([len(x) for x in cKDTree(cent).query_ball_point(cent, 3 * spacing)])
            dense = small[cnt >= 6]
            m = np.isin(lab, dense)
            report[e["label"]] = dict(kind="stipple", dot_px=round(dot, 1), spacing_px=round(spacing, 1), dots=int(len(dense)),
                                      removed_px=state.remove(f"legend:{e['label']}", m, grow=1))
    return report


def grid_stage(state, grid):
    """Long straight strokes at the measured tilts (survey grid / graticule)."""
    return dict(removed_px=state.remove("grid", grid & state.ink, grow=1))
