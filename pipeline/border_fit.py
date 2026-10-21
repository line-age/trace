"""How far off is the georeferencing? Measure it against the 1937 county lines, then fit a fix.

The Newberry Atlas of Historical County Boundaries gives every county as it stood on 1 July 1937
(out/borders_1937.gpkg). County lines on this sheet mostly follow survey township lines, so their
printed position is sharp. For each straight stretch of a 1937 boundary (river-following ones
are left out: rivers have moved since), the printed line is found by sliding a window across the
boundary's perpendicular and taking the offset where ink peaks. Each such measurement gives the
error along one direction only (n . shift = offset), so the fix is fitted from many of them:
a low-order polynomial shift field, kept smooth on purpose. Draftsmen nudged borders around
crowded features, and a fit that chased every measurement would bend the pipelines with them.

Every fifth measurement is held out to score the fit. Writes out/border_offsets.gpkg (points:
measured offset, fit residual, held-out flag) and out/georef_fix.json (polynomial coefficients).
"""

import json

import numpy as np
from osgeo import gdal, ogr, osr

from common import OUT, load_ink, load_masks

SEARCH = 40       # px each side of the projected boundary
WIN = 240         # px of boundary per measurement
STEP = 120
STRAIGHT = 3.0    # px: max deviation from the chord for a window to count as straight
DEG = 2           # polynomial degree of the shift field


def boundary_px(gt):
    ds = ogr.Open(str(OUT / "borders_1937.gpkg"))
    feat = ds.GetLayer("county_lines_1937").GetNextFeature()  # keep: GDAL frees the geometry with it
    g = feat.GetGeometryRef()
    parts = [g.GetGeometryRef(i) for i in range(g.GetGeometryCount())]
    out = []
    for p in parts:
        xy = np.array(p.GetPoints())[:, :2]
        px = np.c_[(xy[:, 0] - gt[0]) / gt[1], (xy[:, 1] - gt[3]) / gt[5]]
        if len(px) >= 2:
            out.append(px)
    return out


def chain_parts(parts):
    """The union leaves thousands of 2-point pieces; join them end to end into polylines."""
    key = lambda p: (round(p[0], 1), round(p[1], 1))
    ends = {}
    for k, p in enumerate(parts):
        ends.setdefault(key(p[0]), []).append((k, 0))
        ends.setdefault(key(p[-1]), []).append((k, -1))
    used, lines = set(), []
    for k in range(len(parts)):
        if k in used:
            continue
        used.add(k)
        line = list(parts[k])
        for direction in (1, -1):
            while True:
                tip = key(line[-1] if direction == 1 else line[0])
                nxt = [(j, e) for j, e in ends.get(tip, []) if j not in used]
                if len(ends.get(tip, [])) != 2 or not nxt:  # stop at junctions (county corners)
                    break
                j, e = nxt[0]
                used.add(j)
                seg = list(parts[j]) if e == 0 else list(parts[j][::-1])
                line = line + seg[1:] if direction == 1 else seg[::-1][:-1] + line
        lines.append(np.array(line))
    return lines


def resample(line, step=4.0):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    t = np.arange(0, d[-1], step)
    return np.c_[np.interp(t, d, line[:, 0]), np.interp(t, d, line[:, 1])]


def measure(ink, lines):
    H, W = ink.shape
    rows = []
    offs = np.arange(-SEARCH, SEARCH + 1)
    for line in lines:
        r = resample(line)
        n_win = int(WIN / 4)
        for s in range(0, len(r) - n_win, int(STEP / 4)):
            w = r[s:s + n_win]
            a, b = w[0], w[-1]
            u = (b - a) / (np.linalg.norm(b - a) + 1e-9)
            nrm = np.array([-u[1], u[0]])
            if np.abs((w - a) @ nrm).max() > STRAIGHT:
                continue  # curving: follows a river
            pts = w[:, None, :] + offs[None, :, None] * nrm
            x, y = np.round(pts[..., 0]).astype(int), np.round(pts[..., 1]).astype(int)
            if x.min() < 0 or y.min() < 0 or x.max() >= W or y.max() >= H:
                continue
            prof = ink[y, x].mean(0)  # share of the window's length with ink, at each offset
            k = int(np.argmax(prof))
            second = np.max(np.r_[prof[:max(k - 8, 0)], prof[k + 9:]]) if len(prof) > 17 else 0
            # a printed boundary covers most of the window (dash-dot ~70%+); require a clear peak
            if prof[k] >= 0.45 and prof[k] - second >= 0.15 and abs(offs[k]) < SEARCH:
                rows.append(dict(c=w.mean(0), n=nrm, off=float(offs[k]), peak=float(prof[k])))
    return rows


def design(c, W, H):
    x, y = c[:, 0] / W - 0.5, c[:, 1] / H - 0.5
    return np.stack([x ** i * y ** j for i in range(DEG + 1) for j in range(DEG + 1 - i)], 1)


def fit(rows, W, H, train):
    """Least squares for the shift field (dx, dy) seen only through n . shift = offset, with
    iterative down-weighting of outliers (nudged borders, a grid line taken for the border)."""
    c = np.array([r["c"] for r in rows])
    nrm = np.array([r["n"] for r in rows])
    off = np.array([r["off"] for r in rows])
    X = design(c, W, H)
    A = np.c_[X * nrm[:, :1], X * nrm[:, 1:]]
    w = np.ones(len(rows))
    for _ in range(6):
        sw = np.sqrt(w * train)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], off * sw, rcond=None)
        res = off - A @ coef
        s = 1.4826 * np.median(np.abs(res[train])) + 1e-6
        w = 1 / (1 + (res / (2.5 * s)) ** 2)
    return coef, res, A


def main():
    ink, gt, _ = load_ink()
    m = load_masks()
    H, W = ink.shape
    lines = chain_parts(boundary_px(gt))
    rows = measure(ink & ~m["pools"], lines)
    off = np.array([r["off"] for r in rows])
    train = np.ones(len(rows), bool)
    train[::5] = False
    coef, res, A = fit(rows, W, H, train)
    test = ~train
    print(f"{len(lines)} boundary polylines, {len(rows)} measurements")
    print(f"as georeferenced now:  median |offset| {np.median(np.abs(off)):.1f} px, "
          f"90th pct {np.percentile(np.abs(off), 90):.1f} px")
    print(f"after the fix (held out): median |residual| {np.median(np.abs(res[test])):.1f} px, "
          f"90th pct {np.percentile(np.abs(res[test]), 90):.1f} px")
    print(f"constant part of the shift: dx {coef[0]:+.1f} px, dy {coef[len(coef) // 2]:+.1f} px")
    json.dump(dict(deg=DEG, coef=coef.tolist(), W=W, H=H), open(OUT / "georef_fix.json", "w"))

    drv = ogr.GetDriverByName("GPKG")
    path = OUT / "border_offsets.gpkg"
    if path.exists():
        path.unlink()
    ds = drv.CreateDataSource(str(path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lyr = ds.CreateLayer("offsets", srs, ogr.wkbPoint)
    for k, t in dict(offset_px=ogr.OFTReal, residual_px=ogr.OFTReal, held_out=ogr.OFTInteger, peak=ogr.OFTReal).items():
        lyr.CreateField(ogr.FieldDefn(k, t))
    for r, e, tr in zip(rows, res, train):
        f = ogr.Feature(lyr.GetLayerDefn())
        p = ogr.Geometry(ogr.wkbPoint)
        p.AddPoint_2D(gt[0] + r["c"][0] * gt[1], gt[3] + r["c"][1] * gt[5])
        f.SetGeometry(p)
        f.SetField("offset_px", r["off"])
        f.SetField("residual_px", float(e))
        f.SetField("held_out", int(not tr))
        f.SetField("peak", r["peak"])
        lyr.CreateFeature(f)
    ds = None


if __name__ == "__main__":
    main()
