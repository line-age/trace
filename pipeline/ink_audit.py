"""How much of the target's own ink did the removal stages erase? (metric `target_ink_wrongly_removed`)

    ./qpy ink_audit.py                  the tuning area(s)
    ./qpy ink_audit.py --role holdout   the held-out area(s): report only

Target ink is found from the hand-traced truth and the scan, not from any pipeline output: at
every pixel step along a true line, the ink profile across the line (out to REACH line widths
either side) is read, and the ink run nearest the trace (within one line width of it) is taken
as the line's own ink if that run is no longer than MAX_RUN line widths. A letter, pool or river
crossing the line makes a long run and is left out: it is not target ink, and removing it is
right. Line width is the legend swatch's (`legend.json`). The hand trace may sit a few pixels
off the ink; the one-width search absorbs that.

    target_ink_wrongly_removed = target ink pixels in the removal raster / target ink pixels

reported in total and per stage (out/explain_removed.tif: 1 text, 2 boundary, 3 river, 4 legend,
5 grid). The pieces are reusable: `line_ink` and `target_ink_removed` take arrays.
"""

import argparse
import json

import numpy as np
from osgeo import gdal, ogr, osr
from PIL import Image, ImageDraw

from common import OUT, S, SHEET
from explain import ink_mask

gdal.UseExceptions()
ogr.UseExceptions()

STAGES = {1: "text", 2: "boundary", 3: "river", 4: "legend", 5: "grid"}
REACH = 3.0     # profile half-length, in line widths (R)
MAX_RUN = 1.6   # a run longer than this many line widths across the line is not the line (R)


def _resample(pts, step=1.0):
    pts = np.asarray(pts, float)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    out.append(pts[-1:])
    return np.vstack(out)


def line_ink(ink, lines_px, width, area=None):
    """Boolean mask of the ink that belongs to the given lines (pixel coords, x = column).
    `area` (bool, same shape) limits where samples are taken."""
    H, W = ink.shape
    out = np.zeros(ink.shape, bool)
    reach = int(np.ceil(REACH * width))
    offs = np.arange(-reach, reach + 1)
    centre = np.abs(offs) <= width
    for line in lines_px:
        s = _resample(line)
        if len(s) < 2:
            continue
        t = np.gradient(s, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        q = s[:, None, :] + offs[None, :, None] * nrm[:, None, :]
        x = np.round(q[..., 0]).astype(int)
        y = np.round(q[..., 1]).astype(int)
        inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        xc, yc = np.clip(x, 0, W - 1), np.clip(y, 0, H - 1)
        v = ink[yc, xc] & inside
        ok_row = np.ones(len(s), bool) if area is None else area[np.clip(np.round(s[:, 1]).astype(int), 0, H - 1),
                                                                   np.clip(np.round(s[:, 0]).astype(int), 0, W - 1)]
        for i in np.nonzero(ok_row & (v & centre).any(1))[0]:
            row = v[i]
            # runs of ink along the profile
            d = np.diff(np.r_[0, row.astype(np.int8), 0])
            starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
            mid = len(offs) // 2
            # the run nearest the trace among those touching the centre window
            best, bd = None, None
            for a, b in zip(starts, ends):
                if not centre[a:b].any():
                    continue
                dist = 0 if a <= mid < b else min(abs(a - mid), abs(b - 1 - mid))
                if bd is None or dist < bd:
                    best, bd = (a, b), dist
            if best is None:
                continue
            a, b = best
            if a == 0 or b == len(offs) or (b - a) > MAX_RUN * width:
                continue  # runs off the profile or too wide: a crossing feature, not the line
            out[yc[i, a:b], xc[i, a:b]] = True
    return out


def target_ink_removed(target, removed_cls, stages=STAGES):
    """target: bool mask of target ink; removed_cls: uint8 stage raster (0 = kept).
    Returns {target_px, removed_px, fraction, by_stage: {name: fraction of target px}}."""
    n = int(target.sum())
    r = removed_cls[target]
    by = {name: round(float((r == k).sum()) / max(n, 1), 4) for k, name in stages.items()}
    rem = int((r > 0).sum())
    return dict(target_px=n, removed_px=rem, fraction=round(rem / max(n, 1), 4), by_stage=by)


def _layer_px(spec, gt, where=None, kind="line"):
    """'path:layer' in the scan's CRS -> list of (pixel coords array(s), attrs)."""
    path, _, name = str(spec).rpartition(":")
    ds = ogr.Open(path)
    lyr = ds.GetLayerByName(name)
    if where:
        lyr.SetAttributeFilter(where)
    sheet_srs = osr.SpatialReference(wkt=gdal.Open(SHEET).GetProjection())
    sheet_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src = lyr.GetSpatialRef()
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, sheet_srs)
    out = []

    def rings(g):
        t = g.GetGeometryName()
        if t in ("LINESTRING", "LINEARRING"):
            return [g]
        return [p for i in range(g.GetGeometryCount()) for p in rings(g.GetGeometryRef(i))]
    for f in lyr:
        g = f.GetGeometryRef().Clone()
        g.Transform(tr)
        parts = [np.array([((p.GetX(i) - gt[0]) / gt[1], (p.GetY(i) - gt[3]) / gt[5]) for i in range(p.GetPointCount())])
                 for p in rings(g)]
        out.append((parts, {k: f.GetField(k) for k in f.keys()}))
    ds = None
    return out


def audit(truth_spec, areas_spec, role, removed_path=OUT / "explain_removed.tif", truth_where=None):
    """Per area of the role: target_ink_removed over that area. Needs the scan and removal raster.
    Truth must be traced over this scan (the profile is read at the trace)."""
    ds = gdal.Open(SHEET)
    gt = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0))
    ink, _ = ink_mask(rgb, arr[3] if arr.shape[0] > 3 else None)
    rem = gdal.Open(str(removed_path)).ReadAsArray()
    width = next(e for e in json.load(open(OUT / "legend.json")) if e["target"])["width"]
    truth = [p for parts, _ in _layer_px(truth_spec, gt, truth_where) for p in parts]
    res = {}
    for k, (parts, at) in enumerate(_layer_px(areas_spec, gt, f"role = '{role}'")):
        img = Image.new("1", (ink.shape[1], ink.shape[0]), 0)
        dr = ImageDraw.Draw(img)
        for ring in parts:  # one patch polygon, no holes expected
            dr.polygon([tuple(q) for q in ring], fill=1)
        area = np.asarray(img, bool)
        tgt = line_ink(ink, truth, width, area) & area
        res[str(at.get("patch_id") or k)] = target_ink_removed(tgt, rem)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tg = S / "truth" / "1937.gpkg"
    ap.add_argument("--truth", default=f"{tg}:lines")
    ap.add_argument("--areas", default=f"{tg}:patches")
    ap.add_argument("--role", default="tune", choices=["tune", "holdout"])
    ap.add_argument("--removed", default=str(OUT / "explain_removed.tif"))
    a = ap.parse_args(argv)
    print(json.dumps(audit(a.truth, a.areas, a.role, a.removed), indent=1))


if __name__ == "__main__":
    main()
