"""The border layer: 1937 county lines, confirmed by the ink.

Outside data says where to look, the ink says what is there:

  1. The Newberry 1937 county lines, moved into this scan's pixel frame with the shift field
     fitted by border_fit.py (held-out error ~4 px median, ~10 px at the 90th percentile).
  2. A strip of +-STRIP px around each line. Nothing outside a strip is ever called border.
  3. Along the line, window by window: find the printed line's exact offset (ink peak across
     the strip) and its stroke width. County lines here are HEAVY: long dashes with a dot now
     and then, ~9-10 px thick, against ~3 px for grid lines and ~6 px for pipeline. (They are
     not a clean dash-dot: a run-pattern test accepted 20 of 723 windows.) Heavy = border. A
     pipeline running along a county line inside the strip is NOT claimed, being thinner:
     pipelines follow section and county lines, and that is exactly what we must not hide.

Writes out/border_mask.npz (owned ink) and out/borders_found.gpkg (accepted windows, as lines,
with their measured offset and dot count) for checking in QGIS.
"""

import json

import numpy as np
from osgeo import ogr, osr
from scipy import ndimage as ndi

from border_fit import boundary_px, chain_parts, design, resample
from common import OUT, load_ink, load_masks, to_lonlat

STRIP = 12     # px: ~90th percentile of the fitted georeferencing error
WIN = 160      # px of boundary per decision
HALF = 4       # px: half-thickness of the band read along the printed line
HEAVY = 8.0    # px stroke width: county lines ~9-10, pipeline ~6, grid ~3
MIN_COVER = 0.4  # share of the window with ink on the printed line


def shift_field(fix):
    k = len(fix["coef"]) // 2
    cx, cy = np.array(fix["coef"][:k]), np.array(fix["coef"][k:])
    return lambda p: np.c_[design(p, fix["W"], fix["H"]) @ cx, design(p, fix["W"], fix["H"]) @ cy]


def find_borders(ink, m, gt, exclude=None):
    """Owned border ink, accepted windows, and window counts.

    exclude: ink other layers own (letters, pipeline). It is left out of both the line search
    and the stroke-width reading, so a label printed on a county line, or a pipeline running
    along one, cannot sway the decision.
    """
    H, W = ink.shape
    shift = shift_field(json.load(open(OUT / "georef_fix.json")))
    lines = [line + shift(line) for line in chain_parts(boundary_px(gt))]
    look = ink & ~m["pools"]
    if exclude is not None:
        look &= ~exclude
    width = 2 * ndi.distance_transform_edt(look)  # local stroke width at each ink pixel (x2 the DT)
    owned = np.zeros(ink.shape, bool)
    found, n_win, kinds = [], 0, dict(heavy=0, thin=0, no_line=0)
    offs = np.arange(-STRIP, STRIP + 1)
    for line in lines:
        r = resample(line, 2.0)
        n = int(WIN / 2)  # samples every 2 px
        for s in range(0, len(r) - n + 1, n):
            w = r[s:s + n]
            a, b = w[0], w[-1]
            u = (b - a) / (np.linalg.norm(b - a) + 1e-9)
            nrm = np.array([-u[1], u[0]])
            pts = w[:, None, :] + offs[None, :, None] * nrm
            x, y = np.round(pts[..., 0]).astype(int), np.round(pts[..., 1]).astype(int)
            if x.min() < 0 or y.min() < 0 or x.max() >= W or y.max() >= H:
                continue
            n_win += 1
            prof = look[y, x].mean(0)
            k = int(np.argmax(prof))
            if prof[k] < MIN_COVER:
                kinds["no_line"] += 1
                continue
            cols = slice(max(k - HALF, 0), k + HALF + 1)
            wb = width[y[:, cols], x[:, cols]].max(1)  # stroke width across the band, per sample
            inked = wb > 0
            stroke = float(np.median(wb[inked])) if inked.any() else 0.0
            if stroke < HEAVY:
                kinds["thin"] += 1
                continue
            kinds["heavy"] += 1
            sel = (slice(None), slice(max(k - HALF - 2, 0), k + HALF + 3))
            owned[y[sel], x[sel]] |= look[y[sel], x[sel]]
            found.append((w + offs[k] * nrm, dict(offset_px=int(offs[k]), stroke_px=int(round(stroke)), cover_pct=int(100 * prof[k]))))
    owned = ndi.binary_dilation(owned, iterations=1) & look
    return owned, found, kinds, n_win


def main():
    ink, gt, _ = load_ink()
    owned, found, kinds, n_win = find_borders(ink, load_masks(), gt)
    np.savez_compressed(OUT / "border_mask.npz", border=np.packbits(owned), shape=np.array(owned.shape))
    write(found, gt)
    print(f"{n_win} windows along the 1937 lines: {kinds}")
    print(f"border ink owned: {owned.sum()} px ({owned.sum() / ink.sum():.1%} of the ink)")


def write(found, gt, path=None):
    path = path or OUT / "borders_found.gpkg"
    if path.exists():
        path.unlink()
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lyr = ds.CreateLayer("border_windows", srs, ogr.wkbLineString)
    for k in ("offset_px", "stroke_px", "cover_pct"):
        lyr.CreateField(ogr.FieldDefn(k, ogr.OFTInteger))
    for xy, attrs in found:
        g = ogr.Geometry(ogr.wkbLineString)
        for lon, lat in to_lonlat(gt, [xy[0], xy[-1]]):
            g.AddPoint_2D(lon, lat)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        for k, v in attrs.items():
            f.SetField(k, int(v))  # numpy ints are not accepted
        lyr.CreateFeature(f)
    ds = None


if __name__ == "__main__":
    main()
