"""Shared pieces for reading the 1937 Illinois Basin sheet.

Everything works in raster pixel space of sheet-warped.tif (x = column, y = row) and only
converts to lon/lat when writing output. The legend numbers below were measured off the
"OIL PIPE LINE" swatch, not guessed.

Where things live. TRACE_WORK is a working directory for inputs and outputs (never committed;
default ../work next to this directory):

  $TRACE_WORK/sheet-warped.tif   the georeferenced scan (TRACE_SHEET overrides the path)
  $TRACE_WORK/ink.tif            ink mask (setup/make_ink.py)
  $TRACE_WORK/data/              outside data: ahcb/, ne/, gaz/ (setup/fetch_data.sh)
  $TRACE_WORK/spotter/           text-spotter venv, model source and weights (spotter/setup.sh)
  $TRACE_WORK/text/sheet.json    whole-sheet text readings (spotter/spot_sheet.py)
  $TRACE_WORK/out/               everything the pipeline writes
"""

import json
import os
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
S = Path(os.environ.get("TRACE_WORK", HERE.parent / "work")).resolve()
OUT = S / "out"
DATA = S / "data"
SPOTS = S / "text" / "sheet.json"
SHEET = os.environ.get("TRACE_SHEET", str(S / "sheet-warped.tif"))
OUT.mkdir(parents=True, exist_ok=True)

LEG = dict(dash=33.0, gap=6.5, thick=6.3)
PERIOD = LEG["dash"] + LEG["gap"]

# Symbol profiles. "trunk" is the legend swatch. "short" is NOT in the legend: it was measured
# off the Dennison gathering spur (dashes 10-14 px, gaps 5-9 px), so it is a reading of the
# map, not of its key. Short dashes look more like letter strokes, so their chains need more
# members before they count.
PROFILES = {
    "trunk": dict(dash=33.0, gap=6.5, slack_lo=14, slack_hi=16, min_chain=3),
    "short": dict(dash=12.0, gap=7.0, slack_lo=4, slack_hi=5, min_chain=4),
}

# What a run uses, from TRACE_RUN (comma-separated; TRACE_TEXT=1 still means "text"):
#   text     the text layer (text_mask.py) is owned: occlusion, and it cannot seed a chain
#   border   the 1937 county lines (borders.py) are owned the same way
#   learned  dash styles learned from the sheet (learn_styles.py) replace the two above
# Each combination writes its own outputs (m2_text_border_learned.gpkg, ...), so runs sit side
# by side, and prep's masks and dash list get a "_learned" variant.
RUN = {s for s in os.environ.get("TRACE_RUN", "").split(",") if s}
if os.environ.get("TRACE_TEXT") == "1":
    RUN.add("text")
TEXT_ON = "text" in RUN
SUFFIX = "".join(f"_{s}" for s in ("text", "border", "learned") if s in RUN)
PREP = "_learned" if "learned" in RUN else ""
MASKS, DASHES = OUT / f"masks{PREP}.npz", OUT / f"dashes{PREP}.json"

if "learned" in RUN:
    PROFILES = {k: dict(v) for k, v in json.load(open(OUT / "styles.json")).items()}
for _p in PROFILES.values():
    _p["period"] = _p["dash"] + _p["gap"]


def _unpack(path, key, shape):
    z = np.load(path)
    return np.unpackbits(z[key])[:shape[0] * shape[1]].reshape(shape).astype(bool)


def load_text(shape):
    """Ink owned by layers other than pipelines in this run: text and/or borders. All False if none."""
    owned = np.zeros(shape, bool)
    if "text" in RUN:
        owned |= _unpack(OUT / "text_mask.npz", "text", shape)
    if "border" in RUN:
        owned |= _unpack(OUT / "border_mask.npz", "border", shape)
    return owned


def anchored(chains, dashes, text, k=2):
    """Chains with at least k members outside text. Text cannot seed a chain on its own (letters
    pass as dashes), but a pipeline passing under a label keeps its dashes there: pipeline-name
    labels are printed ON their line (ILLINOIS), so dropping every dash in a label broke it."""
    return [c for c in chains if sum(not in_mask(text, dashes[i]["c"]) for i in c) >= k]


# Pipeline dashes are 4-6 px wide; county-line dashes are heavy, 9-10 px. Prep's dash test allows
# up to 1.7 x the legend thickness (10.7 px), which let county-line pieces seed pipeline tracks.
MAX_DASH_WIDTH = 8.0
MIN_DASH_WIDTH = 2.5  # thinner is a sliver of something else (a grid line the grid test missed)


def candidates(dashes, prof):
    """Bars that may be dashes of this style: up to its longest, down to half its length.

    A bar's length alone does not decide its style: a crossing letter or line clips a 34 px
    trunk dash to 23 px, and draftsmen shorten dashes in dense areas. Styles overlap here on
    purpose; rhythm_ok() then decides by the chain's spacing, which clipping does not change.
    """
    lo, hi = 0.5 * prof["dash"], prof["dash"] + prof["slack_hi"]
    return [e for e in dashes if lo <= e["length"] <= hi and MIN_DASH_WIDTH <= e.get("width", 3) <= MAX_DASH_WIDTH]


def rhythm_ok(pts, prof):
    """Does a chain's own spacing fit this style? (A step over a hidden dash is two periods,
    so the lower quartile of the spacings is used.)"""
    sp = np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1)
    return len(sp) > 0 and 0.75 * prof["period"] <= np.percentile(sp, 25) <= 1.35 * prof["period"]


def in_mask(mask, c):
    x, y = int(round(c[0])), int(round(c[1]))
    return bool(mask[min(max(y, 0), mask.shape[0] - 1), min(max(x, 0), mask.shape[1] - 1)])


# Classes in the "explained" raster, so each kind of occluder can be inspected separately.
GRID, POOL, THICK, OTHER, DASH = 1, 2, 3, 4, 5


def load_ink():
    ds = gdal.Open(str(S / "ink.tif"))
    return ds.ReadAsArray().astype(bool), ds.GetGeoTransform(), ds.GetProjection()


def line_se(length, theta):
    """A one-pixel-wide straight structuring element: the 'stamp' used for openings."""
    r = int(length) // 2
    t = np.linspace(-r, r, 4 * r + 1)
    k = np.zeros((2 * r + 1, 2 * r + 1), bool)
    k[np.round(t * np.sin(theta)).astype(int) + r, np.round(t * np.cos(theta)).astype(int) + r] = True
    return k


def explain(ink):
    """Split ink into the layers that are NOT pipeline, by shape alone.

    pools : fat blobs (anything an 11x11 square fits inside)
    thick : strokes clearly heavier than a pipeline dash (rivers, big county lettering)
    grid  : thin solid strokes running straight N-S or E-W for 3+ dash periods
    The remainder is everything else: dashes, text, symbols, and dashes fused to those.
    """
    pools = ndi.binary_dilation(ndi.binary_opening(ink, np.ones((11, 11), bool)), iterations=3) & ink
    L = int(3 * PERIOD)
    grid = (ndi.binary_opening(ink, np.ones((1, L), bool)) | ndi.binary_opening(ink, np.ones((L, 1), bool)))
    # The warp leaves survey lines a few degrees off true N-S / E-W: an upright stamp misses where
    # a line drifts, and the slivers it leaves pass as dashes (a "pipeline" along a section line).
    for deg in (-3, -2, -1, 1, 2, 3):
        for base in (0, 90):
            grid |= ndi.binary_opening(ink, line_se(L, np.radians(base + deg)))
    grid &= ~pools
    # Thickness is measured with the grid already gone: where a dash crosses a grid line the ink
    # is locally fat, and measuring first chopped every dash next to a crossing.
    dt = ndi.distance_transform_edt(ink & ~grid)
    thick = ndi.binary_dilation(dt > LEG["thick"] / 2 + 1.5, iterations=5) & ink & ~pools & ~grid
    hard = grid | pools | thick
    rest = ink & ~ndi.binary_dilation(hard, iterations=2)
    return dict(pools=pools, thick=thick, grid=grid, hard=hard, rest=rest)


def bar_stats(mask, lo=None, hi=None):
    """Length, width, centre and axis of every component of mask, by PCA of its pixels.

    For a solid bar, sqrt(12 * variance) along an axis is exactly its extent on that axis.
    """
    lab, _ = ndi.label(mask, np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) < 6 or max(h, w) > 80:
            continue
        ys, xs = np.nonzero(lab[sl] == i)
        if len(ys) < 15:
            continue
        p = np.c_[xs, ys].astype(float)
        mu = p.mean(0)
        ev, V = np.linalg.eigh(np.cov((p - mu).T))
        out.append(dict(id=i, sl=sl, c=mu + [sl[1].start, sl[0].start], ax=V[:, 1],
                        length=float(np.sqrt(12 * ev[1])), width=float(np.sqrt(12 * max(ev[0], 1e-9))),
                        npx=len(ys)))
    return lab, out


def is_dash(b, prof):
    return (prof["dash"] - prof["slack_lo"] <= b["length"] <= prof["dash"] + prof["slack_hi"]
            and b["width"] <= LEG["thick"] * 1.7)


def classify(b):
    """Which profile a component's shape fits, or None. Profiles do not overlap in length."""
    for name, prof in PROFILES.items():
        if is_dash(b, prof):
            return name
    return None


def chain(dashes, prof, cos=0.93, min_len=None):
    """Group dashes whose neighbour sits roughly one period away, along both their axes."""
    radius = 1.6 * prof["period"]
    min_len = prof["min_chain"] if min_len is None else min_len
    if not dashes:
        return []
    c = np.array([d["c"] for d in dashes])
    a = np.array([d["ax"] for d in dashes])
    # A blob too square to have a direction (a dash clipped by a crossing) only has to sit on
    # the line its neighbour points along; it cannot veto the link with its own noisy axis.
    rel = [d.get("reliable", True) for d in dashes]
    nb = [[] for _ in dashes]
    for i, j in cKDTree(c).query_pairs(radius):
        if not (rel[i] or rel[j]):
            continue
        u = c[j] - c[i]
        u /= np.linalg.norm(u)
        if ((not (rel[i] and rel[j]) or abs(a[i] @ a[j]) > cos)
                and (not rel[i] or abs(u @ a[i]) > cos) and (not rel[j] or abs(u @ a[j]) > cos)):
            nb[i].append(j)
            nb[j].append(i)
    seen, chains = set(), []
    for k in range(len(dashes)):
        if k in seen or not nb[k]:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in nb[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(comp) >= min_len:
            chains.append(order(c, comp))
    return chains


def order(c, idx):
    """Order a chain's dashes end to end by projecting onto its main axis."""
    p = c[idx]
    mu = p.mean(0)
    _, V = np.linalg.eigh(np.cov((p - mu).T)) if len(idx) > 2 else (None, np.c_[[0, 1], p[-1] - p[0]])
    t = (p - mu) @ V[:, 1]
    return [idx[k] for k in np.argsort(t)]


def to_lonlat(gt, xy):
    return [(gt[0] + x * gt[1], gt[3] + y * gt[5]) for x, y in xy]


def write_lines(path, layer, gt, feats, fields):
    """feats: list of (pixel_xy_list, {field: value}). Overwrites the layer in a GeoPackage."""
    drv = ogr.GetDriverByName("GPKG")
    ds = drv.Open(str(path), 1) if Path(path).exists() else drv.CreateDataSource(str(path))
    if ds.GetLayerByName(layer):
        ds.DeleteLayer(layer)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lyr = ds.CreateLayer(layer, srs, ogr.wkbLineString)
    for name, kind in fields.items():
        lyr.CreateField(ogr.FieldDefn(name, kind))
    for xy, attrs in feats:
        if len(xy) < 2:
            continue
        g = ogr.Geometry(ogr.wkbLineString)
        for lon, lat in to_lonlat(gt, xy):
            g.AddPoint_2D(lon, lat)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        for k, v in attrs.items():
            f.SetField(k, v)
        lyr.CreateFeature(f)
    ds = None


def write_points(path, layer, gt, feats, fields):
    """feats: list of (pixel_xy, {field: value})."""
    drv = ogr.GetDriverByName("GPKG")
    ds = drv.Open(str(path), 1) if Path(path).exists() else drv.CreateDataSource(str(path))
    if ds.GetLayerByName(layer):
        ds.DeleteLayer(layer)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lyr = ds.CreateLayer(layer, srs, ogr.wkbPoint)
    for name, kind in fields.items():
        lyr.CreateField(ogr.FieldDefn(name, kind))
    for xy, attrs in feats:
        g = ogr.Geometry(ogr.wkbPoint)
        g.AddPoint_2D(*to_lonlat(gt, [xy])[0])
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        for k, v in attrs.items():
            f.SetField(k, v)
        lyr.CreateFeature(f)
    ds = None


def save_raster(path, a, gt, proj):
    d = gdal.GetDriverByName("GTiff").Create(str(path), a.shape[1], a.shape[0], 1, gdal.GDT_Byte, ["COMPRESS=DEFLATE"])
    d.SetGeoTransform(gt)
    d.SetProjection(proj)
    d.GetRasterBand(1).WriteArray(a.astype(np.uint8))
    d.GetRasterBand(1).SetNoDataValue(0)
    d = None


def save_masks(m):
    np.savez_compressed(MASKS, **{k: np.packbits(v) for k, v in m.items()}, shape=np.array(next(iter(m.values())).shape))


def load_masks():
    z = np.load(MASKS)
    shape = tuple(z["shape"])
    n = shape[0] * shape[1]
    return {k: np.unpackbits(z[k])[:n].reshape(shape).astype(bool) for k in z.files if k != "shape"}
