"""Score the traced lines against hand-traced ground truth.

    ./qpy score.py                     score on the tuning area(s)
    ./qpy score.py --role holdout      the held-out area(s): report these, never tune on them

Inputs (defaults are for the 1937 sheet; every path can be overridden):
  --pred   out/explain_joined.gpkg:corrected     the answer, one line per pipeline
  --truth  truth/1937.gpkg:lines                 hand-traced lines (traced over the same scan)
  --areas  truth/1937.gpkg:patches               polygons with a `role` field (tune|holdout)
  --nodes  truth/1937.gpkg:nodes                 optional points, `kind` = junction|end|exits_patch
  --year Y                                       keep only truth alive in year Y (fields
                                                 time_start / time_end; for multi-sheet traces)
  --truth-where SQL                              attribute filter on the truth layer

Matching is by buffered overlap. The buffer is a relative tolerance: BUF_SW stroke widths, the
stroke width being the median `sw` of the target paths (out/explain_paths.gpkg), converted to
metres with the scan's pixel size. Truth traced over the same scan shares its georeferencing,
so the buffer only has to absorb stroke width, simplification (one stroke width, bridge.py)
and the tracer's hand. Truth from another scan (--buffer-m) must use a measured offset.
Recall and precision are also reported at 1, 2 and 4 stroke widths so the choice is visible.

Metrics (prompts.yaml `metrics`, the ones this truth supports):
  length_recall      truth length within the buffer of a traced line / truth length
  length_precision   traced length within the buffer of truth / traced length
  missing_length_km  truth length with no traced line within the buffer
  break_count        traced open ends lying on a true line, away from true ends and the area
                     edge, per 100 km of truth
  false_join_count   traced lines that run along two true lines which do not meet, plus traced
                     lines that carry on from a true line into non-target (each part at least
                     two buffers long), per 100 km of truth
  junction_recall    true junctions where the traced lines also meet: at least three traced
                     arms cross a circle of two buffers around it and they are connected inside
  target_ink_wrongly_removed
                     the target's own ink (read across each true line on the scan) that a
                     removal stage erased / all of it, in total and per stage (ink_audit.py).
                     Only for truth traced over this scan, with out/explain_removed.tif present

Pools are not scored. Inside a solid pool the scan shows no line, so neither the truth nor the
answer can know a route there. The pool mask (step 4, out/masks.npz `pools`) is cut out of each
area, and pool edges count as area edge: a line ending there is not a break. Blobs smaller
than a buffer disk are dropped (they cannot hide a stretch longer than the tolerance). Truth from
another scan (--buffer-m) has no pool mask: --no-pools.

Writes out/scores_<role>.json and out/score_errors_<role>.gpkg (missed, false, breaks,
junctions, and the ignored pools) for viewing over the scan in QGIS.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import shapely
from osgeo import gdal, ogr, osr
from shapely.ops import unary_union

from common import OUT, S, SHEET

gdal.UseExceptions()
ogr.UseExceptions()

BUF_SW = 2.0           # buffer = this many stroke widths (relative tolerance)
SWEEP_SW = (1.0, 2.0, 4.0)
METRIC_EPSG = 5070     # CONUS Albers: metres, < 1% length error over the basin
N_WORST = 3            # worst places reported (prompts.yaml `score` feedback)
GRID = 10              # worst places: the area is cut into GRID x GRID cells (reporting only)


def _srs(epsg):
    s = osr.SpatialReference()
    s.ImportFromEPSG(epsg)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def read_layer(spec, where=None, fields=()):
    """'path:layer' -> list of (shapely geometry in METRIC_EPSG, {field: value})."""
    path, _, name = str(spec).rpartition(":")
    if not path or not Path(path).exists():
        return None
    ds = ogr.Open(path)
    lyr = ds.GetLayerByName(name)
    if lyr is None:
        return None
    if where:
        lyr.SetAttributeFilter(where)
    src = lyr.GetSpatialRef() or _srs(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, _srs(METRIC_EPSG))
    defn = lyr.GetLayerDefn()
    have = {defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())}
    out = []
    for f in lyr:
        g = f.GetGeometryRef()
        if g is None:
            continue
        g = g.Clone()
        g.Transform(tr)
        attrs = {k: f.GetField(k) for k in fields if k in have}
        out.append((shapely.from_wkb(bytes(g.ExportToIsoWkb())), attrs))
    ds = None
    return out


def union(geoms):
    """Union of a list of geometries; an empty collection for an empty list."""
    return unary_union(geoms) if geoms else shapely.GeometryCollection()


def lines_of(g):
    """Flatten to a list of LineStrings."""
    if g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [g]
    if hasattr(g, "geoms"):
        return [p for q in g.geoms for p in lines_of(q)]
    return []


def alive(attrs, year):
    def yr(v):
        return None if v in (None, "") else int(str(v)[:4])
    t0, t1 = yr(attrs.get("time_start")), yr(attrs.get("time_end"))
    return (t0 is None or t0 <= year) and (t1 is None or t1 >= year)


def stroke_width_m():
    """Median stroke width of the target paths, in metres on the scan. None if not available."""
    p = OUT / "explain_paths.gpkg"
    if not p.exists() or not Path(SHEET).exists():
        return None
    ds = ogr.Open(str(p))
    lyr = ds.GetLayerByName("paths")
    lyr.SetAttributeFilter("cls = 'target'")
    sw = [f.GetField("sw") for f in lyr if f.GetField("sw")]
    ds = None
    if not sw:
        return None
    r = gdal.Open(SHEET)
    gt, W, H = r.GetGeoTransform(), r.RasterXSize, r.RasterYSize
    tr = osr.CoordinateTransformation(_srs(int(r.GetSpatialRef().GetAuthorityCode(None) or 4326)), _srs(METRIC_EPSG))
    cx, cy = gt[0] + gt[1] * W / 2, gt[3] + gt[5] * H / 2
    a = tr.TransformPoint(cx, cy)
    b = tr.TransformPoint(cx + gt[1] * 100, cy)
    c = tr.TransformPoint(cx, cy + gt[5] * 100)
    px_m = (np.hypot(b[0] - a[0], b[1] - a[1]) + np.hypot(c[0] - a[0], c[1] - a[1])) / 200
    return float(np.median(sw)) * px_m


def pool_areas(min_area_m2):
    """The step-4 pool mask as polygons in METRIC_EPSG, dropping blobs under min_area_m2.
    None if the mask or the scan is missing."""
    p = OUT / "masks.npz"
    if not p.exists() or not Path(SHEET).exists():
        return None
    z = np.load(p)
    shape = tuple(int(v) for v in z["shape"])
    pools = np.unpackbits(z["pools"])[:shape[0] * shape[1]].reshape(shape).astype(np.uint8)
    r = gdal.Open(SHEET)
    if (r.RasterYSize, r.RasterXSize) != shape:
        raise SystemExit(f"pool mask {shape} does not match the scan {(r.RasterYSize, r.RasterXSize)}")
    mem = gdal.GetDriverByName("MEM").Create("", shape[1], shape[0], 1, gdal.GDT_Byte)
    mem.SetGeoTransform(r.GetGeoTransform())
    mem.SetProjection(r.GetProjection())
    band = mem.GetRasterBand(1)
    band.WriteArray(pools)
    vds = ogr.GetDriverByName("Memory").CreateDataSource("")
    lyr = vds.CreateLayer("pools", r.GetSpatialRef())
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0)
    src = r.GetSpatialRef()
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, _srs(METRIC_EPSG))
    out = []
    for f in lyr:
        g = f.GetGeometryRef().Clone()
        g.Transform(tr)
        s = shapely.from_wkb(bytes(g.ExportToIsoWkb()))
        if s.area >= min_area_m2:
            out.append(s)
    vds = mem = None
    return union(out)


def endpoints(lines):
    pts = []
    for ln in lines:
        c = ln.coords
        pts += [shapely.Point(c[0]), shapely.Point(c[-1])]
    return pts


def degree_one_ends(lines, tol):
    """Line ends not within tol of any other line (open ends of the network)."""
    out = []
    for i, ln in enumerate(lines):
        others = union([l for j, l in enumerate(lines) if j != i])
        for p in (shapely.Point(ln.coords[0]), shapely.Point(ln.coords[-1])):
            if others.is_empty or p.distance(others) > tol:
                out.append(p)
    return out


def truth_junctions(lines, tol):
    """Points where three or more true arms meet: an end touching another line, or ends meeting."""
    ends = endpoints(lines)
    seen, out = [], []
    for p in ends:
        if any(p.distance(q) <= tol for q in seen):
            continue
        arms = 0
        for ln in lines:
            if ln.distance(p) > tol:
                continue
            a, b = shapely.Point(ln.coords[0]), shapely.Point(ln.coords[-1])
            arms += 1 if (a.distance(p) <= tol or b.distance(p) <= tol) else 2
        if arms >= 3:
            out.append(p)
            seen.append(p)
    return out


def arms_at(lines_union, p, r):
    """Number of crossings of the circle (radius r) by these lines, and whether they are one
    connected piece inside it."""
    disk = p.buffer(r)
    inside = lines_union.intersection(disk)
    crossings = lines_union.intersection(disk.exterior)
    n = len(getattr(crossings, "geoms", [crossings])) if not crossings.is_empty else 0
    parts = lines_of(inside)
    connected = len(parts) > 0 and unary_union(parts).buffer(r * 0.05).geom_type == "Polygon"
    return n, connected


def label_runs(ln, truth, truth_ids, buf, step):
    """Sample a traced line every `step` m; label each sample with the nearest true line id
    within the buffer, or None."""
    n = max(int(ln.length / step), 1)
    pts = [ln.interpolate(t) for t in np.linspace(0, ln.length, n + 1)]
    tree = shapely.STRtree(truth)
    labels = []
    for p in pts:
        idx = tree.query(p, predicate="dwithin", distance=buf)
        if len(idx) == 0:
            labels.append(None)
        else:
            k = min(idx, key=lambda i: truth[i].distance(p))
            labels.append(truth_ids[k])
    return labels, pts


def runs(labels, step):
    """Contiguous runs of equal label: list of (label, length_m)."""
    out = []
    for lab in labels:
        if out and out[-1][0] == lab:
            out[-1][1] += step
        else:
            out.append([lab, step])
    return out


def missed_truth(truth, pu, buf):
    """Truth stretches not found. A true point counts as found only if the nearest traced point
    is within the buffer AND that traced point's nearest true line is this one: a single traced
    line between two close parallels (closer than the buffer) finds one of them, not both."""
    if not truth:
        return shapely.GeometryCollection()
    if pu.is_empty:
        return unary_union(truth)
    step = buf / 4
    tree = shapely.STRtree(truth)
    out = []
    for i, ln in enumerate(truth):
        n = max(int(np.ceil(ln.length / step)), 1)
        ts = np.linspace(0, ln.length, n + 1)
        found = []
        for t in ts:
            p = ln.interpolate(t)
            q = shapely.ops.nearest_points(p, pu)[1]
            if p.distance(q) > buf:
                found.append(False)
                continue
            near = tree.query(q, predicate="dwithin", distance=buf)
            k = min(near, key=lambda j: truth[j].distance(q)) if len(near) else None
            found.append(k == i or (k is not None and truth[k].distance(q) == ln.distance(q)))
        # runs of not-found samples -> missed sub-lines (each sample stands for one step)
        k = 0
        while k < len(ts):
            if found[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(ts) and not found[j + 1]:
                j += 1
            a, b = max(ts[k] - step / 2, 0), min(ts[j] + step / 2, ln.length)
            seg = shapely.ops.substring(ln, a, b)
            if seg.length > 0:
                out.append(seg)
            k = j + 1
    return union(out)


def score(pred, truth, area, buf, true_ends=None, true_junctions=None):
    """All metrics for one area. pred, truth: lists of LineStrings (metric CRS). The area may
    have holes (ignored pools): their edges count as area edge."""
    pred = [g for ln in pred for g in lines_of(ln.intersection(area))]
    truth = [g for ln in truth for g in lines_of(ln.intersection(area))]
    pu, tu = union(pred), union(truth)
    T, P = tu.length, pu.length
    tbuf = tu.buffer(buf)
    missed = missed_truth(truth, pu, buf)
    false = pu.difference(tbuf) if not pu.is_empty else pu
    edge = area.boundary.buffer(buf)

    # breaks: traced open ends on a true line, not at a true end or the area edge. True ends are
    # the hand-marked ones in this area; with none marked here, the truth's own open ends.
    true_ends = [p for p in true_ends or [] if p.distance(area) <= buf]
    if not true_ends:
        true_ends = degree_one_ends(truth, buf)
    ends_u = union(true_ends)
    breaks = []
    for p in degree_one_ends(pred, buf):
        gap = p.distance(ends_u) if not ends_u.is_empty else float("inf")
        if p.distance(tu) <= buf and not edge.contains(p) and gap > 2 * buf:
            breaks.append((p, f"{gap:.0f} m from a true end"))

    # false joins
    step = buf / 2
    tids = list(range(len(truth)))
    joins = []
    for ln in pred:
        labels, pts = label_runs(ln, truth, tids, buf, step)
        rs = [r for r in runs(labels, step) if r[1] >= 2 * buf]
        ids = {r[0] for r in rs if r[0] is not None}
        meet = all(truth[a].distance(truth[b]) <= 2 * buf for a in ids for b in ids if a < b)
        if len(ids) >= 2 and not meet:
            joins.append((ln.interpolate(0.5, normalized=True), "two_lines"))
        elif ids and any(r[0] is None for r in rs):
            k = next(i for i, l in enumerate(labels) if l is None)
            joins.append((pts[k], "into_non_target"))

    # junctions: the hand-marked ones in this area; with none marked here, where true arms meet
    true_junctions = [p for p in true_junctions or [] if area.contains(p)]
    if not true_junctions:
        true_junctions = [p for p in truth_junctions(truth, buf) if area.contains(p)]
    jfound = []
    for p in true_junctions:
        n, conn = arms_at(pu, p, 2 * buf) if not pu.is_empty else (0, False)
        jfound.append((p, n >= 3 and conn))

    per100 = 100_000 / T if T > 0 else float("nan")
    m = dict(
        truth_km=round(T / 1000, 2),
        traced_km=round(P / 1000, 2),
        length_recall=round(1 - missed.length / T, 3) if T else None,
        length_precision=round(1 - false.length / P, 3) if P else None,
        missing_length_km=round(missed.length / 1000, 2),
        break_count=round(len(breaks) * per100, 2) if T else None,
        breaks=len(breaks),
        false_join_count=round(len(joins) * per100, 2) if T else None,
        false_joins=len(joins),
        junction_recall=round(sum(f for _, f in jfound) / len(jfound), 3) if jfound else None,
        junctions=len(jfound),
    )
    # point errors are all (point, info) so write_errors treats every point layer alike
    errs = dict(missed=lines_of(missed), false=lines_of(false), breaks=breaks, joins=joins,
                junctions=[(p, "found" if ok else "missed") for p, ok in jfound])
    return m, errs


def worst_places(area, errs, n=N_WORST):
    """Split the area into a GRID x GRID grid; the cells with the most missed + false length.
    Breaks in a cell are listed in its description but do not rank it."""
    x0, y0, x1, y1 = area.bounds
    cw, ch = (x1 - x0) / GRID, (y1 - y0) / GRID
    miss, fals = union(errs["missed"]), union(errs["false"])
    cells = []
    for i in range(GRID):
        for j in range(GRID):
            c = shapely.box(x0 + i * cw, y0 + j * ch, x0 + (i + 1) * cw, y0 + (j + 1) * ch).intersection(area)
            if c.is_empty:
                continue
            mk, fk = miss.intersection(c).length / 1000, fals.intersection(c).length / 1000
            if mk + fk > 0:
                nb = sum(c.contains(p) for p, _ in errs["breaks"])
                cells.append((mk + fk, mk, fk, nb, c.centroid))
    cells.sort(key=lambda t: -t[0])
    back = osr.CoordinateTransformation(_srs(METRIC_EPSG), _srs(4326))
    out = []
    for _, mk, fk, nb, c in cells[:n]:
        lon, lat, _ = back.TransformPoint(c.x, c.y)
        what = []
        if mk:
            what.append(f"{mk:.1f} km missed")
        if fk:
            what.append(f"{fk:.1f} km false")
        if nb:
            what.append(f"{nb} break(s)")
        out.append(dict(lon=round(lon, 5), lat=round(lat, 5), what=", ".join(what)))
    return out


def write_errors(path, errs_by_area):
    drv = ogr.GetDriverByName("GPKG")
    if Path(path).exists():
        drv.DeleteDataSource(str(path))
    ds = drv.CreateDataSource(str(path))
    back = osr.CoordinateTransformation(_srs(METRIC_EPSG), _srs(4326))
    spec = dict(missed=ogr.wkbLineString, false=ogr.wkbLineString, breaks=ogr.wkbPoint,
                joins=ogr.wkbPoint, junctions=ogr.wkbPoint, pools=ogr.wkbPolygon)
    for name, kind in spec.items():
        lyr = ds.CreateLayer(name, _srs(4326), kind)
        lyr.CreateField(ogr.FieldDefn("area", ogr.OFTString))
        points = kind == ogr.wkbPoint
        if points:
            lyr.CreateField(ogr.FieldDefn("info", ogr.OFTString))
        for aid, errs in errs_by_area.items():
            for item in errs.get(name, []):
                g, info = item if points else (item, None)
                og = ogr.CreateGeometryFromWkb(shapely.to_wkb(g))
                og.Transform(back)
                f = ogr.Feature(lyr.GetLayerDefn())
                f.SetGeometry(og)
                f.SetField("area", str(aid))
                if info is not None:
                    f.SetField("info", info)
                lyr.CreateFeature(f)
    ds = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tg = S / "truth" / "1937.gpkg"
    ap.add_argument("--pred", default=f"{OUT / 'explain_joined.gpkg'}:corrected")
    ap.add_argument("--truth", default=f"{tg}:lines")
    ap.add_argument("--areas", default=f"{tg}:patches")
    ap.add_argument("--nodes", default=f"{tg}:nodes")
    ap.add_argument("--role", default="tune", choices=["tune", "holdout"])
    ap.add_argument("--year", type=int)
    ap.add_argument("--truth-where")
    ap.add_argument("--buffer-m", type=float, help="matching buffer in metres (required for truth traced on another scan)")
    ap.add_argument("--no-pools", action="store_true", help="score inside pools too (truth from another scan)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    pred = read_layer(a.pred)
    truth = read_layer(a.truth, a.truth_where, fields=("time_start", "time_end"))
    areas = read_layer(a.areas, f"role = '{a.role}'", fields=("patch_id", "role"))
    if pred is None or truth is None or not areas:
        raise SystemExit(f"missing input: pred={pred is not None} truth={truth is not None} areas={bool(areas)} ({a.role})")
    if a.year is not None:
        truth = [t for t in truth if alive(t[1], a.year)]
    nodes = read_layer(a.nodes, fields=("kind", "patch_id"))

    sw_m = stroke_width_m()
    if a.buffer_m:
        buf, how = a.buffer_m, "given (--buffer-m)"
    elif sw_m:
        buf, how = BUF_SW * sw_m, f"{BUF_SW:g} x median target stroke width ({sw_m:.1f} m)"
    else:
        raise SystemExit("no stroke width measured (out/explain_paths.gpkg); pass --buffer-m")

    pl = [g for g, _ in pred for g in lines_of(g)]
    tl = [g for g, _ in truth for g in lines_of(g)]
    nodes = nodes or []
    t_ends = [g for g, at in nodes if at.get("kind") in ("end", "exits_patch")]
    t_junc = [g for g, at in nodes if at.get("kind") == "junction"]

    pools = None
    if not a.no_pools:
        if a.buffer_m:
            raise SystemExit("--buffer-m means truth from another scan: pass --no-pools too")
        pools = pool_areas(np.pi * buf ** 2)
        if pools is None:
            raise SystemExit("no pool mask (out/masks.npz, step 4); run prep.py or pass --no-pools")

    result = dict(role=a.role, buffer_m=round(buf, 1), buffer=how, pred=a.pred, truth=a.truth,
                  year=a.year, truth_where=a.truth_where, pools_ignored=pools is not None, areas={})
    errs_by_area = {}
    for k, (patch, at) in enumerate(areas):
        aid = at.get("patch_id") or k
        area = patch.difference(pools) if pools is not None else patch
        m, errs = score(pl, tl, area, buf, t_ends, t_junc)
        ign = patch.intersection(pools) if pools is not None else shapely.GeometryCollection()
        m["pool_area_pct"] = round(100 * ign.area / patch.area, 1)
        errs["pools"] = [g for g in getattr(ign, "geoms", [ign]) if g.geom_type == "Polygon"]
        if sw_m and not a.buffer_m:
            m["recall_precision_by_buffer"] = {
                f"{f:g}sw": [score(pl, tl, area, f * sw_m, t_ends, t_junc)[0][x] for x in ("length_recall", "length_precision")]
                for f in SWEEP_SW}
        m["worst"] = worst_places(area, errs)
        result["areas"][str(aid)] = m
        errs_by_area[aid] = errs

    removed = OUT / "explain_removed.tif"
    if not a.buffer_m and removed.exists():  # same scan: the removal raster lines up with the truth
        import ink_audit
        ta = ink_audit.audit(a.truth, a.areas, a.role, removed, a.truth_where)
        for aid, m in result["areas"].items():
            if aid in ta:
                m["target_ink_wrongly_removed"] = ta[aid]["fraction"]
                m["target_ink_removed_by_stage"] = ta[aid]["by_stage"]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"scores_{a.role}.json").write_text(json.dumps(result, indent=1))
    write_errors(out / f"score_errors_{a.role}.gpkg", errs_by_area)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
