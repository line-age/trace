"""County boundaries as of 1 July 1937, clipped to the sheet: $TRACE_WORK/out/borders_1937.gpkg.

From the Newberry Atlas of Historical County Boundaries (fetch_data.sh), IL / IN / KY. A county
is kept if START_N <= 19370701 <= END_N (yyyymmdd integers). Layers: county_lines_1937 (one
multiline, the union of all county outlines) and counties_1937 (polygons with state and name).
"""

import sys
from pathlib import Path

from osgeo import ogr, osr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, OUT  # noqa: E402

DATE = 19370701
SHAPES = {st: f"{st}/{st}_AtlasHCB/{st}_Historical_Counties/{st}_Historical_Counties.shp" for st in ("IL", "IN", "KY")}
SHEET_BOX = "POLYGON((-91.12 39.64,-86.93 39.64,-86.93 36.73,-91.12 36.73,-91.12 39.64))"  # this sheet's extent


def main():
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    sheet = ogr.CreateGeometryFromWkt(SHEET_BOX)
    polys, keep = [], []
    for st, p in SHAPES.items():
        ds = ogr.Open(str(DATA / "ahcb" / p))
        keep.append(ds)
        lyr = ds.GetLayer()
        src = lyr.GetSpatialRef()
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tr = osr.CoordinateTransformation(src, dst)
        lyr.SetAttributeFilter(f"START_N <= {DATE} AND END_N >= {DATE}")
        n = 0
        for f in lyr:
            g = f.GetGeometryRef().Clone()
            g.Transform(tr)
            if g.Intersects(sheet):
                polys.append((st, f.GetField("NAME"), g))
                n += 1
        print(f"{st}: {n} counties on the sheet in 1937")
    lines = ogr.Geometry(ogr.wkbMultiLineString)
    for _, _, g in polys:
        b = g.Boundary()
        if b.GetGeometryType() in (ogr.wkbMultiLineString, ogr.wkbMultiLineString25D):
            for i in range(b.GetGeometryCount()):
                lines.AddGeometry(b.GetGeometryRef(i).Clone())
        else:
            lines.AddGeometry(b.Clone())
    merged = (lines.UnaryUnion() if hasattr(lines, "UnaryUnion") else lines.Union(lines)).Intersection(sheet)
    out = OUT / "borders_1937.gpkg"
    out.unlink(missing_ok=True)
    ods = ogr.GetDriverByName("GPKG").CreateDataSource(str(out))
    ly = ods.CreateLayer("county_lines_1937", dst, ogr.wkbMultiLineString)
    f = ogr.Feature(ly.GetLayerDefn())
    f.SetGeometry(merged)
    ly.CreateFeature(f)
    lp = ods.CreateLayer("counties_1937", dst, ogr.wkbMultiPolygon)
    for k in ("state", "name"):
        lp.CreateField(ogr.FieldDefn(k, ogr.OFTString))
    for st, nm, g in polys:
        f = ogr.Feature(lp.GetLayerDefn())
        f.SetGeometry(ogr.ForceToMultiPolygon(g))
        f.SetField("state", st)
        f.SetField("name", nm)
        lp.CreateFeature(f)
    ods = None
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
