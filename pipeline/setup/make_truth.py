"""Create the empty ground-truth GeoPackage to hand-trace into: $TRACE_WORK/truth/1937.gpkg.

Layers (EPSG:4326, the scan's own CRS):
  patches   polygon  patch_id, role (tune | holdout)
  lines     line     patch_id, line_id, note   every pipeline drawn on the scan inside the patch,
                                               traced over sheet-warped.tif; split at branches
  nodes     point    patch_id, kind (junction | end | exits_patch), note

Trace only what the sheet shows; never use modern pipeline maps as a guide. Never overwrites an
existing file (the tracing is hours of work). Run: ./qpy setup/make_truth.py
"""

import sys
from pathlib import Path

from osgeo import ogr, osr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import S  # noqa: E402

ogr.UseExceptions()


def main():
    path = S / "truth" / "1937.gpkg"
    if path.exists():
        raise SystemExit(f"{path} exists; not touching it")
    path.parent.mkdir(parents=True, exist_ok=True)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    S_, I_ = ogr.OFTString, ogr.OFTInteger
    for name, kind, fields in (
        ("patches", ogr.wkbPolygon, dict(patch_id=S_, role=S_)),
        ("lines", ogr.wkbLineString, dict(patch_id=S_, line_id=I_, note=S_)),
        ("nodes", ogr.wkbPoint, dict(patch_id=S_, kind=S_, note=S_)),
    ):
        lyr = ds.CreateLayer(name, srs, kind)
        for f, t in fields.items():
            lyr.CreateField(ogr.FieldDefn(f, t))
    ds = None
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
