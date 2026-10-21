"""Georeference the sheet: $TRACE_WORK/sheet.jpg + setup/gcps.json -> $TRACE_WORK/sheet-warped.tif.

The control points were read by hand (town symbols, extremities first) for the Lineage legend
spike; ORDER = 2 was chosen there by leave-one-out cross-validation. The warp is EPSG:4326 with
cubic resampling and an alpha band (the margin outside the sheet is transparent).

Run with QGIS's Python (or any Python with GDAL). Takes a minute or two.
"""

import json
from pathlib import Path

from osgeo import gdal

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import S, SHEET  # noqa: E402

ORDER = 2
gdal.UseExceptions()


def main():
    src = S / "sheet.jpg"
    gcps = json.loads((Path(__file__).parent / "gcps.json").read_text())["gcps"]
    print(f"{len(gcps)} GCPs, polynomial order {ORDER}")
    tmp = S / "sheet-gcp.tif"
    gdal.Translate(str(tmp), str(src), format="GTiff", outputSRS="EPSG:4326",
                   GCPs=[gdal.GCP(g["lonlat"][0], g["lonlat"][1], 0, g["px"][0], g["px"][1]) for g in gcps])
    Path(SHEET).unlink(missing_ok=True)
    gdal.Warp(SHEET, str(tmp), dstSRS="EPSG:4326", resampleAlg="cubic", polynomialOrder=ORDER, dstAlpha=True,
              creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
    tmp.unlink()
    ds = gdal.Open(SHEET)
    print(f"wrote {SHEET}: {ds.RasterXSize} x {ds.RasterYSize}, geotransform {ds.GetGeoTransform()}")


if __name__ == "__main__":
    main()
