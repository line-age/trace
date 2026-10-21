"""The first-generation ink mask: $TRACE_WORK/ink.tif (1 = ink).

Ink = mean RGB below 110, inside the sheet (alpha > 0). prep.py, border_fit.py, borders.py,
extend_text.py and learn_styles.py read this mask. The newer stages (explain.ink_mask, used by
run_explain.py and bridge.py) measure the threshold from the sheet's own histogram instead;
the fixed 110 here is a known leftover that should move to the same measured threshold.
"""

import sys
from pathlib import Path

import numpy as np
from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import S, SHEET  # noqa: E402

THRESHOLD = 110


def main():
    ds = gdal.Open(SHEET)
    a = ds.ReadAsArray()
    ink = (a[:3].astype(np.float32).mean(0) < THRESHOLD) & (a[3] > 0)
    out = gdal.GetDriverByName("GTiff").Create(str(S / "ink.tif"), ink.shape[1], ink.shape[0], 1, gdal.GDT_Byte,
                                               ["COMPRESS=DEFLATE"])
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    out.GetRasterBand(1).WriteArray(ink.astype(np.uint8))
    out = None
    print(f"ink {ink.mean():.1%} of the sheet -> {S / 'ink.tif'}")


if __name__ == "__main__":
    main()
