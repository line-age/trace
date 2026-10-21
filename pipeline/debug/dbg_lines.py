import sys, numpy as np
from osgeo import gdal, ogr
from explain import SHEET
from common import OUT
gt = gdal.Open(SHEET).GetGeoTransform()
J = ogr.Open(str(OUT / "explain_joined.gpkg"))
P = ogr.Open(str(OUT / "explain_paths.gpkg"))
paths = {f.GetFID(): dict(cls=f.GetField("cls"), llr=f.GetField("llr_target"), n=f.GetField("n"), sw=f.GetField("sw"), dash=f.GetField("dash")) for f in P.GetLayerByName("paths")}
pix = lambda g: [((g.GetX(k) - gt[0]) / gt[1], (g.GetY(k) - gt[3]) / gt[5]) for k in range(g.GetPointCount())]
joins = [(f.GetField("a"), f.GetField("b"), f.GetField("gap_p"), f.GetField("removed"), f.GetField("score")) for f in J.GetLayerByName("joins")]
for f in J.GetLayerByName("corrected"):
    if f.GetField("line") in map(int, sys.argv[1:]):
        m = list(map(int, f.GetField("members").split(",")))
        xy = np.array(pix(f.GetGeometryRef()))
        print("line", f.GetField("line"), "from", xy[0].round(), "to", xy[-1].round())
        for k in m: print("   path", k, paths[k])
        for j in joins:
            if j[0] in m and j[1] in m: print("   join", j)
