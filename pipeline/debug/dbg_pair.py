import sys, numpy as np
from osgeo import gdal, ogr
from scipy.spatial import cKDTree
from explain import SHEET
from common import OUT
gt = gdal.Open(SHEET).GetGeoTransform()
J = ogr.Open(str(OUT / "explain_joined.gpkg")); Pd = ogr.Open(str(OUT / "explain_paths.gpkg"))
def pix(g): return np.array([((g.GetX(k) - gt[0]) / gt[1], (g.GetY(k) - gt[3]) / gt[5]) for k in range(g.GetPointCount())])
def dens(xy):
    pts = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(int(np.linalg.norm(b - a) / 2), 1); pts += list(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.array(pts)
L = {f.GetField("line"): (dens(pix(f.GetGeometryRef())), f.GetField("members")) for f in J.GetLayerByName("lines")}
a, b = int(sys.argv[1]), int(sys.argv[2])
d, _ = cKDTree(L[b][0]).query(L[a][0])
print(a, L[a][1], "|", b, L[b][1])
print("distance a->b px: within 4:", (d < 4).mean().round(2), "within 8:", (d < 8).mean().round(2), "within 15:", (d < 15).mean().round(2), "median where <15:", np.median(d[d < 15]).round(1))
# raw dash endpoints of member paths: are they the same pieces or alternating?
P = {f.GetFID(): pix(f.GetGeometryRef()) for f in Pd.GetLayerByName("paths")}
ea = np.vstack([P[int(m)] for m in L[a][1].split(",")]); eb = np.vstack([P[int(m)] for m in L[b][1].split(",")])
dd, _ = cKDTree(eb).query(ea)
print("dash endpoints of a within 3px of an endpoint of b:", (dd < 3).mean().round(2), " within 40:", (dd < 40).mean().round(2))
