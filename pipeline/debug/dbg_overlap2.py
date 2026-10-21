import numpy as np
from osgeo import gdal, ogr
from scipy.spatial import cKDTree
from explain import SHEET
from common import OUT
gt = gdal.Open(SHEET).GetGeoTransform()
J = ogr.Open(str(OUT / "explain_joined.gpkg"))
def pix(g): return np.array([((g.GetX(k) - gt[0]) / gt[1], (g.GetY(k) - gt[3]) / gt[5]) for k in range(g.GetPointCount())])
def dens(xy):
    pts = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(int(np.linalg.norm(b - a) / 3), 1); pts += list(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.array(pts)
L = {f.GetField("line"): dens(pix(f.GetGeometryRef())) for f in J.GetLayerByName("corrected")}
T = {k: cKDTree(v) for k, v in L.items()}
for a in L:
    for b in L:
        if a >= b: continue
        da, _ = T[b].query(L[a]); db, _ = T[a].query(L[b])
        f = max((da < 12).mean(), (db < 12).mean())
        if f > 0.15:
            both = np.r_[da[da < 12], db[db < 12]]
            print(f"lines {a} {b}: {f:.0%} of the shorter within 12 px, median gap {np.median(both):.1f} px, overlap length {min((da<12).sum(), (db<12).sum())*3} px")
