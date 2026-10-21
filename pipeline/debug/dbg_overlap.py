import numpy as np
from osgeo import gdal, ogr
from scipy.spatial import cKDTree
from explain import SHEET
from common import OUT
gt = gdal.Open(SHEET).GetGeoTransform()
def load(path, layer, key):
    ds = ogr.Open(str(path)); out = {}
    for f in ds.GetLayerByName(layer):
        g = f.GetGeometryRef(); ll = np.array([g.GetPoint_2D(k) for k in range(g.GetPointCount())])
        xy = np.c_[(ll[:, 0] - gt[0]) / gt[1], (ll[:, 1] - gt[3]) / gt[5]]
        # densify every 3 px
        pts = [xy[0]]
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(int(np.linalg.norm(b - a) / 3), 1); pts += list(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
        out[f.GetField(key) if key == "line" else f.GetFID()] = (np.array(pts), f.GetField("members") if key == "line" else f.GetField("cls"))
    return out
L = load(OUT / "explain_joined.gpkg", "corrected", "line")
T = {k: cKDTree(v[0]) for k, v in L.items()}
for a in L:
    for b in L:
        if a >= b: continue
        d, _ = T[b].query(L[a][0]); fa = (d < 4).mean()
        d, _ = T[a].query(L[b][0]); fb = (d < 4).mean()
        if max(fa, fb) > 0.2: print("lines", a, b, "overlap", round(fa, 2), round(fb, 2), L[a][1], "|", L[b][1])
# same check on the original classified paths: did the classifier already double-trace?
P = load(OUT / "explain_paths.gpkg", "paths", "fid")
P = {k: v for k, v in P.items() if v[1] == "target"}
TP = {k: cKDTree(v[0]) for k, v in P.items()}
n = 0
for a in P:
    for b in P:
        if a >= b: continue
        d, _ = TP[b].query(P[a][0]); fa = (d < 4).mean()
        if fa > 0.2: n += 1; print("paths", a, b, round(fa, 2))
print("overlapping target path pairs:", n)
