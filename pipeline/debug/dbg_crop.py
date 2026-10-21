import sys, numpy as np
from osgeo import gdal, ogr
from scipy.spatial import cKDTree
from PIL import Image
from explain import SHEET
from common import OUT
gt = gdal.Open(SHEET).GetGeoTransform()
J = ogr.Open(str(OUT / "explain_joined.gpkg"))
def pix(g): return np.array([((g.GetX(k) - gt[0]) / gt[1], (g.GetY(k) - gt[3]) / gt[5]) for k in range(g.GetPointCount())])
def dens(xy):
    pts = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(int(np.linalg.norm(b - a)), 1); pts += list(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.array(pts)
a, b, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
L = {f.GetField("line"): dens(pix(f.GetGeometryRef())) for f in J.GetLayerByName("lines")}
d, _ = cKDTree(L[b]).query(L[a]); idx = np.nonzero(d < 12)[0]
c = L[a][idx[len(idx) // 2]].astype(int)
H = 90
x0, y0 = c[0] - H, c[1] - H
rgb = gdal.Open(SHEET).ReadAsArray(int(x0), int(y0), 2 * H, 2 * H)[:3].transpose(1, 2, 0).copy()
S = 4
big = np.kron(rgb, np.ones((S, S, 1), np.uint8))
for k, col in ((a, (255, 0, 255)), (b, (0, 170, 255))):
    for x, y in L[k]:
        X, Y = int((x - x0) * S), int((y - y0) * S)
        if 0 <= X < 2 * H * S and 0 <= Y < 2 * H * S: big[max(Y - 1, 0):Y + 1, max(X - 1, 0):X + 1] = col
Image.fromarray(big).save(f"out/crop_{tag}.png"); print("centre", c)
