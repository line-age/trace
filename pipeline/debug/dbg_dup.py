import numpy as np
from osgeo import gdal, ogr
from scipy import ndimage as ndi
from explain import SHEET, ink_mask
from common import OUT
gt = gdal.Open(SHEET).GetGeoTransform()
ds = ogr.Open(str(OUT / "explain_paths.gpkg")); lyr = ds.GetLayerByName("paths")
xy = {}
for f in lyr:
    if f.GetFID() in (5, 8):
        g = f.GetGeometryRef(); ll = np.array([g.GetPoint_2D(k) for k in range(g.GetPointCount())])
        xy[f.GetFID()] = np.c_[(ll[:, 0] - gt[0]) / gt[1], (ll[:, 1] - gt[3]) / gt[5]]
for k, v in xy.items(): print(k, len(v), v[:6].round(1).tolist())
c = xy[5][len(xy[5]) // 2].astype(int)
x0, y0 = c[0] - 60, c[1] - 60
rem = gdal.Open(str(OUT / "explain_remains.tif")).ReadAsArray(int(x0), int(y0), 120, 120)
rgb = np.ascontiguousarray(rem.transpose(1, 2, 0))
ink, _ = ink_mask(rgb)
lab, n = ndi.label(ink, np.ones((3, 3)))
print("components in crop", n)
for i, sl in enumerate(ndi.find_objects(lab), 1):
    print(i, sl, (lab[sl] == i).sum())
from PIL import Image
big = np.kron(rgb, np.ones((5, 5, 1), np.uint8))
for k, col in ((5, (255, 0, 255)), (8, (0, 200, 0))):
    for x, y in xy[k]:
        X, Y = int((x - x0) * 5), int((y - y0) * 5)
        if 0 <= X < 600 and 0 <= Y < 600: big[max(Y-3,0):Y+3, max(X-3,0):X+3] = col
Image.fromarray(big).save("out/dbg_dup.png")
