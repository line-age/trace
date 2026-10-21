import json, numpy as np
from collections import Counter
from osgeo import gdal, ogr
import bridge as br
from explain import SHEET, ink_mask
from common import OUT
ds = gdal.Open(SHEET); gt = ds.GetGeoTransform(); arr = ds.ReadAsArray()
rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0)); ink, _ = ink_mask(rgb, arr[3])
removed = gdal.Open(str(OUT / "explain_removed.tif")).ReadAsArray()
tgt = next(e for e in json.load(open(OUT / "legend.json")) if e["target"]); period, gap = tgt["dash"] + tgt["gap"], tgt["gap"]
J = ogr.Open(str(OUT / "explain_joined.gpkg"))
lines = []
for f in J.GetLayerByName("lines"):
    g = f.GetGeometryRef(); ll = np.array([g.GetPoint_2D(k) for k in range(g.GetPointCount())])
    xy = np.c_[(ll[:, 0] - gt[0]) / gt[1], (ll[:, 1] - gt[3]) / gt[5]]
    lines.append(dict(line=f.GetField("line"), xy=xy, sw=6.5, dash=34.0, cls="target", llr=f.GetField("score")))
ends = []
for L in lines:
    xy = L["xy"]
    for side, (e, f) in enumerate(((xy[0], xy[1]), (xy[-1], xy[-2]))):
        o = e - f; ends.append((L, side, e, o / np.linalg.norm(o)))
cos = np.cos(np.radians(25)); res = Counter()
for i, A in enumerate(ends):
    best = None
    for j, B in enumerate(ends):
        if B[0] is A[0]: continue
        v = B[2] - A[2]; d = np.linalg.norm(v)
        if d < 1 or d > 20 * period: continue
        u = v / d
        if A[3] @ u < cos or B[3] @ -u < cos: continue
        lat = np.linalg.norm(v - (v @ A[3]) * A[3])
        if best is None or d + 4 * lat < best[0]: best = (d + 4 * lat, j, d, lat)
    if best is None or best[1] < i: continue
    B = ends[best[1]]
    s = br.join_score(A, B, ink, removed, period, gap, 2e-5)
    k = "geometry" if s is None else ("paper run %.1fp" % (s["run"] / period) if "refused" in s else "t=%.1f turn=%.0f lat=%.0f rem=%.1fp blank=%.1fp" % (s["t"], s["turn"], s["lat"], s["rem"] / period, s["blank"] / period))
    print(A[0]["line"], B[0]["line"], "gap %.1fp lat %.0f" % (best[2] / period, best[3]), "|", k, "| at", A[2].round())
