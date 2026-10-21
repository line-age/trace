import json, numpy as np
from collections import Counter
from osgeo import gdal, ogr
import bridge as br
from diagnose_breaks import read_paths
from explain import SHEET, ink_mask
from common import OUT
ds = gdal.Open(SHEET); gt = ds.GetGeoTransform(); arr = ds.ReadAsArray()
rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0)); ink, _ = ink_mask(rgb, arr[3])
removed = gdal.Open(str(OUT / "explain_removed.tif")).ReadAsArray()
tgt = next(e for e in json.load(open(OUT / "legend.json")) if e["target"]); period, gap = tgt["dash"] + tgt["gap"], tgt["gap"]
_ds = ogr.Open(str(OUT / "explain_paths.gpkg")); llr = {f.GetFID(): f.GetField("llr_target") for f in _ds.GetLayerByName("paths")}
paths = {p["fid"]: p for p in read_paths(gt)}
for p in paths.values(): p["llr"] = llr[p["fid"]]
def end(p, side):
    xy = p["xy"]; e, f = (xy[0], xy[1]) if side == 0 else (xy[-1], xy[-2]); o = e - f; return (p, side, e, o / np.linalg.norm(o))
R = json.load(open(OUT / "breaks.json"))
out = Counter()
for r in R:
    if "partner" not in r or r["reason"] == "blank paper run": continue
    A = end(paths[r["fid"]], r["side"]); q = paths[r["partner"]]
    B = min((end(q, 0), end(q, 1)), key=lambda E: np.linalg.norm(E[2] - A[2]))
    s = br.join_score(A, B, ink, removed, period, gap, 2e-5)
    if s is None:
        oa, ob = A[3], B[3]; v = B[2] - A[2]; u = v / np.linalg.norm(v)
        k = f"geometry: oa.u={oa@u:.2f} ob.-u={-(ob@u):.2f} -oa.ob={-(oa@ob):.2f}"
        out["geometry"] += 1
    elif "refused" in s: k = s["refused"] + f" run={s.get('run',0)/period:.2f}p"; out["paper"] += 1
    else:
        k = f"t={s['t']:.1f} lat={s['lat']:.1f} turn={s['turn']:.1f} rem={s['rem']/period:.1f}p blank={s['blank']/period:.2f}p"
        out["scored>0" if s["t"] > 0 else "scored<=0"] += 1
    print(r["fid"], r["partner"], r["reason"][:18], r["gap_periods"], "|", k)
print(out)
print()
for f in [62, 65, 127, 102, 54, 43, 63, 72, 55, 125, 20, 56]:
    p = paths[f]; print(f, p["cls"], "n", p["n"], "sw", p["sw"], "dash", p["dash"], "llr", round(p["llr"], 1))
