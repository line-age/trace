"""Why do target paths break? For every end of every target path, find the path end that most
plausibly continues it (any class), measure the gap between them and say which linker rule
(classify_residual.link_llr) kept them apart. Ends with no continuation are measured too: what
lies ahead of them (removed ground, residual ink, paper).

Writes out/breaks.json and out/breaks.gpkg (layer "gaps": one line per end-to-partner gap,
attribute `reason`) and prints a tally.
"""

import json
from collections import Counter

import numpy as np
from osgeo import gdal, ogr

import classify_residual as cr
from common import OUT, write_lines
from explain import SHEET, ink_mask

REACH_PERIODS = 15.0   # look this far for a continuation (5x the linker's own reach)
CONE_DEG = 25.0        # same cone as the linker


def read_paths(gt):
    ds = ogr.Open(str(OUT / "explain_paths.gpkg"))
    lyr = ds.GetLayerByName("paths")
    out = []
    for f in lyr:
        g = f.GetGeometryRef()
        lonlat = np.array([g.GetPoint_2D(k) for k in range(g.GetPointCount())])
        xy = np.c_[(lonlat[:, 0] - gt[0]) / gt[1], (lonlat[:, 1] - gt[3]) / gt[5]]
        out.append(dict(fid=f.GetFID(), cls=f.GetField("cls"), sw=f.GetField("sw"), dash=f.GetField("dash"),
                        n=f.GetField("n"), xy=xy))
    return out


def ends_of(p):
    xy = p["xy"]
    o0 = xy[0] - xy[1]
    o1 = xy[-1] - xy[-2]
    return [(p, 0, xy[0], o0 / np.linalg.norm(o0)), (p, 1, xy[-1], o1 / np.linalg.norm(o1))]


def sample(a, b, *masks):
    n = max(int(np.linalg.norm(b - a)), 1)
    s = a + (b - a) * np.linspace(0, 1, n + 1)[:, None]
    x = np.clip(np.round(s[:, 0]).astype(int), 0, masks[0].shape[1] - 1)
    y = np.clip(np.round(s[:, 1]).astype(int), 0, masks[0].shape[0] - 1)
    return [m[y, x] for m in masks]


def longest_run(v):
    run = best = 0
    for t in v:
        run = run + 1 if t else 0
        best = max(best, run)
    return best


def main():
    ds = gdal.Open(SHEET)
    gt = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0))
    ink, _ = ink_mask(rgb, arr[3] if arr.shape[0] > 3 else None)
    removed = gdal.Open(str(OUT / "explain_removed.tif")).ReadAsArray()
    residual = ink & (removed == 0)
    tgt = next(e for e in json.load(open(OUT / "legend.json")) if e["target"])
    period = tgt["dash"] + tgt["gap"]

    paths = read_paths(gt)
    all_ends = [e for p in paths for e in ends_of(p)]
    P = np.array([e[2] for e in all_ends])
    rows, feats = [], []
    for e in all_ends:
        p, side, a, oa = e
        if p["cls"] != "target":
            continue
        v = P - a
        d = np.linalg.norm(v, axis=1)
        along = v @ oa
        lat = np.abs(v[:, 0] * oa[1] - v[:, 1] * oa[0])
        ok = (d > 1) & (d < REACH_PERIODS * period) & (along > 0) & (lat < np.tan(np.radians(CONE_DEG)) * along)
        best, score = None, np.inf
        for k in np.nonzero(ok)[0]:
            q, _, b, ob = all_ends[k]
            if q["fid"] == p["fid"] or ob @ (-(b - a) / d[k]) < np.cos(np.radians(CONE_DEG)):
                continue
            s = d[k] + 4 * lat[k]
            if s < score:
                best, score = k, s
        row = dict(fid=p["fid"], side=side, x=round(float(a[0])), y=round(float(a[1])), n=p["n"])
        if best is None:
            # nothing continues it: what lies ahead for 5 periods?
            b = a + oa * 5 * period
            o, r, res = sample(a + oa * 3, b, ink, removed > 0, residual)
            row.update(reason="no continuation", ahead_removed=round(float(r.mean()), 2),
                       ahead_residual=round(float(res.mean()), 2), ahead_blank_run=round(longest_run(~o & ~r) / period, 2),
                       ahead_by=dict(Counter(int(v) for v in sample(a + oa * 3, b, removed)[0] if v)))
            rows.append(row)
            continue
        q, _, b, ob = all_ends[best]
        o, r, res = sample(a, b, ink, removed > 0, residual)
        blank_run = longest_run(~o & ~r)
        rem = float(r.sum())
        turn = float(np.degrees(np.arccos(min(1.0, abs(oa @ ob)))))
        if q["cls"] != "target":
            reason = f"partner is {q['cls']}"
        elif d[best] > cr.LINK_PERIODS * period:
            reason = "beyond link reach"
        elif blank_run > 1.5 * period:
            reason = "blank paper run"
        elif rem > cr.MAX_REMOVED_PERIODS * period:
            reason = "too much removed ground"
        else:
            reason = "within rules (lost in scoring or matching)"
        by = Counter(int(v) for v in sample(a, b, removed)[0] if v)
        row.update(reason=reason, partner=q["fid"], partner_cls=q["cls"], gap_periods=round(float(d[best]) / period, 2),
                   lat_px=round(float(lat[best]), 1), turn_deg=round(turn, 1), blank_run_periods=round(blank_run / period, 2),
                   removed_frac=round(rem / max(d[best], 1), 2), residual_frac=round(float(res.mean()), 2),
                   removed_by={ {1: "text", 2: "boundary", 3: "river", 4: "legend", 5: "grid"}[k]: v for k, v in by.items()})
        rows.append(row)
        feats.append(([a, b], dict(fid_a=p["fid"], fid_b=q["fid"], reason=reason, gap_p=row["gap_periods"])))
    json.dump(rows, open(OUT / "breaks.json", "w"), indent=1)
    write_lines(OUT / "breaks.gpkg", "gaps", gt, feats,
                dict(fid_a=ogr.OFTInteger, fid_b=ogr.OFTInteger, reason=ogr.OFTString, gap_p=ogr.OFTReal))
    print(f"{len(rows)} target path ends")
    for k, v in Counter(r["reason"] for r in rows).most_common():
        print(f"  {v:4d}  {k}")
    for reason in sorted({r["reason"] for r in rows}):
        sub = [r for r in rows if r["reason"] == reason]
        print(f"\n== {reason} ({len(sub)})")
        for r in sub[:12]:
            print("  ", {k: v for k, v in r.items() if k not in ("reason",)})


if __name__ == "__main__":
    main()
