"""Join each pipeline's pieces into one line, find junctions, and say why every remaining end
is open. The rules and scores in full: docs/algorithm.md, step 14.

The piece linker (classify_residual) reaches a fixed 3 legend periods. diagnose_breaks showed
most breaks are longer than that but lie across removed ground (text, symbols). Here:

  partners  target paths only: a path that cannot pass as the target alone is never glued
            onto one (that pulled in town lines). The legend's own swatches are dropped.
  reach     no fixed distance: a join is refused by evidence, a run of blank paper longer
            than the target's own gap allows (the search radius is a compute bound)
  corners   two ends nearly touching are one line turning, at any angle; two ends side by
            side pointing the same way are two lines (no hairpins)
  others    ink within a stroke width of another target line is already explained: it never
            supports a join running beside it, and a join may not run along that line
  decision  maximum-weight matching over path ends, as for pieces
  junction  an end that stops on another line's side is a T: recorded, not joined
  ends      every open end gets a cause from what lies ahead

Writes out/explain_joined.gpkg: layers "corrected" (the answer: one simplified line per
pipeline), "lines" (the same, unsimplified, with scores), "joins" (each bridged gap), "ends"
(open ends with cause), "junctions".
"""

import json
from collections import Counter

import networkx as nx
import numpy as np
from osgeo import gdal, ogr
from scipy.spatial import cKDTree

import classify_residual as cr
from common import OUT, write_lines, write_points
from diagnose_breaks import longest_run, read_paths, sample
from explain import SHEET, ink_mask

SEARCH_PERIODS = 20.0   # compute bound only; joins that would need more are counted
CONE_DEG = 25.0
DENSITY_RADIUS = cr.DENSITY_RADIUS
STAGES = {1: "text", 2: "boundary", 3: "river", 4: "legend", 5: "grid"}


def sample_band(a, b, half, *masks, others=None, own=(), near=0.0):
    """Like sample(), but across a band +-half px wide: a step is ink (or removed) if any pixel
    across the band is. A dash a few px off the chord between two ends is still the line.
    Pixels within `near` of another target line (not one of `own`) count as neither: that ink
    is already explained, and must not support a join running beside it."""
    v = b - a
    nrm = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-9)
    n = max(int(np.linalg.norm(v)), 1) + 1
    out = [np.zeros(n, bool) for _ in masks]
    for k in np.arange(-half, half + 1):
        got = sample(a + k * nrm, b + k * nrm, *masks)
        if others is not None:
            pts = (a + k * nrm) + v * np.linspace(0, 1, n)[:, None]
            tree, _, owner = others
            dist, j = tree.query(pts, distance_upper_bound=near)
            hit = np.isfinite(dist)
            hit[hit] = ~np.isin(owner[j[hit]], own)
            got = [g & ~hit for g in got]
        for acc, g in zip(out, got):
            acc |= g
    return out


def simplify(pts, tol):
    """Douglas-Peucker: the joined chain of dash endpoints as one clean polyline."""
    pts = np.asarray(pts)
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n if n > 0 else np.linalg.norm(pts - a, axis=1)
    k = int(np.argmax(d))
    if d[k] <= tol:
        return np.array([a, b])
    return np.vstack([simplify(pts[:k + 1], tol)[:-1], simplify(pts[k:], tol)])


def log_n(x, mu, sd):
    return cr.log_n(x, mu, sd)


def densify(xy, step=3.0):
    pts, dirs = [xy[0]], []
    for a, b in zip(xy[:-1], xy[1:]):
        v = b - a
        n = max(int(np.linalg.norm(v) / step), 1)
        pts += list(a + v * np.linspace(0, 1, n + 1)[1:, None])
        dirs += [v / max(np.linalg.norm(v), 1e-9)] * n
    return np.array(pts), np.array([dirs[0]] + dirs)


def join_score(A, B, ink, removed, period, gap, lam, others=None):
    """A, B: (path, side, point, outward dir). Log-odds that B continues A, vs chance."""
    pa, _, a, oa = A
    pb, _, b, ob = B
    v = b - a
    d = float(np.linalg.norm(v))
    sw = max(pa["sw"], pb["sw"], 2.0)
    close = d < 2 * sw  # ends nearly touch: the direction between them is noise
    corner = d <= period  # two ends this close are one line turning, at any angle
    u = oa if close else v / d
    cos = np.cos(np.radians(CONE_DEG))
    if not corner and (oa @ u < cos or ob @ -u < cos):
        return None
    if corner and oa @ ob > 0.5:
        return None  # both ends point the same way: two lines side by side, not a turn (a hairpin)
    if others is not None and not close:
        # a join may not run along another target line: that line already explains the ink
        # in the gap, and the join would trace it a second time (a skip, one level up)
        tree, dirs, owner = others
        n_s = max(int(d / 3), 2)
        s_pts = a + v * np.linspace(0, 1, n_s)[:, None]
        dist, k = tree.query(s_pts, distance_upper_bound=sw)  # on top of it; real parallels sit ~2 widths apart
        ok = np.isfinite(dist)
        along = ok.copy()
        along[ok] = (np.abs(dirs[k[ok]] @ u) > cos) & ~np.isin(owner[k[ok]], (pa["fid"], pb["fid"]))
        if along.mean() > 0.25:
            return dict(refused="runs along another line", d=d, run=0.0)
    o, r = sample_band(a, b, int(np.ceil(sw)), ink, removed > 0, others=others, own=(pa["fid"], pb["fid"]), near=sw)
    blank_mask = ~o & ~r
    run = longest_run(blank_mask)
    if run > 1.5 * period:
        return dict(refused="blank paper run", d=d, run=float(run))
    n, blank, rem = float(len(o)), float(blank_mask.sum()), float(r.sum())
    # bend: how far each end points away from the other. Two ends that both point at each
    # other may differ in direction (the line bends in the gap); what counts is how far each
    # misses the other, not the angle between them.
    ang = lambda x, y: float(np.degrees(np.arccos(np.clip(x @ y, -1.0, 1.0))))
    turn = ang(oa, -ob) if close else max(ang(oa, u), ang(ob, -u))
    lat = float(np.linalg.norm(v - (v @ oa) * oa))
    if not close:
        lat = min(lat, float(np.linalg.norm(-v - (-v @ ob) * ob)))
    if corner:
        # no direction evidence either way; chance is judged by how crowded the spot is (below)
        t_turn = t_lat = 0.0
    else:
        t_turn = log_n(turn, 0, 8) + np.log(2) + np.log(CONE_DEG)
        t_lat = log_n(lat, 0, sw) + np.log(2) + np.log(max(d, 1) * np.tan(np.radians(CONE_DEG)))
    t_sw = log_n(np.log(pb["sw"] / pa["sw"]), 0, 0.25) + np.log(3.0)
    t_len = log_n(np.log(pb["dash"] / pa["dash"]), 0, 0.35) + np.log(3.0)
    # paper beyond what the dash pattern itself leaves blank, on the part not removed
    expected = (gap / period) * (n - rem)
    t_paper = -max(0.0, blank - expected) / period
    t_rem = -0.3 * rem / period
    wedge = np.pi * max(d, sw) ** 2 if corner else max(d, period) ** 2 * np.tan(np.radians(CONE_DEG))
    t_exist = np.log(0.8) - np.log(1 - np.exp(-max(lam * wedge, 1e-6)))
    t = t_turn + t_lat + t_sw + t_len + t_paper + t_rem + t_exist
    return dict(t=float(t), d=d, blank=blank, rem=rem, run=float(run), lat=lat, turn=turn)


def main():
    ds = gdal.Open(SHEET)
    gt = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0))
    ink, _ = ink_mask(rgb, arr[3] if arr.shape[0] > 3 else None)
    removed = gdal.Open(str(OUT / "explain_removed.tif")).ReadAsArray()
    residual = ink & (removed == 0)
    tgt = next(e for e in json.load(open(OUT / "legend.json")) if e["target"])
    period, gap = tgt["dash"] + tgt["gap"], tgt["gap"]

    ogr_ds = ogr.Open(str(OUT / "explain_paths.gpkg"))
    llr = {f.GetFID(): f.GetField("llr_target") for f in ogr_ds.GetLayerByName("paths")}
    # target paths only: a path that cannot pass as the target on its own is not added to one
    # (absorbing them pulled in grid and town-line remnants). Its ink still counts in a gap.
    paths = [p for p in read_paths(gt) if p["cls"] == "target"]
    # the legend's own swatches are not map features (legend bboxes are [row0, col0, row1, col1])
    legend = json.load(open(OUT / "legend.json"))
    bb = np.array([e["bbox"] for e in legend], float)
    m = 2 * period
    r0, c0, r1, c1 = bb[:, 0].min() - m, bb[:, 1].min() - m, bb[:, 2].max() + m, bb[:, 3].max() + m
    inside = lambda xy: np.all((xy[:, 0] >= c0) & (xy[:, 0] <= c1) & (xy[:, 1] >= r0) & (xy[:, 1] <= r1))
    dropped = [p["fid"] for p in paths if inside(p["xy"])]
    paths = [p for p in paths if not inside(p["xy"])]
    print(f"legend swatches dropped: paths {dropped}")
    for p in paths:
        p["llr"] = llr[p["fid"]]
    ends = []
    for p in paths:
        xy = p["xy"]
        for side, (e, f) in enumerate(((xy[0], xy[1]), (xy[-1], xy[-2]))):
            o = e - f
            ends.append((p, side, e, o / np.linalg.norm(o)))
    P = np.array([e[2] for e in ends])
    tree = cKDTree(P)
    lam = np.array([len(x) for x in tree.query_ball_point(P, DENSITY_RADIUS)]) / (np.pi * DENSITY_RADIUS ** 2)

    dense = [densify(p["xy"]) for p in paths]
    others = (cKDTree(np.vstack([d[0] for d in dense])), np.vstack([d[1] for d in dense]),
              np.concatenate([[p["fid"]] * len(d[0]) for p, d in zip(paths, dense)]))
    G, info, refused = nx.Graph(), {}, Counter()
    for i, j in tree.query_pairs(SEARCH_PERIODS * period):
        A, B = ends[i], ends[j]
        if A[0] is B[0] or "target" not in (A[0]["cls"], B[0]["cls"]):
            continue
        r = join_score(A, B, ink, removed, period, gap, lam[i], others)
        if r is None:
            continue
        if "refused" in r:
            refused[r["refused"]] += 1
            continue
        # a chance path may join only if it does not lower the target's score
        w = r["t"] + sum(min(0.0, X[0]["llr"]) for X in (A, B) if X[0]["cls"] == "chance")
        if w > 0:
            G.add_edge(i, j, weight=w)
            info[frozenset((i, j))] = r
    match = {}
    for a, b in nx.max_weight_matching(G):
        match[a], match[b] = b, a
    print(f"{len(paths)} target paths; {G.number_of_edges()} possible joins; {len(match) // 2} made; refused: {dict(refused)}")

    # walk chains of paths: path -> its other end -> matched end -> next path
    seen, lines, joins = set(), [], []
    order = sorted(range(len(paths)), key=lambda k: paths[k]["cls"] != "target")
    loops = 0
    for sweep, start in [(0, k) for k in order] + [(1, k) for k in order]:
        if start in seen:
            continue
        e0, e1 = 2 * start, 2 * start + 1
        if e0 in match and e1 in match:
            if sweep == 0:
                continue  # interior: reached from a chain end
            loops += 1  # second sweep: whatever is left sits in a loop; open it here
            out_end = e1
        else:
            out_end = e1 if e0 not in match else e0
        chain, pts = [start], list(paths[start]["xy"] if out_end == e1 else paths[start]["xy"][::-1])
        seen.add(start)
        score = paths[start]["llr"]
        while out_end in match:
            nxt = match[out_end]
            k = nxt // 2
            if k in seen:
                break
            r = info[frozenset((out_end, nxt))]
            joins.append(([pts[-1], ends[nxt][2]], dict(a=paths[chain[-1]]["fid"], b=paths[k]["fid"], gap_p=round(r["d"] / period, 2),
                                                          removed=round(r["rem"] / max(r["d"], 1), 2), score=round(r["t"], 1))))
            score += r["t"] + paths[k]["llr"]
            seen.add(k)
            chain.append(k)
            xy = paths[k]["xy"]
            pts += list(xy if nxt % 2 == 0 else xy[::-1])
            out_end = 2 * k + (1 - nxt % 2)
        if not any(paths[k]["cls"] == "target" for k in chain):
            continue
        cls = "target" if score > cr.MARGIN else "chance"
        if cls != "target":
            continue
        lines.append(dict(pts=np.array(pts), members=[paths[k]["fid"] for k in chain], score=score,
                          sw=float(np.median([paths[k]["sw"] for k in chain]))))

    # junctions and open-end causes, on the joined lines
    segs = []
    for li, L in enumerate(lines):
        for a, b in zip(L["pts"][:-1], L["pts"][1:]):
            segs.append((li, a, b))
    def seg_dist(p, a, b):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        return float(np.linalg.norm(p - (a + t * ab)))

    def crosses(p, q, a, b):
        c = lambda x, y, z: (y[0] - x[0]) * (z[1] - x[1]) - (y[1] - x[1]) * (z[0] - x[0])
        return c(p, q, a) * c(p, q, b) < 0 and c(a, b, p) * c(a, b, q) < 0
    H, W = ink.shape
    end_feats, junctions, causes = [], [], Counter()
    for li, L in enumerate(lines):
        for e, f in ((L["pts"][0], L["pts"][1]), (L["pts"][-1], L["pts"][-2])):
            o = (e - f) / max(np.linalg.norm(e - f), 1e-9)
            probe = e + o * 1.5 * period
            hit = None
            for lj, a, b in segs:
                # a T: the end stops on another line, or would cross it within 1.5 periods.
                # (Merely passing near a parallel line is not a junction.)
                ab = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                if lj != li and abs(o @ ab) < np.cos(np.radians(CONE_DEG)) and (
                        seg_dist(e, a, b) < 1.5 * L["sw"] or crosses(e, probe, a, b)):
                    hit = lj
                    break
            if hit is not None:
                junctions.append((e, dict(line=li + 1, meets=hit + 1)))
                cause = "junction"
            elif min(e[0], e[1], W - 1 - e[0], H - 1 - e[1]) < 2 * period:
                cause = "sheet edge"
            else:
                ahead = e + o * 5 * period
                o_, r_, res_ = sample(e + o * 3, ahead, ink, removed > 0, residual)
                by = Counter(int(v) for v in sample(e + o * 3, ahead, removed)[0] if v)
                if r_.mean() > 0.5:
                    cause = "into removed " + STAGES[max(by, key=by.get)]
                elif res_.mean() > 0.3:
                    cause = "unlinked ink ahead"
                elif longest_run(~o_ & ~r_) >= 3 * period:
                    cause = "paper ahead"
                else:
                    cause = "other"
            causes[cause] += 1
            end_feats.append((e, dict(line=li + 1, cause=cause)))

    out = OUT / "explain_joined.gpkg"
    if out.exists():
        out.unlink()
    write_lines(out, "lines", gt, [(L["pts"], dict(line=k + 1, n_paths=len(L["members"]), members=",".join(map(str, L["members"])),
                                                   score=round(L["score"], 1))) for k, L in enumerate(lines)],
                dict(line=ogr.OFTInteger, n_paths=ogr.OFTInteger, members=ogr.OFTString, score=ogr.OFTReal))
    # the corrected map: one clean line per pipeline (tolerance: one stroke width)
    write_lines(out, "corrected", gt, [(simplify(L["pts"], L["sw"]), dict(line=k + 1, n_paths=len(L["members"]),
                                                                            members=",".join(map(str, L["members"]))))
                                       for k, L in enumerate(lines)],
                dict(line=ogr.OFTInteger, n_paths=ogr.OFTInteger, members=ogr.OFTString))
    write_lines(out, "joins", gt, joins, dict(a=ogr.OFTInteger, b=ogr.OFTInteger, gap_p=ogr.OFTReal, removed=ogr.OFTReal, score=ogr.OFTReal))
    write_points(out, "ends", gt, end_feats, dict(line=ogr.OFTInteger, cause=ogr.OFTString))
    write_points(out, "junctions", gt, junctions, dict(line=ogr.OFTInteger, meets=ogr.OFTInteger))
    n_t = sum(p["cls"] == "target" for p in paths)
    cls_of = {p["fid"]: p["cls"] for p in paths}
    absorbed = sum(cls_of[m] == "chance" for L in lines for m in L["members"])
    covered = {m for L in lines for m in L["members"]}
    lost = [p["fid"] for p in paths if p["cls"] == "target" and p["fid"] not in covered]
    print(f"loops opened: {loops}; target paths in no line: {lost}")
    print(f"{n_t} target paths -> {len(lines)} lines ({len(joins)} joins, {absorbed} chance paths absorbed)")
    print(f"open ends {2 * len(lines)}: {dict(causes.most_common())}")


if __name__ == "__main__":
    main()
