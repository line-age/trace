"""Explain what remains after removal: which shapes are best explained by the target legend entry.

  1. Pieces: connected bits of residual ink, each with stroke width (area / length), length,
     axis and colour (CIELAB, from the original scan).
  2. Links: two piece ends that face each other within a few legend periods. A link's evidence
     is its gap, sampled pixel by pixel: blank paper, removed ground ("can't tell") or ink.
  3. Global linking: a maximum-weight matching over all ends (each end joins at most one other),
     so every link is decided against every alternative at once, not greedily.
  4. Competition, per resulting path, in log-likelihood ratios against CHANCE (pieces lined up
     by accident, given how many piece ends lie nearby: clutter raises the bar automatically):
       target  the legend line: piece weight and colour from the swatch, dash length free per
               path around the swatch's (draftsmen vary it), blank gaps of regular length
       solid   a continuous line that removed features broke up: gaps must be removed or tiny
     A path is the target if its target score beats chance by MARGIN and beats solid.

Every parameter is either measured (legend swatch, sheet statistics) or a relative tolerance.
"""

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from common import bar_stats
from explain import rgb_to_lab

MARGIN = 8.0          # nats over chance
MAX_GROUP = 400       # ends per matching problem; larger groups are cut at their weakest links
LINK_PERIODS = 3.0    # how far a link may reach, in legend periods
MAX_REMOVED_PERIODS = 8.0
DENSITY_RADIUS = 300  # px, for the local density of piece ends


def pieces(ink, rgb, legend_dash):
    lab_img, bars = bar_stats(ink)
    bars = [b for b in bars if b["length"] >= max(6, 0.25 * legend_dash)]
    idx = np.nonzero(ink)
    labs = rgb_to_lab(rgb[idx])
    ids = lab_img[idx]
    out = []
    want = sorted({b["id"] for b in bars})
    mean = [ndi.mean(labs[:, k], ids, want) if want else [] for k in range(3)]
    colour = {i: np.array([mean[0][n], mean[1][n], mean[2][n]]) for n, i in enumerate(want)}
    for b in bars:
        sw = b["npx"] / max(b["length"], 1)
        if b["length"] < 2 * sw:
            continue  # not elongated: a dot or blob, not a line piece
        out.append(dict(id=b["id"], c=np.asarray(b["c"]), ax=np.asarray(b["ax"]) / np.linalg.norm(b["ax"]), L=float(b["length"]),
                        sw=float(sw), lab=colour[b["id"]]))
    return out, lab_img


def gap_evidence(a, b, orig_ink, removed, lab=None, own=(), axes=None):
    n = max(int(np.linalg.norm(b - a)), 1)
    s = a + (b - a) * np.linspace(0, 1, n + 1)[:, None]
    x = np.clip(np.round(s[:, 0]).astype(int), 0, orig_ink.shape[1] - 1)
    y = np.clip(np.round(s[:, 1]).astype(int), 0, orig_ink.shape[0] - 1)
    if lab is not None:
        # a link may not pass through another piece: that piece is the next link. Without this,
        # links skipping every other dash score well (the skipped dash reads as ink in the gap)
        # and each dashed line is traced twice, as two interleaved chains.
        hit = np.unique(lab[y, x])
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        for h in hit:
            if h > 0 and h not in own and abs(axes[h] @ u) > np.cos(np.radians(25)):
                return None  # (a piece crossing the gap, like a grid sliver, does not block it)
    rem = removed[y, x]
    blank = ~orig_ink[y, x] & ~rem
    run = best = 0
    for v in blank:
        run = run + 1 if v else 0
        best = max(best, run)
    return float(n), float(blank.sum()), float(rem.sum()), float(best)


def log_n(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))


def bar_llr(p, target, bg):
    """Piece under the target style vs under the sheet's own piece distribution."""
    lt = log_n(p["sw"], target["width"], 0.25 * target["width"]) + log_n(np.linalg.norm(p["lab"] - target["lab"]), 0, 12) + np.log(2)
    lb = bg(p)
    return lt - lb


def background_model(ps):
    """The sheet's own distribution of piece width and colour distance (smoothed histograms)."""
    sw = np.array([p["sw"] for p in ps])
    hs, es = np.histogram(sw, bins=np.arange(0, 30, 0.5), density=True)
    hs = ndi.gaussian_filter1d(hs, 2) + 1e-4

    def f(p, _hs=hs, _es=es):
        k = int(np.clip(np.searchsorted(_es, p["sw"]) - 1, 0, len(_hs) - 1))
        return np.log(_hs[k]) + np.log(1 / 60)  # colour: flat over a broad range (delta E 0-60)
    return f


def link_llr(pi, pj, ei, ej, orig_ink, removed, target, lam, lab=None, axes=None):
    """Link between end ei of piece pi and end ej of pj: target-line likelihood vs chance."""
    a, oa = ei
    b, ob = ej
    v = b - a
    d = np.linalg.norm(v)
    if d < 1:
        return None
    u = v / d
    if oa @ u < 0.9 or ob @ -u < 0.9 or abs(pi["ax"] @ pj["ax"]) < np.cos(np.radians(25)):
        return None
    period = target["dash"] + target["gap"]
    ev = gap_evidence(a, b, orig_ink, removed, lab, (pi.get("id", 0), pj.get("id", 0)), axes)
    if ev is None:
        return None
    n, blank, rem, run = ev
    if run > 1.5 * period or rem > MAX_REMOVED_PERIODS * period:
        return None  # a long stretch of paper: the line stopped
    turn = np.degrees(np.arccos(min(1.0, abs(pi["ax"] @ pj["ax"]))))
    lat = np.linalg.norm(v - (v @ oa) * oa)
    sd_lat = max(pi["sw"], 2.0)
    reach = LINK_PERIODS * period
    # Every term is target density / chance density over the SAME variable and range, so the
    # sum is a dimensionless log-odds (chance: uniform over what the search accepts).
    t_turn = log_n(turn, 0, 8) + np.log(2) + np.log(25)                   # half-normal vs U[0, 25 deg]
    t_lat = log_n(lat, 0, sd_lat) + np.log(2) + np.log(max(d, 1) * np.tan(np.radians(25)))  # vs U[0, wedge]
    t_len = log_n(np.log(pj["L"] / pi["L"]), 0, 0.35) + np.log(3.0)       # vs U over log-ratio [-1.5, 1.5]
    t_sw = log_n(np.log(pj["sw"] / pi["sw"]), 0, 0.25) + np.log(3.0)
    if blank > 0:  # blank part near the legend gap (log-normal, wide: draftsmen vary it) vs log-uniform
        t_gap = log_n(np.log(blank + 1), np.log(target["gap"] + 1), 0.6) + np.log(np.log(1.5 * period + 1))
    else:
        t_gap = 0.0
    t_rem = -0.3 * rem / period                                           # can't tell: a small cost per period
    # existence: the target expects a next piece (p 0.8); chance puts one in the wedge with
    # probability 1 - exp(-lambda * area). Clutter (high lambda) raises the bar automatically.
    wedge = reach ** 2 * np.tan(np.radians(25))
    t_exist = np.log(0.8) - np.log(1 - np.exp(-max(lam * wedge, 1e-6)))
    t = t_turn + t_lat + t_len + t_sw + t_gap + t_rem + t_exist
    # solid: a continuous line broken by removed features: the gap must be removed ground or tiny
    if blank <= 2:
        s = log_n(turn, 0, 15) + np.log(2) + np.log(25) + t_lat + t_sw - 0.1 * rem / period + t_exist
    else:
        s = -np.inf
    return dict(t=t, s=s, d=d, blank=blank, rem=rem)


def classify(ink, rgb, orig_ink, removed, target, log=print):
    """Returns paths: list of dict(points, n, llr_target, llr_solid, cls)."""
    ps, lab_all = pieces(ink, rgb, target["dash"])
    kept = np.zeros(lab_all.max() + 1, np.int32)
    kept[[p["id"] for p in ps]] = [p["id"] for p in ps]
    lab = kept[lab_all]  # label image of the kept pieces only
    axes = {p["id"]: p["ax"] for p in ps}
    log(f"{len(ps)} line pieces in the residual")
    bg = background_model(ps)
    ends = []  # (piece, side, point, outward dir)
    for i, p in enumerate(ps):
        h = p["ax"] * p["L"] / 2
        ends.append((i, 0, p["c"] - h, -p["ax"]))
        ends.append((i, 1, p["c"] + h, p["ax"]))
    P = np.array([e[2] for e in ends])
    tree = cKDTree(P)
    dens = np.array([len(x) for x in tree.query_ball_point(P, DENSITY_RADIUS)]) / (np.pi * DENSITY_RADIUS ** 2)
    period = target["dash"] + target["gap"]
    G = nx.Graph()
    info = {}
    for i, j in tree.query_pairs(LINK_PERIODS * period):
        ei, ej = ends[i], ends[j]
        if ei[0] == ej[0]:
            continue
        r = link_llr(ps[ei[0]], ps[ej[0]], (ei[2], ei[3]), (ej[2], ej[3]), orig_ink, removed, target, dens[i], lab, axes)
        if r is None:
            continue
        w = max(r["t"], r["s"])
        if w > 0:
            G.add_edge(i, j, weight=w)
            info[frozenset((i, j))] = r
    sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    log(f"{G.number_of_edges()} possible links among {len(ends)} piece ends; largest groups {sizes[:5]}")
    # Blossom matching scales badly with group size. Lines are chains, not tangles, so a big
    # group is cut at its weakest links until every piece is small enough (logged).
    match, cut = {}, 0
    groups = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    while groups:
        sub = groups.pop()
        if sub.number_of_nodes() > MAX_GROUP:
            weakest = sorted(sub.edges(data="weight"), key=lambda e: e[2])[:max(1, sub.number_of_edges() // 10)]
            sub.remove_edges_from([(a, b) for a, b, _ in weakest])
            cut += len(weakest)
            groups += [sub.subgraph(c).copy() for c in nx.connected_components(sub)]
            continue
        for a, b in nx.max_weight_matching(sub):
            match[a], match[b] = b, a
    log(f"matched {len(match) // 2} links (cut {cut} weak links to keep groups <= {MAX_GROUP} ends)")
    # walk paths: piece -> its other end -> matched end -> next piece
    seen, paths = set(), []
    for start in range(len(ps)):
        if start in seen:
            continue
        e0, e1 = 2 * start, 2 * start + 1
        if e0 in match and e1 in match:
            continue  # interior piece: reached from a path end (or part of a loop, picked up below)
        chain, side = [start], (1 if e0 not in match else 0)
        seen.add(start)
        out_end = 2 * start + side
        links = []
        while out_end in match:
            nxt = match[out_end]
            links.append(info[frozenset((out_end, nxt))])
            k = nxt // 2
            if k in seen:
                break
            seen.add(k)
            chain.append(k)
            out_end = 2 * k + (1 - nxt % 2)
        if len(chain) >= 2:
            paths.append((chain, links))
    out = []
    for chain, links in paths:
        bars = [ps[k] for k in chain]
        lt = sum(bar_llr(b, target, bg) for b in bars) + sum(l["t"] for l in links)
        ls = sum(l["s"] for l in links)
        dash_med = float(np.median([b["L"] for b in bars]))
        # dash length free per path, but drawn from around the swatch's (log-normal, wide)
        lt += log_n(np.log(dash_med / target["dash"]), 0, 0.7)
        # stroke weight is a legend channel and, unlike dash length, the draftsman's pen does not
        # vary it: the path's median weight must match the swatch's (log-normal, narrow) against
        # a uniform log-ratio over [-1.5, 1.5]. Per-piece width alone was outvoted by alignment:
        # a grid line's remnants line up perfectly.
        sw_med = float(np.median([b["sw"] for b in bars]))
        lt += log_n(np.log(sw_med / target["width"]), 0, 0.2) + np.log(3.0)
        cls = "target" if (lt > MARGIN and lt > ls) else ("solid" if ls > MARGIN else "chance")
        pts = []
        for k in chain:
            b = ps[k]
            h = b["ax"] * b["L"] / 2
            a0, a1 = b["c"] - h, b["c"] + h
            if pts and np.linalg.norm(pts[-1] - a1) < np.linalg.norm(pts[-1] - a0):
                a0, a1 = a1, a0
            pts += [a0, a1]
        out.append(dict(points=pts, n=len(chain), llr_target=float(lt), llr_solid=float(ls) if np.isfinite(ls) else -999.0,
                        cls=cls, dash=round(dash_med, 1), sw=round(float(np.median([b["sw"] for b in bars])), 1),
                        removed_gap=round(float(sum(l["rem"] for l in links)), 1)))
    return out
