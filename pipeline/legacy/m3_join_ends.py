"""M3: keep the baseline detector, then join chain ends across gaps ("good continuation").

Fragments are chains of 2+ clean dashes. Two fragment ends are a candidate join when each
points at the other, within six dash periods. What decides it is what the gap is made of:

  covered gap   ink of something else (text, river, grid, symbol). Cheap: the line may run
                under it.
  blank gap     empty paper. A pattern gap is blank too, so one missing dash's worth of blank
                (dash + gap) is allowed; anything longer means the line really stopped there.

Joins are taken cheapest first, each end used once. A group survives if its fragments add up
to a profile's minimum chain. Joins that cross a pool are kept but flagged: the topology may
be right (a line through a field) while the drawn geometry is not.

Writes out/m3.gpkg: m3_fragments and m3_joins.
"""

import json
import time

import numpy as np
from osgeo import ogr
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from common import DASHES, OUT, PROFILES, SUFFIX, anchored, candidates, chain, rhythm_ok, in_mask, load_ink, load_masks, load_text, write_lines

t = time.time()
ink, gt, proj = load_ink()
m = load_masks()
H, W = ink.shape
inky = ndi.binary_dilation(ink, iterations=2)
text = load_text(ink.shape)
covered = ndi.binary_dilation(m["hard"] | (m["rest"] & ~m["dash"]) | text, iterations=2)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def gap_profile(a, b):
    """Walk the straight segment a->b a pixel at a time: longest blank run, covered share, pool share."""
    n = max(int(np.linalg.norm(b - a)), 1)
    s = a + (b - a) * np.linspace(0, 1, n)[:, None]
    x = np.clip(np.round(s[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(s[:, 1]).astype(int), 0, H - 1)
    blank = ~inky[y, x]
    run = best = 0
    for v in blank:
        run = run + 1 if v else 0
        best = max(best, run)
    return best, float(covered[y, x].mean()), float(m["pools"][y, x].mean())


dashes = json.load(open(DASHES))
frag_feats, join_feats = [], []
gid = 0
for name, prof in PROFILES.items():
    ds = [dict(c=np.array(e["c"]), ax=np.array(e["ax"])) for e in candidates(dashes, prof)]
    frags = [c for c in anchored(chain(ds, prof, min_len=2), ds, text) if rhythm_ok([ds[k]["c"] for k in c], prof)]
    P = [np.array([ds[k]["c"] for k in c]) for c in frags]
    ends = []  # (fragment, point, outward direction)
    for f, p in enumerate(P):
        ends.append((f, p[0], unit(p[0] - p[1])))
        ends.append((f, p[-1], unit(p[-1] - p[-2])))
    tree = cKDTree([e[1] for e in ends])
    cand = []
    for i, j in tree.query_pairs(6 * prof["period"]):
        fi, pi, oi = ends[i]
        fj, pj, oj = ends[j]
        if fi == fj:
            continue
        u = unit(pj - pi)
        ca, cb = oi @ u, oj @ -u
        if ca < 0.9 or cb < 0.9:
            continue
        half = prof["dash"] / 2
        blank, cov, pool = gap_profile(pi + u * half, pj - u * half)
        if blank > prof["dash"] + prof["gap"]:
            continue
        dist = float(np.linalg.norm(pj - pi))
        cost = dist / prof["period"] + 4 * (2 - ca - cb) + blank / prof["period"]
        cand.append((cost, i, j, dict(dist_px=dist, blank_px=int(blank), covered=round(cov, 2),
                                      through_pool=int(pool > 0.2))))
    cand.sort(key=lambda x: x[0])
    used, parent = set(), list(range(len(P)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    joins = []
    for cost, i, j, attrs in cand:
        if i in used or j in used:
            continue
        fi, fj = ends[i][0], ends[j][0]
        if root(fi) == root(fj):  # would close a loop
            continue
        used |= {i, j}
        parent[root(fi)] = root(fj)
        joins.append((i, j, attrs))
    size = {}
    for f, c in enumerate(frags):
        size[root(f)] = size.get(root(f), 0) + len(c)
    keep = {r for r, n in size.items() if n >= prof["min_chain"]}
    gids = {r: gid + k for k, r in enumerate(sorted(keep))}
    gid += len(keep)
    for f, c in enumerate(frags):
        if root(f) in keep:
            frag_feats.append((list(P[f]), dict(group=gids[root(f)], profile=name, n_dash=len(c))))
    for i, j, attrs in joins:
        if root(ends[i][0]) in keep:
            join_feats.append(([ends[i][1], ends[j][1]], dict(attrs, group=gids[root(ends[i][0])], profile=name)))
    print(f"{name}: fragments {len(frags)}  candidate joins {len(cand)}  joins made {len(joins)}  "
          f"groups kept {len(keep)}  dashes in kept groups {sum(size[r] for r in keep)}")

write_lines(OUT / f"m3{SUFFIX}.gpkg", "m3_fragments", gt, frag_feats,
            dict(group=ogr.OFTInteger, profile=ogr.OFTString, n_dash=ogr.OFTInteger))
write_lines(OUT / f"m3{SUFFIX}.gpkg", "m3_joins", gt, join_feats,
            dict(group=ogr.OFTInteger, profile=ogr.OFTString, dist_px=ogr.OFTReal, blank_px=ogr.OFTInteger,
                 covered=ogr.OFTReal, through_pool=ogr.OFTInteger))
print(f"{time.time() - t:.0f}s")
