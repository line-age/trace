"""M1: explain away the non-pipeline layers, then accept dashes that those layers clipped.

Two additions over the baseline, both asking "what would this blob be if the thing on top of
it were not there?":

  partial  a thin stroke shorter than a full dash that runs INTO an occluder (grid, pool,
           thick stroke). It may be a dash whose other end is underneath. Too square to have a
           direction? Then its own axis is not trusted when chaining.
  unfused  inside an "other ink" blob (text, symbols), any straight stroke longer than a
           letter stroke. Letters on this sheet are ~15-20 px tall, a trunk dash is 33, so an
           opening with a 23 px line stamp keeps dashes and drops letters. Trunk profile only:
           a short dash is letter-sized and cannot be separated this way.

Writes out/m1.gpkg: m1_chains (lines) and m1_dashes (points, with kind).
"""

import time

import numpy as np
from osgeo import ogr
from scipy import ndimage as ndi

from common import (LEG, OUT, PROFILES, SUFFIX, anchored, bar_stats, chain, classify, in_mask, is_dash, line_se, load_ink,
                    load_masks, load_text, write_lines, write_points)

t = time.time()
ink, gt, proj = load_ink()
m = load_masks()
text = load_text(ink.shape)
# a label is an occluder like any other: a dash running into one may be clipped by it
near_hard = ndi.binary_dilation(m["hard"] | text, iterations=4)
lab, bars = bar_stats(m["rest"])

per = {name: [] for name in PROFILES}
other = []
for b in bars:
    b["reliable"] = b["length"] / max(b["width"], 1) >= 1.6
    hit = False
    for name, prof in PROFILES.items():
        if is_dash(b, prof):
            per[name].append(dict(b, kind="clean"))
            hit = True
        elif (0.35 * prof["dash"] <= b["length"] <= prof["dash"] + prof["slack_hi"]
              and b["width"] <= LEG["thick"] * 1.7
              and near_hard[b["sl"]][lab[b["sl"]] == b["id"]].any()):
            per[name].append(dict(b, kind="partial"))
            hit = True
    if not hit and b["npx"] >= 40:
        other.append(b)

trunk = PROFILES["trunk"]
angles = np.radians(np.arange(0, 180, 15))
stamps = [line_se(23, a) for a in angles]
n_unfused = 0
for b in other:
    if in_mask(text, b["c"]):
        continue  # a label: its strokes are letters, not dashes fused to something
    sl = b["sl"]
    pad = 2
    y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
    crop = lab[y0:sl[0].stop + pad, x0:sl[1].stop + pad] == b["id"]
    kept = np.zeros_like(crop)
    for k in stamps:
        kept |= ndi.binary_opening(crop, k)
    if not kept.any():
        continue
    _, pieces = bar_stats(kept)
    for p in pieces:
        if is_dash(p, trunk) and p["length"] / max(p["width"], 1) >= 2.5:
            p["c"] = p["c"] + [x0, y0]
            p["reliable"] = True
            per["trunk"].append(dict(p, kind="unfused"))
            n_unfused += 1

line_feats, point_feats = [], []
for name, prof in PROFILES.items():
    ds = per[name]
    chains = anchored(chain(ds, prof), ds, text)
    kinds = [[ds[k]["kind"] for k in c] for c in chains]
    line_feats += [([ds[k]["c"] for k in c], dict(profile=name, n_dash=len(c), n_partial=kk.count("partial"),
                                                   n_unfused=kk.count("unfused")))
                   for c, kk in zip(chains, kinds)]
    point_feats += [(ds[k]["c"], dict(profile=name, kind=ds[k]["kind"])) for c in chains for k in c]
    print(f"{name}: candidates {len(ds)} (clean {sum(d['kind'] == 'clean' for d in ds)}, "
          f"partial {sum(d['kind'] == 'partial' for d in ds)}, unfused {sum(d['kind'] == 'unfused' for d in ds)})  "
          f"chains {len(chains)}  dashes in chains {sum(map(len, chains))}")

write_lines(OUT / f"m1{SUFFIX}.gpkg", "m1_chains", gt, line_feats,
            dict(profile=ogr.OFTString, n_dash=ogr.OFTInteger, n_partial=ogr.OFTInteger, n_unfused=ogr.OFTInteger))
write_points(OUT / f"m1{SUFFIX}.gpkg", "m1_dashes", gt, point_feats, dict(profile=ogr.OFTString, kind=ogr.OFTString))
print(f"{time.time() - t:.0f}s")
