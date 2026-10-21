"""Step 0: build the explained-layer raster and the baseline (M0) chains.

Writes
  out/explained.tif   1 grid, 2 pool, 3 thick, 4 other ink, 5 clean dash (either profile)
  out/masks.npz       the same masks, for the method scripts to reuse
  out/dashes.json     clean dashes (centre, axis, profile) for M2 and M3 to start from
  out/m0.gpkg         baseline chains as lines: what we had before, for comparison
"""

import json
import time

import numpy as np
from osgeo import ogr

from common import (DASH, DASHES, GRID, OTHER, OUT, POOL, PREP, PROFILES, THICK, bar_stats, chain, classify, explain,
                    load_ink, save_masks, save_raster, write_lines)

t = time.time()
ink, gt, proj = load_ink()
m = explain(ink)
print(f"explained layers {time.time() - t:.0f}s")

lab, bars = bar_stats(m["rest"])
dashes = []
for b in bars:
    b["profile"] = classify(b)
    if b["profile"]:
        dashes.append(b)
dash_px = np.isin(lab, [d["id"] for d in dashes])

cls = np.zeros(ink.shape, np.uint8)
cls[m["rest"] & ~dash_px] = OTHER
cls[m["grid"]] = GRID
cls[m["thick"]] = THICK
cls[m["pools"]] = POOL
cls[dash_px] = DASH
save_raster(OUT / f"explained{PREP}.tif", cls, gt, proj)
m["dash"] = dash_px
save_masks(m)

json.dump([dict(c=list(map(float, d["c"])), ax=list(map(float, d["ax"])), length=d["length"], width=d["width"], profile=d["profile"])
           for d in dashes], open(DASHES, "w"))

feats = []
for name, prof in PROFILES.items():
    ds = [d for d in dashes if d["profile"] == name]
    chains = chain(ds, prof)
    feats += [([ds[k]["c"] for k in c], dict(profile=name, n_dash=len(c))) for c in chains]
    print(f"{name}: clean dashes {len(ds)}  chains {len(chains)}  dashes in chains {sum(map(len, chains))}")
write_lines(OUT / f"m0{PREP}.gpkg", "m0_chains", gt, feats, dict(profile=ogr.OFTString, n_dash=ogr.OFTInteger))
print(f"{time.time() - t:.0f}s")
