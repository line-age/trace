"""M2: follow each chain outward one dash period at a time, coasting through covered gaps.

From each end of a seed chain (2+ clean dashes in a row), predict where the next dash should
be: one period further along the current direction. Then look at that spot:

  dash     a thin bar of ink fits there (searching a little along, across and in angle),
           and it is not just a river or grid line lying along the path.
  coast    the spot is covered by something else (text, symbol, river, grid), so we CAN'T
           TELL. Keep going on the prediction.
  blank    the spot is empty paper. Real evidence against: two in a row ends the track.
  pool     the path runs into an oil pool. Lines end at pools, so the track ends there.

Five non-dash steps in a row also end the track ("lost"). Trailing coast/blank steps are
trimmed so a track never ends on a guess. Reaching a dash that belongs to another seed chain
ends the track as "joined".

Writes out/m2.gpkg: m2_tracks, one feature per step, with status and the track's stop reason.
"""

import json
import time

import numpy as np
from osgeo import ogr
from scipy.spatial import cKDTree

from common import DASHES, LEG, OUT, PROFILES, SUFFIX, anchored, candidates, chain, rhythm_ok, in_mask, load_ink, load_masks, load_text, write_lines

MAX_MISS, MAX_BLANK = 5, 2


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def rot(d, deg):
    r = np.radians(deg)
    return np.array([d[0] * np.cos(r) - d[1] * np.sin(r), d[0] * np.sin(r) + d[1] * np.cos(r)])


class Tracker:
    """The sheet as the tracker sees it: ink, the explained layers, and ink other layers own."""

    def __init__(self, ink, m, text):
        self.ink, self.m, self.text = ink, m, text
        self.H, self.W = ink.shape
        # ink that is something other than a clean dash, or that another layer owns
        self.covered = m["hard"] | (m["rest"] & ~m["dash"]) | text
        self.along_path = m["grid"] | m["thick"]  # ink that can masquerade as a dash if it runs our way
        self.flank_ink = ink & ~self.along_path

    def bar(self, prof, c, d):
        """Pixel samples for a dash-sized bar at c pointing along d, plus its two side strips."""
        n = np.array([-d[1], d[0]])
        a = np.arange(-prof["dash"] / 2, prof["dash"] / 2 + 1)
        half = LEG["thick"] / 2
        core = c + a[:, None, None] * d + np.arange(-half, half + 1)[None, :, None] * n
        side = np.r_[np.arange(-half - 6, -half - 2), np.arange(half + 3, half + 7)]
        flank = c + a[:, None, None] * d + side[None, :, None] * n
        return core.reshape(-1, 2), flank.reshape(-1, 2)

    def frac(self, mask, pts):
        x = np.clip(np.round(pts[:, 0]).astype(int), 0, self.W - 1)
        y = np.clip(np.round(pts[:, 1]).astype(int), 0, self.H - 1)
        return float(mask[y, x].mean())

    def look(self, prof, pred, d):
        core, flank = self.bar(prof, pred, d)
        here = dict(fill=self.frac(self.ink, core), flank=self.frac(self.flank_ink, flank),
                    pool=self.frac(self.m["pools"], core), covered=self.frac(self.covered, core))
        best, best_score = None, -9
        s = prof["dash"] / 33
        n = np.array([-d[1], d[0]])
        for da in np.array([-6, -3, 0, 3, 6]) * s:
            for dn in (-4, -2, 0, 2, 4):
                for ang in (-10, -5, 0, 5, 10):
                    dd = rot(d, ang)
                    c = pred + da * d + dn * n
                    core, flank = self.bar(prof, c, dd)
                    fill, fl = self.frac(self.ink, core), self.frac(self.flank_ink, flank)
                    if fill - fl > best_score:
                        best_score, best = fill - fl, dict(c=c, d=dd, fill=fill, flank=fl,
                                                           along=self.frac(self.along_path, core))
        # a dash-shaped hit inside a label is a letter stroke as often as not: coast through it instead
        if best["fill"] >= 0.6 and best["flank"] <= 0.3 and best["along"] < 0.5 and not in_mask(self.text, best["c"]):
            return "dash", best
        if here["pool"] >= 0.3:
            return "pool", None
        if here["fill"] < 0.2 and here["flank"] < 0.1:
            return "blank", None
        return "coast", None

    def track(self, prof, pts, owner, tree, me):
        last, d = pts[-1], unit(pts[-1] - pts[-2])
        steps, miss, blank, why = [], 0, 0, "lost"
        for _ in range(500):
            pred = last + d * prof["period"]
            if not (0 <= pred[0] < self.W and 0 <= pred[1] < self.H):
                why = "edge"
                break
            status, hit = self.look(prof, pred, d)
            if status == "dash":
                steps.append((hit["c"], "dash"))
                d = unit(0.6 * d + 0.4 * unit(hit["c"] - last))
                last, miss, blank = hit["c"], 0, 0
                dist, k = tree.query(hit["c"])
                if dist < 0.4 * prof["period"] and owner[k] != me:
                    why = "joined"
                    break
                continue
            if status == "pool":
                steps.append((pred, "coast"))
                why = "pool"
                break
            steps.append((pred, status))
            last = pred
            miss += 1
            blank = blank + 1 if status == "blank" else 0
            if blank >= MAX_BLANK:
                why = "blank"
                break
            if miss >= MAX_MISS:
                why = "lost"
                break
        # Never end on a guess. At a pool, keep the one step into it only if a dash came right before:
        # keeping every coast let a track drift off its line (along "1672") and still "reach" a pool.
        last_into_pool = steps.pop() if why == "pool" and len(steps) >= 2 and steps[-2][1] == "dash" else None
        while steps and steps[-1][1] != "dash":
            steps.pop()
        if last_into_pool:
            steps.append(last_into_pool)
        return steps, why


def run(ink, m, text, dashes, log=print):
    """Track every anchored seed chain; returns line features (xy, attrs), one per step."""
    tr = Tracker(ink, m, text)
    feats = []
    tid = 0
    for name, prof in PROFILES.items():
        ds = [dict(c=np.array(e["c"]), ax=np.array(e["ax"])) for e in candidates(dashes, prof)]
        if len(ds) < 2:
            continue
        seeds = [c for c in anchored(chain(ds, prof, min_len=2), ds, text) if rhythm_ok([ds[k]["c"] for k in c], prof)]
        owner = np.full(len(ds), -1)
        for s, c in enumerate(seeds):
            owner[c] = s
        tree = cKDTree([e["c"] for e in ds])
        kept = 0
        for s, c in enumerate(seeds):
            pts = [ds[k]["c"] for k in c]
            # Predict with this line's own spacing, not the style's average: draftsmen vary the gap
            # line to line. A step over a hidden dash is two periods, so take the shortest typical one.
            sp = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            own = float(np.percentile(sp, 25)) if len(sp) else prof["period"]
            line_prof = dict(prof, period=float(np.clip(own, 0.7 * prof["period"], 1.5 * prof["period"])))
            found, parts = 0, []
            for direction in (pts, pts[::-1]):
                steps, why = tr.track(line_prof, direction, owner, tree, s)
                found += sum(st == "dash" for _, st in steps)
                prev = direction[-1]
                for p, st in steps:
                    parts.append(([prev, p], dict(status=st, stop=why)))
                    prev = p
            if len(c) + found < prof["min_chain"]:
                continue
            kept += 1
            feats.append((pts, dict(track=tid, profile=name, status="seed", stop="")))
            feats += [(xy, dict(a, track=tid, profile=name)) for xy, a in parts]
            tid += 1
        log(f"{name}: seeds {len(seeds)}  tracks kept {kept}")
    return feats


FIELDS = dict(track=ogr.OFTInteger, profile=ogr.OFTString, status=ogr.OFTString, stop=ogr.OFTString)


if __name__ == "__main__":
    t = time.time()
    ink, gt, proj = load_ink()
    feats = run(ink, load_masks(), load_text(ink.shape), json.load(open(DASHES)))
    write_lines(OUT / f"m2{SUFFIX}.gpkg", "m2_tracks", gt, feats, FIELDS)
    st = [a["status"] for _, a in feats]
    print({k: st.count(k) for k in set(st)}, f"{time.time() - t:.0f}s")
