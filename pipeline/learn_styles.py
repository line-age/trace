"""Learn the sheet's dash styles from the ink instead of hand-picking them.

The two hand-picked profiles (legend trunk 33 px, Dennison spur 12 px) left the commonest dash
on the sheet, ~17 px, falling between them: the Lawrence field lines barely chained at all.
Draftsmen vary dash length to keep dense areas readable, so the styles are read off the map:

  1. Thin bars outside grid, pools and text.
  2. Pairs of bars lying end to end on one line, of similar length, with a small gap
     (a gap of 2-12 px: wider gaps are dash-dot borders, a dot sitting in the gap).
  3. The pairs' dash lengths, smoothed; each peak is a style, the valleys between peaks are
     the length windows (windows do not overlap, so a bar belongs to one style).

The style nearest the legend swatch (33 px) is named "trunk"; the legend only confirms it.
Writes out/styles.json, used by common.PROFILES when TRACE_RUN includes "learned".
"""

import json

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from common import LEG, OUT, bar_stats, load_ink, load_masks

GAP = (2, 12)
MIN_PEAK = 0.15  # of the tallest peak


def pairs(bars):
    c = np.array([b["c"] for b in bars])
    a = np.array([b["ax"] for b in bars])
    L = np.array([b["length"] for b in bars])
    out = []
    for i, j in cKDTree(c).query_pairs(70):
        if abs(a[i] @ a[j]) < 0.96:
            continue
        v = c[j] - c[i]
        dist = np.linalg.norm(v)
        u = v / dist
        if abs(u @ a[i]) < 0.96 or abs(u @ a[j]) < 0.96 or max(L[i], L[j]) / min(L[i], L[j]) > 1.35:
            continue
        gap = dist - (L[i] + L[j]) / 2
        if GAP[0] <= gap <= GAP[1]:
            out.append(((L[i] + L[j]) / 2, gap))
    return np.array(out)


def main():
    ink, _, _ = load_ink()
    m = load_masks()
    z = np.load(OUT / "text_mask.npz")
    text = np.unpackbits(z["text"])[:ink.size].reshape(ink.shape).astype(bool)
    _, bars = bar_stats(m["rest"] & ~text)
    bars = [b for b in bars if b["width"] <= 9 and b["length"] >= max(7, 2 * b["width"])]
    P = pairs(bars)
    edges = np.arange(6, 52)
    h = ndi.gaussian_filter1d(np.histogram(P[:, 0], bins=edges)[0].astype(float), 2)
    centres = edges[:-1] + 0.5
    peaks = [k for k in range(1, len(h) - 1) if h[k] >= h[k - 1] and h[k] > h[k + 1] and h[k] >= MIN_PEAK * h.max()]
    cuts = [centres[0] - 0.5] + [centres[a + int(np.argmin(h[a:b]))] for a, b in zip(peaks, peaks[1:])] + [centres[-1] + 0.5]
    styles = {}
    for k, pk in enumerate(peaks):
        lo, hi = cuts[k], cuts[k + 1]
        sel = (P[:, 0] >= lo) & (P[:, 0] < hi)
        dash, gap = float(np.median(P[sel, 0])), float(np.median(P[sel, 1]))
        lo = max(lo, 0.5 * dash)  # the lowest window would otherwise run down to dot-sized blobs
        styles[k] = dict(dash=round(dash, 1), gap=round(gap, 1), slack_lo=round(dash - lo, 1), slack_hi=round(hi - dash, 1),
                         min_chain=3 if dash >= 20 else 4, pairs=int(sel.sum()))
    trunk = min(styles, key=lambda k: abs(styles[k]["dash"] - LEG["dash"]))
    names = {trunk: "trunk"}
    rest = sorted((k for k in styles if k != trunk), key=lambda k: -styles[k]["dash"])
    for k, nm in zip(rest, ("medium", "short", "tiny")):
        names[k] = nm
    styles = {names[k]: v for k, v in styles.items()}
    json.dump(styles, open(OUT / "styles.json", "w"), indent=1)
    print(f"{len(bars)} thin bars, {len(P)} end-to-end pairs")
    for nm, s in styles.items():
        print(f"  {nm:<7} dash {s['dash']:>5} px (window {s['dash'] - s['slack_lo']:.0f}-{s['dash'] + s['slack_hi']:.0f}), "
              f"gap {s['gap']} px, {s['pairs']} pairs")


if __name__ == "__main__":
    main()
