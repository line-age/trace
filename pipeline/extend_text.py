"""Step 1c: finish letter-spaced labels the spotter only partly read ("C L I N" of CLINTON).

County names on this sheet are printed with wide, even letter spacing, so they have a rhythm
just as a dashed line does: the next letter sits one pitch further along the baseline, at the
same height. The spotter reads them a few letters at a time and stops where a pipeline or pool
runs into a letter. So, for each letter-spaced reading:

  1. Straighten a strip along its baseline (strip coordinates: t along, b across).
  2. Find letters as runs of inked columns in the band |b| <= h/2. Long strokes running along
     the strip (a pipeline through the tops of T O N) are removed first: no letter part is
     longer than a letter is tall. A letter must also fill most of the band's height and must
     not carry on past it (a river or N-S line crossing the band does).
  3. Walk outward one pitch at a time with the same three-valued evidence as the dash tracker:
       letter   a letter blob sits within a third of a pitch of the prediction
       covered  pool, grid or pipeline ink is there: can't tell, coast (at most 2 in a row)
       blank    paper: the word has ended
  4. Read each letter on its own (Tesseract, one character, capitals only) and match the
     string, with "?" for coasted letters, against the county gazetteer.

Pieces of the same name (GREE + ENE) end up as one extended label; readings that fall inside
an extended label's span are absorbed into it.

Writes out/text_extended.json (pixel coordinates, for the text mask) and out/text.gpkg layers
text_extended_letters (one box per letter) and text_extended: one polygon per extended label, with the spotter's
reading, the letter-by-letter reading, the gazetteer match, and which pieces it absorbed.
"""

import csv
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr
from PIL import Image
from scipy import ndimage as ndi

from common import DATA, OUT, SPOTS, load_ink, load_masks, to_lonlat
from paint_out import chained_dash_pixels

TESS = os.environ.get("TRACE_TESSERACT", shutil.which("tesseract") or "tesseract")
GAZ = DATA / "gaz/fips.csv"
STATES = ("IL", "IN", "KY", "MO", "TN")
MIN_H, MIN_PITCH = 30, 1.5     # county lettering: 40-55 px tall, pitch > 1.5 letter heights
MAX_COAST = 2
POOL_SQUARE = 15  # county letter strokes are ~12 px; pools are far wider


def geom(r):
    p = np.array(r["poly"])
    n = len(p) // 2
    top, bot = p[:n], p[n:][::-1]
    mid = (top + bot) / 2
    L = float(np.linalg.norm(mid[-1] - mid[0]))
    u = (mid[-1] - mid[0]) / (L + 1e-9)
    return dict(o=mid[0], u=u, n=np.array([-u[1], u[0]]), L=L, h=float(np.linalg.norm(top - bot, axis=1).mean()))


def strip(arrays, g, t0, t1, order=0):
    """Sample each array on a (b, t) grid along the label's baseline, b in [-h, h]."""
    t = np.arange(t0, t1)
    b = np.arange(-int(g["h"]), int(g["h"]) + 1)
    xy = g["o"][None, None, :] + t[None, :, None] * g["u"] + b[:, None, None] * g["n"]
    H, W = arrays[0].shape
    x0, y0 = max(int(xy[..., 0].min()) - 2, 0), max(int(xy[..., 1].min()) - 2, 0)
    x1, y1 = min(int(xy[..., 0].max()) + 3, W), min(int(xy[..., 1].max()) + 3, H)
    coords = [xy[..., 1] - y0, xy[..., 0] - x0]  # sample a crop: converting the whole sheet per label is slow
    return [ndi.map_coordinates(a[y0:y1, x0:x1].astype(float), coords, order=order, cval=0) for a in arrays], t, b


def letters(ink_s, b, h, chained_s=None):
    """Letter blobs in a straightened strip: (t_start, t_end) column ranges.

    chained_s (pipeline dash ink) is cut away before the spill test only, so a pipeline fused to
    a letter (the T of CLINTON) cannot make the letter look like it runs out of the band.
    """
    band = np.abs(b) <= 0.55 * h
    flank = (np.abs(b) > 0.75 * h) & (np.abs(b) <= 0.95 * h)
    # Strokes running along the strip: anything longer than any letter part (1.5 h), plus pipeline
    # dashes running our way (33 px, through the tops of O and N). Only CHAINED dash ink is cut at
    # the shorter length: cutting every 0.7 h stroke took S, C and O apart, having no stems.
    along = ndi.binary_opening(ink_s, np.ones((1, int(1.5 * h)), bool))
    if chained_s is not None:
        along |= ndi.binary_opening(chained_s & ink_s, np.ones((1, int(0.6 * h)), bool))
    let = ink_s & ~ndi.binary_dilation(along, iterations=1)
    # Letter columns are TALL columns: the band holds at least 0.3 h of ink there. A dash lying
    # along the label is ~7 px deep and never qualifies, so dashes beside a letter cannot widen it
    # into a non-letter (T O N of CLINTON). Gaps up to 0.6 h close, keeping H's two stems together.
    tall = let[band].sum(0) >= 0.3 * h
    cols = ndi.binary_closing(np.pad(tall, int(h)), np.ones(int(0.6 * h), bool))[int(h):-int(h)]
    lab, n = ndi.label(cols)
    # Spill is ink CONNECTED to the letter that carries on past the band (a river or N-S line
    # through it). Unconnected ink in the flank (a dashed line passing just above) is not spill.
    sep = let & ~ndi.binary_dilation(chained_s, iterations=1) if chained_s is not None else let
    comp, _ = ndi.label(sep, np.ones((3, 3)))
    reach = np.zeros(comp.max() + 1, bool)
    reach[np.unique(comp[flank[:, None] & sep])] = True
    out = []
    for sl in ndi.find_objects(lab):
        a, z = sl[0].start, sl[0].stop
        blk = let[:, a:z]
        cover = blk[band].any(1).mean()         # share of the band's height that has ink
        ids = comp[:, a:z][band][sep[:, a:z][band]]
        spill = reach[ids].mean() if len(ids) else 0.0  # share of the letter's ink that runs out of the band
        if z - a >= 3 and cover >= 0.7 and spill < 0.5 and z - a <= 1.3 * h:
            out.append((a, z))
    return out


def walk(blobs, cover_s, t, b, h, seed, pitch):
    """Extend the seed letters both ways; returns the letter list with coasted gaps as None."""
    centre = {i: (t[a] + t[z - 1]) / 2 for i, (a, z) in enumerate(blobs)}
    band = np.abs(b) <= 0.55 * h
    seq = [("L", i) for i in seed]
    for direction in (1, -1):
        last = centre[seed[-1] if direction == 1 else seed[0]]
        p = pitch
        coast = 0
        while True:
            if p is None:  # one seed letter: the nearest letter-sized blob within 6 heights sets the pitch
                cand = [i for i in centre if 1.0 * h <= direction * (centre[i] - last) <= 6 * h]
                if not cand:
                    break
                k = min(cand, key=lambda i: direction * (centre[i] - last))
                p = abs(centre[k] - last)
            else:
                pred = last + direction * p
                near = [i for i in centre if abs(centre[i] - pred) <= p / 3 and ("L", i) not in seq]
                if near:
                    k = min(near, key=lambda i: abs(centre[i] - pred))
                else:
                    cols = np.abs(t - pred) <= 0.4 * p
                    if cols.any() and cover_s[band][:, cols].mean() > 0.15 and coast < MAX_COAST:
                        seq = seq + [("C", pred)] if direction == 1 else [("C", pred)] + seq
                        last, coast = pred, coast + 1
                        continue
                    break
            seq = seq + [("L", k)] if direction == 1 else [("L", k)] + seq
            last, coast = centre[k], 0
        while seq and seq[-1][0] == "C":  # never end on a guess
            seq.pop()
        while seq and seq[0][0] == "C":
            seq.pop(0)
    return seq, centre


def read_char(img):
    with tempfile.TemporaryDirectory(dir=OUT) as d:
        p = Path(d) / "c.png"
        img.save(p)
        r = subprocess.run([TESS, str(p), "stdout", "--psm", "10", "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
                           capture_output=True, text=True)
    s = re.sub(r"[^A-Z]", "", r.stdout)
    return s[:1] or "?"


def gazetteer():
    names = set()
    for row in csv.DictReader(open(GAZ)):
        if row["state"] in STATES and row["name"].endswith(" County"):
            names.add(row["name"][:-7].upper().replace(" ", "").replace(".", ""))
    return sorted(names)


def match(s, names):
    """Best gazetteer name for a reading with '?' for unread letters, or '' if nothing is close."""
    def score(n):
        if len(n) != len(s):
            return difflib.SequenceMatcher(None, s.replace("?", ""), n).ratio() * 0.9
        known = [(a, c) for a, c in zip(s, n) if a != "?"]
        return sum(a == c for a, c in known) / max(len(known), 1) * (len(known) / len(s)) ** 0.25
    best = max(names, key=score)
    return (best, round(score(best), 2)) if score(best) >= 0.7 else ("", round(score(best), 2))


def main(cut=None, quiet=False):
    """cut: ink owned by other line layers (pipeline, borders) from an earlier pass. Without one,
    chained dash pixels stand in for the pipeline. It is cover, and is cut from the spill test,
    but never removed from the letter ink (see below)."""
    spots = json.load(open(SPOTS))
    ink, gt, _ = load_ink()
    m = load_masks()
    chained = cut if cut is not None else chained_dash_pixels(m)
    # Chained dashes are NOT removed from the letter ink: the diagonals of W and N are dash-sized
    # bars that chain, and removing them erased those letters. Pipelines running along the label
    # are dropped by the along-stroke opening in letters(), ones crossing it by the spill test.
    # Pools are NOT taken from prep's pool mask either: county lettering is bold enough that an
    # 11x11 square fits in the strokes of N and W, so prep calls them pool. Here a pool is ink a
    # 15x15 square fits inside, found per strip (see POOL_SQUARE).
    ink_l = ink & ~m["grid"]
    cover = m["grid"] | chained
    gray = np.asarray(Image.open(OUT / "sheet-painted.png").convert("L"))
    names = gazetteer()

    gs = [geom(r) for r in spots]
    seeds = [i for i, (r, g) in enumerate(zip(spots, gs))
             if g["h"] >= MIN_H and len(r["text"].strip("#")) >= 1
             and (len(r["text"].strip("#")) == 1 or g["L"] / len(r["text"].strip("#")) / g["h"] >= MIN_PITCH)]
    used, labels = set(), []
    for i in sorted(seeds, key=lambda i: -len(spots[i]["text"].strip("#"))):
        if i in used:
            continue
        g, r = dict(gs[i]), spots[i]
        if len(r["text"].strip("#")) == 1:
            if not re.fullmatch(r"[A-Z]", r["text"].strip("#")):
                continue  # a lone digit is a township number, not the start of a name
            # one letter's box gives no usable baseline angle; county names here run E-W
            c = g["o"] + g["u"] * g["L"] / 2
            g.update(u=np.array([1.0, 0.0]), n=np.array([0.0, 1.0]), o=c - [g["L"] / 2, 0])
        reach = 30 * g["h"]  # CLINTON runs ~1000 px from its C: 12 h stopped the strip before the T
        (ink_s, cov_s, ch_s, gray_s), t, b = strip([ink_l, cover, chained, gray], g, -reach, g["L"] + reach)
        ink_s, cov_s, ch_s = ink_s > 0.5, cov_s > 0.5, ch_s > 0.5
        pool = ndi.binary_dilation(ndi.binary_opening(ink_s, np.ones((POOL_SQUARE, POOL_SQUARE), bool)), iterations=3) & ink_s
        ink_s, cov_s = ink_s & ~pool, cov_s | pool
        blobs = letters(ink_s, b, g["h"], ch_s)
        centre = [(t[a] + t[z - 1]) / 2 for a, z in blobs]
        seed = [k for k, c in enumerate(centre) if -0.3 * g["h"] <= c <= g["L"] + 0.3 * g["h"]]
        if not seed:
            continue
        pitch = float(np.median(np.diff([centre[k] for k in seed]))) if len(seed) > 1 else None
        if pitch is not None and pitch < MIN_PITCH * g["h"]:
            continue  # tightly set word after all
        seq, cmap = walk(blobs, cov_s, t, b, g["h"], seed, pitch)
        if sum(1 for k, _ in seq if k == "L") < 2:
            continue
        h = g["h"]
        pos = [cmap[v] if k == "L" else v for k, v in seq]

        def span_of(j):
            gj = gs[j]
            p0 = (gj["o"] - g["o"]) @ g["u"]
            return sorted((p0, p0 + gj["L"] * (gj["u"] @ g["u"])))

        def on_line(j):
            gj = gs[j]
            c = gj["o"] + gj["u"] * gj["L"] / 2 - g["o"]
            return j != i and abs(c @ g["n"]) <= 0.4 * h and abs(gj["h"] - h) < 0.35 * h

        # A whole name the spotter read on its own (JOHNSON next to POPE) is a different label:
        # the walk may not run into it, so cut the sequence at the first slot inside one.
        own = {k for k, x in enumerate(pos) if -0.3 * h <= x <= g["L"] + 0.3 * h}
        for j in range(len(spots)):
            if on_line(j) and spots[j]["text"] in names:
                a, z = span_of(j)
                inside = [k for k, x in enumerate(pos) if a - 0.3 * h <= x <= z + 0.3 * h and k not in own]
                for k in inside:
                    if k > max(own):
                        seq, pos = seq[:k], pos[:k]
                        break
                    if k < min(own):
                        cut = max(x for x in inside if x < min(own)) + 1
                        own = {o - cut for o in own}
                        seq, pos = seq[cut:], pos[cut:]
                        break
        if sum(1 for k, _ in seq if k == "L") < 2:
            continue
        span0, span1 = pos[0] - 0.6 * h, pos[-1] + 0.6 * h
        absorbed = [j for j in range(len(spots)) if on_line(j) and span0 <= np.mean(span_of(j)) <= span1]
        used |= {i, *absorbed}

        # Reading: the spotter's own characters wherever one of its pieces lines up one-to-one
        # with the letter slots under it; Tesseract, one character at a time, for the rest.
        chars = [None] * len(seq)
        for j in [i] + absorbed:
            a, z = span_of(j)
            slots = [k for k, x in enumerate(pos) if a - 0.3 * h <= x <= z + 0.3 * h]
            txt = re.sub(r"[^A-Z#]", "", spots[j]["text"].upper())
            if len(txt) == len(slots):
                for k, ch in zip(slots, txt):
                    if ch != "#" and chars[k] is None:
                        chars[k] = ch
        band = np.abs(b) <= 0.6 * h
        for k, (kind, v) in enumerate(seq):
            if chars[k] is not None:
                continue
            if kind == "C":
                chars[k] = "?"
                continue
            a, z = blobs[v]
            crop = np.clip(gray_s[band][:, max(a - 4, 0):z + 4], 0, 255).astype(np.uint8)
            img = Image.fromarray(np.pad(crop, 20, constant_values=235))
            chars[k] = read_char(img.resize((img.width * 2, img.height * 2), Image.LANCZOS)).lower()
        chars = "".join(chars)  # lower case marks a Tesseract read, upper case the spotter's
        name, sc = match(chars.upper(), names)
        o, u, n = g["o"], g["u"], g["n"]
        poly = [o + u * span0 - n * h / 2, o + u * span1 - n * h / 2, o + u * span1 + n * h / 2, o + u * span0 + n * h / 2]
        cells = []  # one box per letter actually seen, for masking: the gaps between letters are paper
        for kind, v in seq:
            if kind == "L":
                a, z = blobs[v]
                t0, t1 = t[a] - 2, t[z - 1] + 2
                cells.append([o + u * t0 - n * 0.6 * h, o + u * t1 - n * 0.6 * h, o + u * t1 + n * 0.6 * h, o + u * t0 + n * 0.6 * h])
        labels.append(dict(poly=poly, cells=cells, pieces=",".join(map(str, [i] + absorbed)),
                           spotter=r["text"], letters=chars, gazetteer=name, match=sc,
                           pitch=round(float(np.median(np.diff([cmap[v] for k, v in seq if k == "L"]))), 1),
                           absorbed=" + ".join(spots[j]["text"] for j in absorbed)))
    write(labels, gt)
    json.dump([dict(cells=[np.asarray(c).round(1).tolist() for c in lb["cells"]], pieces=lb["pieces"],
                    name=lb["gazetteer"], letters=lb["letters"]) for lb in labels],
              open(OUT / "text_extended.json", "w"))
    if quiet:
        return labels
    for lb in sorted(labels, key=lambda x: x["spotter"]):
        print(f"{lb['spotter']:>14} -> {lb['letters']:<14} {lb['gazetteer'] or '-':<14} {lb['match']:.2f}  absorbed: {lb['absorbed']}")
    print(len(labels), "extended labels,", sum(bool(lb["gazetteer"]) for lb in labels), "matched")
    return labels


def write(labels, gt):
    path = OUT / "text.gpkg"
    ds = ogr.Open(str(path), 1) if path.exists() else ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    if ds.GetLayerByName("text_extended"):
        ds.DeleteLayer("text_extended")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lyr = ds.CreateLayer("text_extended", srs, ogr.wkbPolygon)
    fields = dict(spotter=ogr.OFTString, letters=ogr.OFTString, gazetteer=ogr.OFTString, match=ogr.OFTReal,
                  pitch=ogr.OFTReal, absorbed=ogr.OFTString, pieces=ogr.OFTString)
    for k, v in fields.items():
        lyr.CreateField(ogr.FieldDefn(k, v))
    for lb in labels:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for lon, lat in to_lonlat(gt, list(lb["poly"]) + [lb["poly"][0]]):
            ring.AddPoint_2D(lon, lat)
        g = ogr.Geometry(ogr.wkbPolygon)
        g.AddGeometry(ring)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        for k in fields:
            f.SetField(k, lb[k])
        lyr.CreateFeature(f)
    if ds.GetLayerByName("text_extended_letters"):
        ds.DeleteLayer("text_extended_letters")
    cl = ds.CreateLayer("text_extended_letters", srs, ogr.wkbPolygon)
    cl.CreateField(ogr.FieldDefn("label", ogr.OFTString))
    for lb in labels:
        for cell in lb["cells"]:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for lon, lat in to_lonlat(gt, list(cell) + [cell[0]]):
                ring.AddPoint_2D(lon, lat)
            g = ogr.Geometry(ogr.wkbPolygon)
            g.AddGeometry(ring)
            f = ogr.Feature(cl.GetLayerDefn())
            f.SetGeometry(g)
            f.SetField("label", lb["gazetteer"] or lb["letters"])
            cl.CreateFeature(f)
    ds = None


if __name__ == "__main__":
    main()
