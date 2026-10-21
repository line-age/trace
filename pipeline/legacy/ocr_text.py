"""Step 1b: OCR the sheet into a text layer. Runs after grid removal, before everything else.

Order matters both ways. Ruled lines through words wreck OCR, so the grid (the one class that
cannot be confused with a letter) is painted out first. Everything after this step needs the
text layer: text is where a track may coast, and where a seed may never start.

Labels that follow a pipeline are rotated to follow it ("ILLINOIS" runs vertically beside the
N-S line, "GULF" along the trunks), so Tesseract is run at several rotations and the boxes are
rotated back. Boxes found at more than one rotation keep the most confident reading.

The danger is OCR reading a dashed line AS text ("- - - -", "I I I"). Masking those would hide
the very thing we are looking for, so a word made only of dash-like characters is rejected,
except a lone I/l/1/| that shares a text line with real words (a letter of "C L I N T O N").
Rejected boxes are written out too, with the reason, so the filter can be checked.

Writes out/text.gpkg: text_words (accepted) and text_rejected, polygons in lon/lat.
"""

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal, ogr, osr
from PIL import Image
from qgis.core import QgsGeometry, QgsPointXY
from scipy import ndimage as ndi

from common import OUT, PROFILES, chain, load_masks, to_lonlat

TESS = os.environ.get("TRACE_TESSERACT", shutil.which("tesseract") or "tesseract")
from common import SHEET  # noqa: E402
ANGLES = (0, 20, -20, 45, -45, 90, -90)
UPSCALE = 2
MIN_CONF_WORD = 30
MIN_CONF_CHAR = 80
DASHLIKE = set("Il1|!ij-_.,:;'\"`~=—–")

# lon/lat windows to OCR; None means the whole sheet
WINDOWS = {
    "mt_carmel": (-87.80, 38.40, -87.55, 38.62),
    "clinton": (-89.72, 38.30, -88.98, 38.80),
}


def window_px(gt, w):
    lon0, lat0, lon1, lat1 = w
    x0, x1 = int((lon0 - gt[0]) / gt[1]), int((lon1 - gt[0]) / gt[1])
    y0, y1 = int((lat1 - gt[3]) / gt[5]), int((lat0 - gt[3]) / gt[5])
    return x0, y0, x1 - x0, y1 - y0


def gray_without_grid(ds, grid, x0, y0, w, h):
    rgb = ds.ReadAsArray(x0, y0, w, h).astype(float)
    g = rgb[:3].mean(0)
    g[ndi.binary_dilation(grid[y0:y0 + h, x0:x0 + w], iterations=1)] = 255
    g[rgb[3] == 0] = 255
    return Image.fromarray(g.astype(np.uint8))


def tesseract(img):
    with tempfile.TemporaryDirectory(dir=OUT) as d:
        p = os.path.join(d, "in.png")
        img.save(p)
        r = subprocess.run([TESS, p, "stdout", "--psm", "11", "-l", "eng", "tsv"], capture_output=True, text=True,
                           env=dict(os.environ, OMP_THREAD_LIMIT="1"))
    rows = list(csv.DictReader(r.stdout.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE))
    return [w for w in rows if w["level"] == "5" and w["text"].strip()]


def unrotate(pts, angle, size_rot, size_orig):
    """Map points in a PIL-rotated (expand=True) image back to the unrotated image.

    PIL turns the picture counterclockwise as seen on screen, about the centre, and expand keeps
    the new centre at the middle of the bigger canvas.
    """
    t = np.radians(angle)
    c1 = np.array(size_rot) / 2
    c0 = np.array(size_orig) / 2
    q = np.asarray(pts, float) - c1
    x = q[:, 0] * np.cos(t) - q[:, 1] * np.sin(t)
    y = q[:, 0] * np.sin(t) + q[:, 1] * np.cos(t)
    return np.c_[x, y] + c0


def self_test():
    a = np.full((200, 300), 255, np.uint8)
    a[40:44, 250:254] = 0
    img = Image.fromarray(a)
    for ang in ANGLES:
        r = np.asarray(img.rotate(ang, expand=True, fillcolor=255))
        ys, xs = np.nonzero(r < 128)
        back = unrotate([[xs.mean(), ys.mean()]], ang, (r.shape[1], r.shape[0]), (300, 200))[0]
        assert np.allclose(back, [251.5, 41.5], atol=1.5), (ang, back)


def ocr_window(img, x0, y0):
    big = img.resize((img.width * UPSCALE, img.height * UPSCALE), Image.LANCZOS)

    def one(ang):
        r = big.rotate(ang, expand=True, fillcolor=255, resample=Image.BICUBIC)
        out = []
        for w in tesseract(r):
            l, t, ww, hh = (int(w[k]) for k in ("left", "top", "width", "height"))
            corners = [[l, t], [l + ww, t], [l + ww, t + hh], [l, t + hh]]
            p = unrotate(corners, ang, r.size, big.size) / UPSCALE + [x0, y0]
            out.append(dict(text=w["text"].strip(), conf=float(w["conf"]), angle=ang, poly=p,
                            line=(ang, w["block_num"], w["par_num"], w["line_num"])))
        return out

    with ThreadPoolExecutor(len(ANGLES)) as ex:
        return [w for ws in ex.map(one, ANGLES) for w in ws]


def judge(words):
    """Accept a reading if it looks like text THIS map prints, not by Tesseract's confidence.

    Tesseract was 90+ sure of "\\\\" (a diagonal dash) and 53 sure of BARTELSO, so confidence
    alone is the wrong filter. Every label on this sheet is capitals or a 3-4 digit well depth.
    Single characters are the risky case: a lone I, l, 1 or | is how a dash gets read, so those
    are never accepted alone; other single capitals and digits (the letters of letter-spaced
    county names, township numbers) need high confidence.
    """
    for w in words:
        s = w["text"].strip(".,:;'\"()")
        if s and all(ch in DASHLIKE or ch in "/\\" for ch in s):
            w["reason"] = "looks like a dash"
        elif w["angle"] != 0 and not re.fullmatch(r"[A-Z][A-Z.\-']{3,}|\d{3,4}", s):
            # Rotated labels on this sheet follow pipelines and are words ("GULF", "ILLINOIS");
            # short rotated readings were all dashed line read as "SS", "NF", "EEE".
            w["reason"] = "short rotated reading"
        elif re.fullmatch(r"[A-Z][A-Z.\-']{2,}|[A-Z]{2}|\d{3,4}", s):
            w["reason"] = "" if w["conf"] >= MIN_CONF_WORD else "low confidence"
        elif re.fullmatch(r"[A-HJ-Z2-9]", s):
            w["reason"] = "" if w["conf"] >= MIN_CONF_CHAR else "low confidence"
        else:
            w["reason"] = "not text this map prints"
    return words


def dash_evidence():
    """Every clean dash (centre, axis) plus which ones sit in baseline chains."""
    ds = json.load(open(OUT / "dashes.json"))
    c = np.array([e["c"] for e in ds])
    ax = np.array([e["ax"] for e in ds])
    chained = np.zeros(len(ds), bool)
    for name, prof in PROFILES.items():
        idx = [i for i, e in enumerate(ds) if e["profile"] == name]
        sub = [dict(c=c[i], ax=ax[i]) for i in idx]
        for ch in chain(sub, prof):
            chained[[idx[k] for k in ch]] = True
    return c, ax, chained


def reject_pipeline_readings(words, evidence):
    """At the diagonal rotations Tesseract reads a stretch of dashed line as "SS", "SSS", "NF".

    A short reading is the line, not a label, if its box holds two or more chained dashes, or
    two or more clean dashes lying along the box's reading direction (the stretches baseline
    chaining missed are exactly the ones OCR is most likely to misread). Real labels that sit ON
    a line ("ILLINOIS") are long words and survive this.
    """
    from matplotlib.path import Path as MPath
    c, ax, chained = evidence
    for w in words:
        if w["reason"] or len(w["text"]) > 4:
            continue
        inside = MPath(w["poly"]).contains_points(c)
        p = np.asarray(w["poly"])
        along = (p[1] - p[0]) / (np.linalg.norm(p[1] - p[0]) + 1e-9)
        aligned = inside & (np.abs(ax @ along) > np.cos(np.radians(25)))
        # The looser aligned-dash test only for rotated readings: upright letters have straight
        # strokes that pass as aligned dashes (it rejected G, O, T of WASHINGTON).
        if (inside & chained).sum() >= 2 or (w["angle"] != 0 and aligned.sum() >= 2):
            w["reason"] = "OCR reading a pipeline"
    return words


def dedupe(words):
    """Same word found at several rotations: keep the best reading.

    "Best" weighs length as well as confidence. A two-letter fragment read at the wrong rotation
    can be more confident than the whole word read at the right one (BARTELSO lost that way).
    """
    kept = []
    for w in sorted(words, key=lambda w: -w["conf"] * min(len(w["text"]), 8) ** 0.5):
        g = QgsGeometry.fromPolygonXY([[QgsPointXY(*p) for p in w["poly"]]])
        if any(g.intersection(k["geom"]).area() > 0.5 * min(g.area(), k["geom"].area()) for k in kept):
            continue
        w["geom"] = g
        kept.append(w)
    return kept


def write_polys(path, layer, gt, words, with_reason):
    drv = ogr.GetDriverByName("GPKG")
    ds = drv.Open(str(path), 1) if path.exists() else drv.CreateDataSource(str(path))
    if ds.GetLayerByName(layer):
        ds.DeleteLayer(layer)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lyr = ds.CreateLayer(layer, srs, ogr.wkbPolygon)
    fields = dict(text=ogr.OFTString, conf=ogr.OFTReal, angle=ogr.OFTInteger, window=ogr.OFTString)
    if with_reason:
        fields["reason"] = ogr.OFTString
    for k, v in fields.items():
        lyr.CreateField(ogr.FieldDefn(k, v))
    for w in words:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for lon, lat in to_lonlat(gt, list(w["poly"]) + [w["poly"][0]]):
            ring.AddPoint_2D(lon, lat)
        g = ogr.Geometry(ogr.wkbPolygon)
        g.AddGeometry(ring)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        for k in fields:
            f.SetField(k, w[k])
        lyr.CreateFeature(f)
    ds = None


if __name__ == "__main__":
    self_test()
    t = time.time()
    ds = gdal.Open(SHEET)
    gt = ds.GetGeoTransform()
    grid = load_masks()["grid"]
    evidence = dash_evidence()
    accepted, rejected = [], []
    for name, win in WINDOWS.items():
        x0, y0, w, h = window_px(gt, win) if win else (0, 0, ds.RasterXSize, ds.RasterYSize)
        words = reject_pipeline_readings(judge(ocr_window(gray_without_grid(ds, grid, x0, y0, w, h), x0, y0)), evidence)
        for wd in words:
            wd["window"] = name
        good = dedupe([wd for wd in words if not wd["reason"]])
        bad = [wd for wd in words if wd["reason"]]
        accepted += good
        rejected += bad
        print(f"{name}: raw words {len(words)}  accepted {len(good)}  rejected {len(bad)} "
              f"(dash-like {sum(b['reason'] == 'looks like a dash' for b in bad)}, "
              f"pipeline {sum(b['reason'] == 'OCR reading a pipeline' for b in bad)})  {time.time() - t:.0f}s")
    write_polys(OUT / "text.gpkg", "text_words", gt, accepted, False)
    write_polys(OUT / "text.gpkg", "text_rejected", gt, rejected, True)
