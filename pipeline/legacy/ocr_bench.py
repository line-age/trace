"""Scratch benchmark for OCR settings on the Clinton window.

TARGET is the list of labels in that window that are readable as whole words (letter-spaced
county names left out), read by eye off the scan. A label counts as found if any reading at
any rotation is a close spelling match, whatever Tesseract's confidence.
"""

import csv
import difflib
import os
import re
import runpy
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from osgeo import gdal
from PIL import Image

import common

L = runpy.run_path(str(Path(__file__).with_name("ocr_text.py")), run_name="lib")
TARGET = ("CARLYLE FROGTOWN ABAN BARTELSO ILLINOIS GULF NASHVILLE SANDOVAL JUNCTION CITY BROWN WAMAC "
          "PATOKA LANGEWISCH KUESTER BASE").split()


def clinton_gray():
    m = common.load_masks()
    ink, gt, _ = common.load_ink()
    ds = gdal.Open(L["SHEET"])
    x0, y0, w, h = L["window_px"](gt, L["WINDOWS"]["clinton"])
    return L["gray_without_grid"](ds, m["grid"], x0, y0, w, h)


def tess(img, psm=11, extra=()):
    with tempfile.TemporaryDirectory(dir=str(common.OUT)) as d:
        p = os.path.join(d, "i.png")
        img.save(p)
        r = subprocess.run([L["TESS"], p, "stdout", "--psm", str(psm), "-l", "eng", *extra, "tsv"],
                           capture_output=True, text=True, env=dict(os.environ, OMP_THREAD_LIMIT="1"))
    return [x for x in csv.DictReader(r.stdout.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
            if x["level"] == "5" and x["text"].strip()]


def run(img, up=2, psm=11, angles=(0, 20, -20, 45, -45, 90, -90), extra=()):
    big = img.resize((img.width * up, img.height * up), Image.LANCZOS)

    def one(a):
        return [(a, x["text"], float(x["conf"]))
                for x in tess(big.rotate(a, expand=True, fillcolor=255, resample=Image.BICUBIC), psm, extra)]

    with ThreadPoolExecutor(len(angles)) as ex:
        return [y for ys in ex.map(one, angles) for y in ys]


def shear(img, s):
    w, h = img.size
    off = abs(s) * h
    return img.transform((int(w + off), h), Image.AFFINE, (1, s, -off if s > 0 else 0, 0, 1, 0),
                         resample=Image.BICUBIC, fillcolor=255)


def score(words):
    found = {}
    for a, t, c in words:
        tt = re.sub(r"[^A-Z]", "", t.upper())
        for T in TARGET:
            if len(tt) >= 3 and difflib.SequenceMatcher(None, tt, T).ratio() >= 0.8:
                if T not in found or c > found[T][1]:
                    found[T] = (a, round(c), t)
    return found
