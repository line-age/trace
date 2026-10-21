"""Step 1a: paint out ink that is already explained, so the text spotter sees only what is left.

Grid lines and chained clean dashes are painted paper-white. Only CHAINED dashes go: a lone
dash-shaped bar may be a letter stroke, and painting a letter out would cost us the word. A dash
fused to a letter ("T" under the ILLINOIS line) is not a clean dash and stays, so this cannot
remove a letter; the pitch-based extension downstream is what handles those.

Writes out/sheet-painted.png (RGB, same pixel grid as the warped sheet).
"""

import json

import numpy as np
from osgeo import gdal
from PIL import Image
from scipy import ndimage as ndi

from common import OUT, PROFILES, SHEET, chain, load_masks


def chained_dash_pixels(m):
    ds = json.load(open(OUT / "dashes.json"))
    lab, _ = ndi.label(m["dash"], np.ones((3, 3)))
    keep = []
    for name, prof in PROFILES.items():
        idx = [i for i, e in enumerate(ds) if e["profile"] == name]
        for ch in chain([dict(c=np.array(ds[i]["c"]), ax=np.array(ds[i]["ax"])) for i in idx], prof):
            for k in ch:
                x, y = np.round(ds[idx[k]]["c"]).astype(int)
                # a bar's centroid can fall just off its own pixels; take the nearest label nearby
                win = lab[y - 2:y + 3, x - 2:x + 3]
                keep += list(np.unique(win[win > 0]))
    return np.isin(lab, keep)


if __name__ == "__main__":
    m = load_masks()
    paint = ndi.binary_dilation(m["grid"] | chained_dash_pixels(m), iterations=2)
    rgba = gdal.Open(SHEET).ReadAsArray().transpose(1, 2, 0)
    rgb = rgba[:, :, :3].copy()
    paper = np.median(rgb[(rgba[:, :, 3] > 0) & ~ndi.binary_dilation(m["hard"] | m["rest"], iterations=3)], axis=0)
    rgb[paint | (rgba[:, :, 3] == 0)] = paper.astype(np.uint8)
    Image.fromarray(rgb).save(OUT / "sheet-painted.png")
    print(f"painted {paint.mean():.1%} of the sheet, paper colour {paper}")
