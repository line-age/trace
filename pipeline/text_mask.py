"""Step 1d: rasterize the text layer into the mask the methods use.

Text is occlusion, never "nothing here": a track may coast through it, and a dash inside it can
never start a chain (letters pass as dashes). The mask is the spotter's readings plus, for the
letter-spaced names the extension rebuilt, one box per letter. The spotter's own box for those
names is dropped: it spans the paper between letters, and the E-W Illinois line runs 12 dash
periods through the gaps of C L I N T O N.

Writes out/text_mask.npz (packbits, same layout as masks.npz).
"""

import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from common import OUT, SPOTS, load_ink

GROW = 2  # px: the spotter's outlines hug the letters


def build(shape):
    spots = json.load(open(SPOTS))
    ext = json.load(open(OUT / "text_extended.json"))
    replaced = {int(k) for lb in ext for k in lb["pieces"].split(",")}
    img = Image.new("1", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for k, r in enumerate(spots):
        if k not in replaced:
            d.polygon([tuple(p) for p in r["poly"]], fill=1)
    for lb in ext:
        for cell in lb["cells"]:
            d.polygon([tuple(p) for p in cell], fill=1)
    return ndi.binary_dilation(np.asarray(img, bool), iterations=GROW), len(spots) - len(replaced), sum(len(lb["cells"]) for lb in ext)


if __name__ == "__main__":
    ink, _, _ = load_ink()
    mask, n_spot, n_cells = build(ink.shape)
    np.savez_compressed(OUT / "text_mask.npz", text=np.packbits(mask), shape=np.array(mask.shape))
    print(f"text mask: {n_spot} spotter readings + {n_cells} letter boxes, {mask.mean():.2%} of the sheet, "
          f"{(mask & ink).sum() / ink.sum():.1%} of the ink")
