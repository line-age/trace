"""Re-detect text inside given boxes: the "bound, then re-detect" half of the text stage.

usage: spot_crops.py IMAGE CROPS_JSON OUT_JSON [MIN_SIZE]
  IMAGE       the working image (e.g. the sheet with borders already removed)
  CROPS_JSON  [[x0, y0, w, h], ...] in sheet pixels; each is padded with context to >= PAD px
  OUT_JSON    [{crop, text, score, poly}], polygons in sheet pixels
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # run with python -I, which drops it
import spot  # noqa: E402

PAD = 384  # px of context: the spotter reads a word better when it sees its surroundings


def main(image, crops_json, out_json, min_size=1024):
    Image.MAX_IMAGE_PIXELS = None
    rgb = np.asarray(Image.open(image).convert("RGB"))
    H, W = rgb.shape[:2]
    pred, labels = spot.predictor(min_size), spot.labels148()
    out, t = [], time.time()
    crops = json.load(open(crops_json))
    for k, (x0, y0, w, h) in enumerate(crops):
        cx, cy = x0 + w / 2, y0 + h / 2
        hw, hh = max(w / 2 + 40, PAD / 2), max(h / 2 + 40, PAD / 2)
        a0, b0 = int(max(cx - hw, 0)), int(max(cy - hh, 0))
        a1, b1 = int(min(cx + hw, W)), int(min(cy + hh, H))
        for r in spot.spot(pred, labels, rgb[b0:b1, a0:a1], 1, a0, b0):
            out.append(dict(r, crop=k))
        if k % 50 == 0:
            print(f"crop {k + 1}/{len(crops)}  {len(out)} readings  {time.time() - t:.0f}s", flush=True)
    json.dump(out, open(out_json, "w"))
    print(f"{len(crops)} crops, {len(out)} readings, {time.time() - t:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 1024)
