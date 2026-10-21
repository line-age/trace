"""Fit test: MapTextPipeline (Rumsey fine-tune, the mapKurator-style spotter MapReader ships) on our windows.

Run with the spotter venv (spotter/setup.sh), not QGIS's Python. main() is the original fit
test: it writes $TRACE_WORK/text/<window>_<scale>.json, one record per text instance with the
decoded string, score, and boundary polygon in full-sheet pixel coordinates. spot_sheet.py and
spot_crops.py use predictor() and spot() from here.
"""

import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from maptextpipeline.utils.vitae_predictor import ViTAEPredictor
from maptextpipeline.config import get_cfg
from maptextpipeline.modeling.vitae_v2 import vitae_v2  # noqa: F401  registers build_vitaev2_backbone

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("TRACE_WORK", HERE.parent.parent / "work")).resolve()
SPOTTER = WORK / "spotter"   # venv, model source and weights (spotter/setup.sh)
TEXT = WORK / "text"         # readings go here
CFG = SPOTTER / "src/MapTextPipeline/configs/ViTAEv2_S/rumsey/final_rumsey.yaml"
WEIGHTS = SPOTTER / "rumsey-finetune.pth"


def labels148():
    """The 147-character vocabulary, lifted from MapReader's runner rather than retyped."""
    tree = ast.parse((SPOTTER / "maptext_runner_ref.py").read_text())  # MapReader's, fetched by setup.sh
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and len(node.elts) == 147:
            return [ast.literal_eval(e) for e in node.elts]
    raise RuntimeError("148-voc label list not found")


def decode(rec, labels):
    out, last = "", None
    for c in rec:
        c = int(c)
        if c < len(labels):
            if c != last:
                out += labels[c]
            last = c
        else:
            last = None
    return out


def predictor(min_size=None):
    torch.set_num_threads(8)
    cfg = get_cfg()
    cfg.merge_from_file(str(CFG))
    cfg.MODEL.WEIGHTS = str(WEIGHTS)
    cfg.MODEL.DEVICE = "cpu"
    if min_size:  # the config upsamples every image to a 2000 px short side
        cfg.INPUT.MIN_SIZE_TEST = min_size
    return ViTAEPredictor(cfg)  # pads to a multiple of 32, which ViTAE needs (MapReader overwrites it by mistake)


def spot(pred, labels, rgb, s, x0, y0):
    """Text instances in an RGB array upscaled by s, polygons in sheet pixels (offset x0, y0)."""
    img = Image.fromarray(np.asarray(rgb))
    big = img.resize((img.width * s, img.height * s), Image.LANCZOS) if s != 1 else img
    inst = pred(np.asarray(big)[:, :, ::-1])["instances"].to("cpu")  # predictor wants BGR
    recs = []
    for score, rec, bd in zip(inst.scores.tolist(), inst.recs, np.asarray(inst.bd)):
        top, bottom = np.hsplit(bd, 2)
        poly = np.vstack([top, bottom[::-1]]) / s + [x0, y0]
        recs.append(dict(text=decode(rec, labels), score=round(score, 3), poly=poly.round(1).tolist()))
    return recs


def main(scales, min_size=None):
    pred, labels = predictor(min_size), labels148()
    offsets = json.load(open(TEXT / "in/offsets.json"))
    TEXT.mkdir(exist_ok=True)
    for name, (x0, y0, _, _) in offsets.items():
        img = Image.open(TEXT / f"in/{name}.png").convert("RGB")
        for s in scales:
            t = time.time()
            recs = spot(pred, labels, img, s, x0, y0)
            json.dump(recs, open(TEXT / f"{name}_x{s}_m{min_size or 2000}.json", "w"))
            print(f"{name} x{s} min{min_size or 2000}: {len(recs)} instances, {time.time() - t:.0f}s  "
                  + " ".join(r["text"] for r in sorted(recs, key=lambda r: -r["score"])[:60]), flush=True)


if __name__ == "__main__":
    # usage: spot.py MIN_SIZE [SCALE ...]; MIN_SIZE 0 keeps the config value
    main([int(a) for a in sys.argv[2:]] or [1], int(sys.argv[1]) if len(sys.argv) > 1 else None)
