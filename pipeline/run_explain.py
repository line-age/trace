"""Remove, then explain what remains. The order:

  pass 1   text detections removed -> boundaries -> rivers   (to get a borderless map)
  pass 2   text RE-DETECTED inside each grown bound, on the borderless map
  final    from the original: all text found removed -> boundaries and rivers (on the text-free
           map) -> legend symbols and areas -> grid; then the residual is classified

Writes out/explain_paths.gpkg (every path, with its class and scores), out/explain_remains.tif
(the cleaned image: what the classifier saw), out/explain_removed.tif (which stage removed
what: 1 text, 2 boundary, 3 river, 4 legend symbols/areas, 5 grid) and prints each stage.

Run with TRACE_RUN=learned (for the tilt-aware grid in the learned prep masks).
"""

import json
import sys
import time

import numpy as np
from osgeo import gdal, ogr

import classify_residual as cr
import stages
from common import OUT, SPOTS, load_masks, write_lines
from explain import SHEET, State, ink_mask, paper_background


def save_rgb(path, rgb, gt, proj):
    d = gdal.GetDriverByName("GTiff").Create(str(path), rgb.shape[1], rgb.shape[0], 3, gdal.GDT_Byte, ["COMPRESS=DEFLATE"])
    d.SetGeoTransform(gt)
    d.SetProjection(proj)
    for k in range(3):
        d.GetRasterBand(k + 1).WriteArray(rgb[..., k])
    d = None


def save_classes(path, state, gt, proj):
    cls = np.zeros(state.ink.shape, np.uint8)
    for k, name in enumerate(("text", "boundary", "river", "legend", "grid"), 1):
        for stage, m in state.by.items():
            if stage == name or stage.startswith(name + ":"):
                cls[m] = k
    d = gdal.GetDriverByName("GTiff").Create(str(path), cls.shape[1], cls.shape[0], 1, gdal.GDT_Byte, ["COMPRESS=DEFLATE"])
    d.SetGeoTransform(gt)
    d.SetProjection(proj)
    d.GetRasterBand(1).WriteArray(cls)
    d.GetRasterBand(1).SetNoDataValue(0)
    d = None


def main():
    t0 = time.time()
    ds = gdal.Open(SHEET)
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    arr = ds.ReadAsArray()
    rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0))
    ink, paper_L = ink_mask(rgb, arr[3] if arr.shape[0] > 3 else None)
    paper = paper_background(rgb, ink)
    print(f"ink {ink.mean():.1%} of the sheet (paper L* {paper_L:.0f})  {time.time() - t0:.0f}s", flush=True)

    legend = json.load(open(OUT / "legend.json"))
    tgt = next(e for e in legend if e["target"])
    target = dict(width=tgt["width"], lab=np.array(tgt["colour_lab"]), dash=tgt["dash"], gap=tgt["gap"])
    spots = json.load(open(SPOTS))
    extended = json.load(open(OUT / "text_extended.json"))
    grid = load_masks()["grid"]

    # pass 1
    s1 = State(rgb, ink, paper)
    bounds, redo = stages.text_bounds(s1.ink, spots, extended)
    text1 = [r["poly"] for r in spots] + [c for lb in extended for c in lb["cells"]]
    s1.remove("text", stages.polys_mask(ink.shape, text1))
    stages.boundary_stage(s1, gt)
    stages.river_stage(s1, gt)
    print(f"pass 1 done; {sum(redo)} of {len(bounds)} text bounds grew past their detection  {time.time() - t0:.0f}s", flush=True)

    # pass 2: re-detect text inside the grown bounds, on the borderless map
    img2 = rgb.copy()
    geo = s1.by["boundary"] | s1.by["river"]
    img2[geo] = paper[geo]
    grown = [bounds[k] for k in range(len(bounds)) if redo[k]]
    found = stages.redetect(img2, grown, "p2") if grown else []
    print(f"pass 2: {len(found)} readings re-detected inside {len(grown)} bounds  {time.time() - t0:.0f}s", flush=True)

    # final
    s = State(rgb, ink, paper)
    n = s.remove("text", stages.polys_mask(ink.shape, text1 + [f["poly"] for f in found]))
    print(f"text: removed {n} px", flush=True)
    b = stages.boundary_stage(s, gt)
    print(f"boundaries: style {b['style_width']} px wide, Lab {b['style_lab']}; {b['accepted']}/{b['windows']} windows; removed {b['removed_px']} px", flush=True)
    r = stages.river_stage(s, gt)
    print(f"rivers: {r['rivers']}; removed {r['removed_px']} px", flush=True)
    lg = stages.legend_stage(s, legend, ink)
    for k, v in lg.items():
        print(f"legend {k}: {v}", flush=True)
    g = stages.grid_stage(s, grid)
    print(f"grid: removed {g['removed_px']} px; remaining ink {s.ink.sum()} px ({s.ink.sum() / ink.sum():.0%} of the original)  {time.time() - t0:.0f}s", flush=True)

    save_rgb(OUT / "explain_remains.tif", s.rgb, gt, proj)
    save_classes(OUT / "explain_removed.tif", s, gt, proj)

    paths = cr.classify(s.ink, rgb, ink, s.removed, target)
    feats = [(p["points"], {k: p[k] for k in ("n", "llr_target", "llr_solid", "cls", "dash", "sw", "removed_gap")}) for p in paths]
    write_lines(OUT / "explain_paths.gpkg", "paths", gt, feats,
                dict(n=ogr.OFTInteger, llr_target=ogr.OFTReal, llr_solid=ogr.OFTReal, cls=ogr.OFTString, dash=ogr.OFTReal,
                     sw=ogr.OFTReal, removed_gap=ogr.OFTReal))
    cls = [p["cls"] for p in paths]
    print(f"paths: {len(paths)} ({ {c: cls.count(c) for c in set(cls)} })  {time.time() - t0:.0f}s")


LOG = OUT / "explain.log"


def say(msg, t0=None):
    """Progress to the log file as well as stdout: a long run inside QGIS returns stdout only at the end."""
    line = msg if t0 is None else f"{msg}  [{time.time() - t0:.0f}s]"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + line + "\n")


def classify_only():
    """Re-run the classification from the saved removal raster (minutes, not the full pipeline).
    Symbols are matched again first (cheap, and their threshold is still being set)."""
    t0 = time.time()
    LOG.write_text("")
    say("classify-only: loading sheet", t0)
    ds = gdal.Open(SHEET)
    gt = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0))
    ink, _ = ink_mask(rgb, arr[3] if arr.shape[0] > 3 else None)
    removed = gdal.Open(str(OUT / "explain_removed.tif")).ReadAsArray() > 0
    say("ink and paper", t0)
    s = State(rgb, ink, paper_background(rgb, ink))
    s.remove("earlier", removed & ink, grow=0)
    legend = json.load(open(OUT / "legend.json"))
    cache = OUT / "after_symbols.npz"
    if "--reuse-symbols" in sys.argv and cache.exists():
        z = np.load(cache)
        s.ink, s.removed = z["ink"], z["removed"]
        say("legend symbols: reused saved state", t0)
    else:
        say("matching legend symbols", t0)
        for k, v in stages.legend_stage(s, legend, ink, kinds=("symbol",), log=lambda m: say(m, t0)).items():
            say(f"legend {k}: {v}", t0)
        np.savez_compressed(cache, ink=s.ink, removed=s.removed)
    tgt = next(e for e in legend if e["target"])
    target = dict(width=tgt["width"], lab=np.array(tgt["colour_lab"]), dash=tgt["dash"], gap=tgt["gap"])
    paths = cr.classify(s.ink, rgb, ink, s.removed, target, log=lambda m: say(m, t0))
    feats = [(p["points"], {k: p[k] for k in ("n", "llr_target", "llr_solid", "cls", "dash", "sw", "removed_gap")}) for p in paths]
    write_lines(OUT / "explain_paths.gpkg", "paths", gt, feats,
                dict(n=ogr.OFTInteger, llr_target=ogr.OFTReal, llr_solid=ogr.OFTReal, cls=ogr.OFTString, dash=ogr.OFTReal,
                     sw=ogr.OFTReal, removed_gap=ogr.OFTReal))
    cls = [p["cls"] for p in paths]
    say(f"paths: {len(paths)} ({ {c: cls.count(c) for c in set(cls)} })", t0)


if __name__ == "__main__":
    classify_only() if "--classify-only" in sys.argv else main()
