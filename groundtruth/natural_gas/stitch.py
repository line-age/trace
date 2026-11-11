# /// script
# requires-python = ">=3.11"
# dependencies = ["geopandas>=1.0", "pyogrio>=0.10", "pyarrow>=17", "shapely>=2.0", "pyyaml", "python-ulid>=3"]
# ///
"""Stitch the research partner's natural gas trace files into one GeoPackage.

Input is the "Natural Gas Traces" folder: a 1929 basemap plus, per period A_B, files of
lines added (`add`) and removed (`sub`) between the year-A and year-B sheets. Output layers:

- `pipelines`: the basemap and every add, dated by the map-date rule. Where a sub
  plausibly removes a feature (or part of one), that feature gets a *proposed* end.
- `removals`: every sub line as drawn, with how well it matched.
- `sheets`: the sheet date used for each boundary year, and the evidence for it.

Dates follow the Legend profile (controlplane I-0003, Legend 0007/0011): EDTF as
authored in `existed_from`/`existed_to`, and derived DATE columns `time_start`/`time_end`
for the QGIS Temporal Controller. Start dates collapse to the first day of their EDTF
period, end dates to the last.

Matching a sub to the feature it removes is a judgement, not a fact (I-0004), so every
end it sets is marked `end_status = proposed`.

The sheet dates come from a YAML file kept with the traces, not in this repo, because it
names the research partner's sheets: one entry per boundary year, with `date` (EDTF),
`files` (the sheet scans) and `evidence` (what the sheet says, so the date can be checked).

    uv run groundtruth/natural_gas/stitch.py "../maps/Natural Gas Traces" \
        ../maps/natural_gas_sheets.yaml work/groundtruth/natural_gas.gpkg
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyogrio
import shapely
import yaml
from shapely.geometry import MultiLineString
from ulid import ULID

NAME = re.compile(r"ngpipe_(add|sub|base)_(\d{4})_(\d{4}|NA)_([a-z]+)_([a-z]+?)(_s)?\.gpkg$")
WORK_CRS = 5070  # CONUS Albers, metres

# Sub matching. A feature is a candidate when at least S_COV_MIN of the sub's length runs
# within TOL_M of it; it ends whole when WHOLE_FRAC of its own length is inside the sub's
# buffer, otherwise it is split at the buffer edge.
TOL_M = 1000.0
S_COV_MIN = 0.3
WHOLE_FRAC = 0.9
MIN_PIECE_M = 500.0


def first_day(edtf: str) -> dt.date:
    parts = [int(p) for p in edtf.split("-")]
    return dt.date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)


def last_day(edtf: str) -> dt.date:
    parts = [int(p) for p in edtf.split("-")]
    if len(parts) == 3:
        return dt.date(*parts)
    month = parts[1] if len(parts) > 1 else 12
    return dt.date(parts[0], month, calendar.monthrange(parts[0], month)[1])


def new_id() -> str:
    return f"lgd:edge:{ULID()}"


def as_multi(geom):
    """Every output geometry is a MultiLineString, whatever the input file used."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return MultiLineString([geom])
    if geom.geom_type == "MultiLineString":
        return geom
    lines = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("LineString", "MultiLineString")]
    merged = shapely.line_merge(shapely.union_all(lines)) if lines else None
    return as_multi(merged) if merged is not None and not merged.is_empty else None


def read_traces(root: Path) -> list[dict]:
    """One row per traced feature. Rows with no geometry (there are some) are skipped and
    reported, since there is nothing to stitch."""
    rows, skipped = [], 0
    for path in sorted(root.rglob("ngpipe_*.gpkg")):
        m = NAME.search(path.name)
        if not m:
            sys.exit(f"unrecognised trace filename: {path}")
        kind, y0, y1, tracer, region, _ = m.groups()
        for layer, _ in pyogrio.list_layers(path):
            g = gpd.read_file(path, layer=layer, fid_as_index=True)
            for fid, geom in zip(g.index, g.geometry):
                geom = as_multi(geom)
                if geom is None:
                    skipped += 1
                    print(f"skipped: {path.relative_to(root)} fid {int(fid)} has no geometry")
                    continue
                rows.append(dict(kind=kind, y0=int(y0), y1=int(y0) if y1 == "NA" else int(y1),
                                 tracer=tracer, region=region, source=str(path.relative_to(root)),
                                 source_layer=layer, source_fid=int(fid), geometry=geom))
    if skipped:
        print(f"skipped {skipped} features with no geometry")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path, help='the "Natural Gas Traces" folder')
    ap.add_argument("sheets", type=Path, help="sheet dates, one entry per boundary year")
    ap.add_argument("out", type=Path, help="GeoPackage to write (replaced if it exists)")
    args = ap.parse_args()

    sheets = {int(k): v for k, v in yaml.safe_load(args.sheets.read_text()).items()}
    traces = read_traces(args.traces)
    missing = {y for t in traces for y in (t["y0"], t["y1"])} - sheets.keys()
    if missing:
        sys.exit(f"{args.sheets} has no entry for {sorted(missing)}")

    def sheet_files(year):
        return "; ".join(sheets[year]["files"])

    # --- pipelines: basemap and adds -------------------------------------------------
    pipes: list[dict] = []
    for t in traces:
        if t["kind"] == "sub":
            continue
        seen = sheets[t["y1"]]["date"]  # first sheet the line appears on
        pipes.append(dict(
            id=new_id(), carrier="gas", substance="natural", kind=t["kind"],
            existed_from=f"../{seen}", existed_to=None,
            time_start=first_day(seen), time_end=None,
            date_precision="bound", end_precision=None, end_status=None, ended_by=None, split_from=None,
            period=f"{t['y0']}" if t["kind"] == "base" else f"{t['y0']}_{t['y1']}",
            period_end_year=t["y1"], tracer=t["tracer"], region=t["region"],
            sheet=sheet_files(t["y1"]),
            sheet_before=None if t["kind"] == "base" else sheet_files(t["y0"]),
            source=t["source"], source_layer=t["source_layer"], source_fid=t["source_fid"],
            geometry=t["geometry"],
        ))

    to_work = lambda g: gpd.GeoSeries([g], crs=4326).to_crs(WORK_CRS).iloc[0]
    to_wgs = lambda g: gpd.GeoSeries([g], crs=WORK_CRS).to_crs(4326).iloc[0]
    for p in pipes:
        p["_w"] = to_work(p["geometry"])

    # --- removals: match each sub to the alive feature(s) it most plausibly ends -------
    removals = []
    subs = sorted((t for t in traces if t["kind"] == "sub"), key=lambda t: (t["y0"], t["source"], t["source_fid"]))
    for t in subs:
        a, b = sheets[t["y0"]]["date"], sheets[t["y1"]]["date"]
        rid = new_id()
        s = to_work(t["geometry"])
        s_buf = s.buffer(TOL_M)
        alive = [p for p in pipes if p["time_end"] is None and p["period_end_year"] <= t["y0"]]
        hits = []
        for p in alive:
            if not p["_w"].intersects(s_buf):
                continue
            s_cov = s.intersection(p["_w"].buffer(TOL_M)).length / s.length
            if s_cov >= S_COV_MIN:
                hits.append((p, s_cov))

        # Score before ending anything: a split replaces p["_w"] with the piece outside the sub.
        covered = (s.intersection(shapely.union_all([p["_w"].buffer(TOL_M) for p, _ in hits])).length / s.length
                   if hits else 0.0)

        ended = []
        for p, _ in hits:
            inside = p["_w"].intersection(s_buf)
            frac = inside.length / p["_w"].length
            end = dict(existed_to=f"{a}/{b}", time_end=last_day(b), end_precision="range",
                       end_status="proposed", ended_by=rid)
            outside = as_multi(p["_w"].difference(s_buf))
            if frac >= WHOLE_FRAC or outside is None or outside.length < MIN_PIECE_M:
                p.update(end)
                ended.append(p["id"])
                continue
            # Split: the part under the sub ends; the rest carries on unchanged.
            inside = as_multi(inside)
            if inside is None or inside.length < MIN_PIECE_M:
                continue
            orig = p["id"]
            gone = {**p, **end, "id": new_id(), "split_from": orig, "_w": inside, "geometry": to_wgs(inside)}
            p.update(id=new_id(), split_from=orig, _w=outside, geometry=to_wgs(outside))
            pipes.append(gone)
            ended.append(gone["id"])

        removals.append(dict(
            id=rid, carrier="gas", substance="natural",
            existed_from=f"../{a}", existed_to=f"{a}/{b}",
            period=f"{t['y0']}_{t['y1']}", tracer=t["tracer"], region=t["region"],
            match_status="matched" if covered >= 0.8 else "partial" if covered > 0 else "unmatched",
            match_score=round(covered, 3), ends="; ".join(ended) or None,
            sheet_before=sheet_files(t["y0"]), sheet=sheet_files(t["y1"]),
            source=t["source"], source_layer=t["source_layer"], source_fid=t["source_fid"],
            geometry=t["geometry"],
        ))

    # --- write -----------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)

    def frame(rows, date_cols=()):
        df = gpd.GeoDataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
                              geometry="geometry", crs=4326)
        for c in date_cols:
            df[c] = pd.array(df[c].tolist(), dtype=pd.ArrowDtype(pa.date32()))
        return df

    pipe_df = frame(pipes, ("time_start", "time_end")).sort_values(["time_start", "period", "tracer", "source_fid"])
    pyogrio.write_dataframe(pipe_df, args.out, layer="pipelines", driver="GPKG", use_arrow=True,
                            geometry_type="MultiLineString")
    pyogrio.write_dataframe(frame(removals), args.out, layer="removals", driver="GPKG", use_arrow=True,
                            geometry_type="MultiLineString", append=True)
    sheet_df = pd.DataFrame([dict(year=y, date=v["date"], files="; ".join(v["files"]), evidence=v["evidence"])
                             for y, v in sorted(sheets.items())])
    pyogrio.write_dataframe(sheet_df, args.out, layer="sheets", driver="GPKG", append=True)

    rem = pd.DataFrame(removals)
    print(f"pipelines: {len(pipe_df)} features "
          f"({(pipe_df.kind == 'base').sum()} base, {(pipe_df.kind == 'add').sum()} add, "
          f"{pipe_df.split_from.notna().sum()} split pieces, {pipe_df.end_status.notna().sum()} with a proposed end)")
    print(f"removals: {len(rem)} subs; " + ", ".join(f"{k} {v}" for k, v in rem.match_status.value_counts().items()))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
