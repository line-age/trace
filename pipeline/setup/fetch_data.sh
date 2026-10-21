#!/usr/bin/env bash
# Download everything the pipeline reads that is not in this repo, into $TRACE_WORK.
#
#   sheet.jpg     the 1937 O&GJ Illinois Basin map, full size, from the Indiana Historical
#                 Society's IIIF server (dc035/217). The image is marked "All Rights Reserved":
#                 fine for this research, never commit it or publish derived images.
#   data/ahcb/    Newberry Atlas of Historical County Boundaries, IL / IN / KY (CC0)
#   data/ne/      Natural Earth 10m river centrelines (public domain)
#   data/gaz/     a county-name list (state_and_county_fips_master.csv, kjhealy/fips-codes)
#
# Safe to re-run: existing files are kept.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${TRACE_WORK:-$HERE/../../work}"
mkdir -p "$WORK/data/ahcb" "$WORK/data/ne" "$WORK/data/gaz"
WORK="$(cd "$WORK" && pwd)"
echo "TRACE_WORK=$WORK"

if [ ! -s "$WORK/sheet.jpg" ]; then
  echo "sheet: Indiana Historical Society dc035/217 (8400 x 7246)"
  curl -sSfL -o "$WORK/sheet.jpg" "https://images.indianahistory.org/digital/iiif/dc035/217/full/full/0/default.jpg"
fi

cd "$WORK/data/ahcb"
for st in IL IN KY; do
  [ -d "$st" ] && continue
  for base in https://digital.newberry.org/ahcb/downloads/gis https://publications.newberry.org/ahcb/downloads/gis; do
    if curl -sSfL -o "$st.zip" "$base/${st}_AtlasHCB.zip" && file "$st.zip" | grep -q Zip; then
      mkdir -p "$st" && unzip -oq "$st.zip" -d "$st" && echo "ahcb $st ok" && break
    fi
  done
  [ -d "$st" ] || { echo "ahcb $st: download failed (try the Newberry site by hand)"; exit 1; }
done

cd "$WORK/data/ne"
for f in ne_10m_rivers_lake_centerlines ne_10m_rivers_north_america; do
  [ -d "$f" ] && continue
  curl -sSfL -m 300 -o "$f.zip" "https://naciscdn.org/naturalearth/10m/physical/$f.zip"
  mkdir -p "$f" && unzip -oq "$f.zip" -d "$f" && echo "ne $f ok"
done

cd "$WORK/data/gaz"
[ -s fips.csv ] || curl -sSfL -o fips.csv https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv
echo "done"
