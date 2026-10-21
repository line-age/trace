#!/usr/bin/env bash
# Build the text spotter in $TRACE_WORK/spotter: a Python 3.11 venv (uv), detectron2 and
# MapTextPipeline from source, the Rumsey fine-tuned weights, and MapReader's runner file
# (spot.py reads its 147-character vocabulary from it rather than retyping it).
#
# Keep it out of QGIS's Python: torch and detectron2 do not belong there. CPU is fine
# (1-4 s per 1024 px tile).
#
# Licences: MapTextPipeline is built on AdelaiDet ("academic use"; contact the authors for
# commercial use) and the weights state no licence. Settle that before production use.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${TRACE_WORK:-$HERE/../../work}"
mkdir -p "$WORK/spotter/src"
cd "$WORK/spotter"

[ -d venv ] || uv venv -q -p 3.11 venv
. venv/bin/activate
uv pip install -q numpy==1.26.4 torch torchvision wheel setuptools ninja pillow "huggingface-hub[cli]>=0.34,<1.0" \
  opencv-python-headless shapely pyyaml timm==0.9.16 scipy editdistance rapidfuzz

cd src
[ -d detectron2 ] || git clone -q --depth 1 https://github.com/facebookresearch/detectron2.git
[ -d MapTextPipeline ] || git clone -q --depth 1 https://github.com/maps-as-data/MapTextPipeline.git
cd ..
uv pip install --no-build-isolation -q ./src/detectron2
uv pip install --no-build-isolation -q ./src/MapTextPipeline

[ -s rumsey-finetune.pth ] || hf download rwood-97/MapTextPipeline_rumsey rumsey-finetune.pth --local-dir .
[ -s maptext_runner_ref.py ] || curl -sSfL -o maptext_runner_ref.py \
  https://raw.githubusercontent.com/maps-as-data/MapReader/main/mapreader/spot_text/maptext_runner.py
echo "spotter ready: $WORK/spotter/venv/bin/python"
