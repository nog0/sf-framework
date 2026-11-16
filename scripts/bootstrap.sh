#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run in an activated sf-framework environment. No keys or login state are written here.
python -m pip install 'torch==2.2.2+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-binary textworld -r requirements/benchmarks.txt -e '.[api,data,dev]'
python -m spacy download en_core_web_sm
sf doctor
printf '\nNext: sf setup alfworld; sf setup webshop\n'
