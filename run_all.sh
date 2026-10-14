#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 src/prepare.py
python3 src/sanity_checks.py
python3 src/compare_models.py
python3 src/threshold_check.py
python3 src/train_final.py
python3 -m pytest -q