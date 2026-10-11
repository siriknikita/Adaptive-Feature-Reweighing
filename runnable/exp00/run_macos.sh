#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Full Apple Silicon / MPS run
python experiment.py --device mps --epochs 18 --shots 8 --out results

echo
echo "Done. Send me results/summary.json and results/metrics.csv."
