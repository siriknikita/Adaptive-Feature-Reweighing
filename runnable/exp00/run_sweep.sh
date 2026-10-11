#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

for shots in 1 2 4 8 16; do
  python experiment.py --device mps --shots "$shots" --out "results_shots_${shots}"
done
