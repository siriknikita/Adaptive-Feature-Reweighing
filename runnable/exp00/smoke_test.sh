#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv sync
uv run python experiment.py --device cpu --quick --epochs 1 --shots 2 --out smoke_results
test -s smoke_results/summary.json
test -s smoke_results/metrics.csv

