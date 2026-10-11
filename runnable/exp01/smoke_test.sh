#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv sync
uv run python run_experiment.py --device cpu --synthetic-train 160 --synthetic-val 64 --synthetic-test 96 --expert-epochs 1 --global-epochs 1 --meta-episodes 2 --support-size 4 --query-size 8 --k-shots 2 --test-adapt-steps 1 --output smoke_results
test -s smoke_results/results.json
test -s smoke_results/results.csv

