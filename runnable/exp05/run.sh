#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv sync
uv run python run_experiment.py --device auto --seeds 11 22 33 44 55 --output results

