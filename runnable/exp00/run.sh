#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv sync
uv run python experiment.py --device auto --epochs 18 --shots 8 --out results

