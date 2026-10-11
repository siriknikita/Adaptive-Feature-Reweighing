#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv sync
uv run python -m py_compile exp6_real_recoverability.py
uv run python exp6_real_recoverability.py --help >/dev/null

