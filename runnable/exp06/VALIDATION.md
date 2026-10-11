# Validation

- Environment: Linux CPU, Python 3.12.14, `uv`, torch 2.14.1, datasets 5.1.0.
- Repairs: added explicit full and structural-smoke entrypoints; scientific code unchanged.
- Smoke command: `./smoke_test.sh`.
- Result: dependency resolution, bytecode compilation, and CLI construction passed.
- Unverified: dataset download, T4/CUDA execution, expert training, cache construction, routing runs, and final result packaging.
- Scientific status: code-only; no real-data result is claimed.

