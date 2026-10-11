# Validation

- Environment: Linux CPU, Python 3.12.14, `uv`, torch 2.14.1.
- Repairs: converted dependency execution to `uv`; added full and smoke entrypoints; scientific code unchanged.
- Smoke command: `./smoke_test.sh`.
- Result: passed; produced nonempty `results.json` and `results.csv`.
- Scope: the deliberately tiny smoke dataset tests plumbing only.

