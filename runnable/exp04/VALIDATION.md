# Validation

- Environment: Linux CPU, Python 3.12.14, `uv`, torch 2.14.1.
- Repairs: added complete run/smoke entrypoints; scientific code unchanged.
- Smoke command: `./smoke_test.sh`.
- Result: passed across all slow/fast adaptation paths and produced the expected result files.
- Scope: validates paired-path execution and diagnostics, not historical accuracy.

