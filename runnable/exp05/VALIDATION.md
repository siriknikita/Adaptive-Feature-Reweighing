# Validation

- Environment: Linux CPU, Python 3.12.14, `uv`, torch 2.14.1.
- Repairs: added complete run/smoke entrypoints; scientific code unchanged.
- Smoke command: `./smoke_test.sh`.
- Result: passed across slow, fast, global-mixture, prototype-switch, memory-switch, and oracle paths.
- Scope: validates routing/switch plumbing and output contracts, not historical performance.

