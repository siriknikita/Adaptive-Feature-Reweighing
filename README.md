# Recoverable Feature Routing — Reconstructed Research Repository

This repository is a transparent reconstruction of the adaptive feature re-weighting experiment lineage created on 2026-10-04 and archived on 2026-10-11.

The Git graph is intentionally organized as a sequence of research pull requests. Each experiment branch contains proposal, implementation, reproducibility, results, and analysis commits, followed by a dated non-fast-forward merge and annotated release tag.

Git commit, merge, and tag dates represent the best-supported historical experimental times. GitHub pull-request server dates cannot be backdated; portable reconstructed PR records therefore live in `.github/reconstructed-pull-requests/` and distinguish historical time from reconstruction time.

## Read first

- `JOURNAL.md`: complete hypothesis-to-decision narrative.
- `EXPERIMENT_STATUS.csv`: compact status and headline results.
- `PROVENANCE.md`: original-file mapping and dating evidence.
- `historical/`: preserved source and result artifacts.
- `runnable/`: complete `uv`-based snapshots with tested entrypoints.
- `.github/reconstructed-pull-requests/`: portable PR descriptions.

## Run an experiment

```bash
cd runnable/exp05
./smoke_test.sh
./run.sh
```

Experiment 6 is a real-data/T4 bridge. Its smoke test validates the environment, source, and CLI without downloading or training on the dataset.

## Inspect the evolution

```bash
git log --graph --all --decorate --date=iso --pretty=fuller
git branch --all
git tag --list
```

## Publish reconstructed PRs

`scripts/publish_reconstructed_prs.sh` pushes preserved branches and opens present-day GitHub PR records. It never claims GitHub's displayed timestamps are historical. Supply an existing empty GitHub repository and an authenticated `gh` session.

