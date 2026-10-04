
# Experiment #2 — Rediscovering Neglected Evidence

This experiment is deliberately constructed to fix the failure mode we saw in Experiment #1.

## Question

Can a model learn to **rapidly reweight feature families** when the feature that is usually best becomes unreliable on a novel input regime?

The key distinction is:

- ordinary global learning asks, “Which evidence is best on average?”
- this experiment asks, “Which evidence is trustworthy **for this sample/regime**?”

No larger teacher model is used.

## Synthetic world

Every image contains three deliberately separated cues:

- **A — fragile dominant cue**: a high-signal marker in the eye band.
- **B — stable geometry cue**: a weaker lower-face shape cue.
- **C — stable chroma cue**: a weaker global color-balance cue.

On clean training data, A is much easier than B/C, so the global fusion should prefer A.

During meta-training, the router sees several *other* perturbation families that sometimes damage one source of evidence. This teaches the router a general strategy: do not trust the same evidence under every regime.

The final test perturbation is **held-out synthetic glasses**. It is never used during meta-training. Glasses deliberately destroy the eye-band cue A while leaving B and C mostly available.

## Models compared

1. **Uniform fusion** — equal weights.
2. **Global fusion** — one learned weight vector for all samples.
3. **Adaptive zero-shot** — a learned per-sample gate, with no glasses-specific adaptation.
4. **Adaptive few-shot** — the same gate after K labeled glasses examples.
5. **Oracle-per-condition** — a diagnostic upper bound: learn only three fusion logits directly on the glasses support set. This is *not* the proposed method; it tells us whether useful alternative evidence exists at all.

The experts themselves are frozen before the adaptation experiment. Only routing/fusion changes.

## Anti-collapse measures

Experiment #1 collapsed almost completely to the edge expert. Here we use:

- explicit feature-isolated experts;
- expert dropout during router meta-training;
- a small entropy floor on router weights;
- held-out perturbation testing;
- a direct diagnostic of each expert's standalone accuracy.

The entropy term is intentionally weak. It prevents dead routing pathways but does not force uniform weights.

## Run on your M4 Pro with uv + MPS

```bash
uv sync
uv run python run_experiment.py --device mps --output results
```

Or let it select MPS automatically:

```bash
uv run python run_experiment.py --device auto --output results
```

For a stronger run across five seeds:

```bash
uv run python run_experiment.py \
  --device mps \
  --seeds 11 22 33 44 55 \
  --output results_5seed
```

## What success looks like

A useful result should have this qualitative structure:

```text
clean:
  expert A >> expert B ~= expert C
  global fusion performs very well

glasses:
  expert A collapses
  B/C remain above chance
  fixed global fusion drops substantially

few-shot:
  router shifts weight away from A
  accuracy recovers as K increases
```

We do **not** require monotonic improvement at every K. Few-shot optimization is noisy.

The strongest signal is:

```text
global_glasses_accuracy << adaptive_few_shot_glasses_accuracy
```

together with a clear routing shift:

```text
weight_A decreases
weight_B and/or weight_C increase
```

## Output

The script writes:

- `results.json`
- `results.csv`
- `summary.txt`
- `checkpoint_seed_<seed>.pt`

Send me the whole results directory afterward.

## Important interpretation rule

This is still a mechanism test, not a deepfake benchmark.

If Experiment #2 succeeds, the next step is a real deepfake experiment with:

- identity-disjoint splits;
- generator-disjoint evaluation;
- naturally occurring glasses/sunglasses;
- realistic generated eyewear edits;
- calibration metrics;
- multiple seeds;
- comparison against ordinary fine-tuning and mixture-of-experts baselines.
