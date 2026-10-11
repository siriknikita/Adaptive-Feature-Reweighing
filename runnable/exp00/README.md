# Adaptive Feature Reweighting — Toy Experiment

This experiment tests the idea:

> A feature that is weak globally may become crucial for a specific unfamiliar input regime. A lightweight judge should reweight the model's evidence policy per input, and a few-shot recalibration step should quickly resurrect a normally dormant feature on an unseen transformation.

## What the synthetic world does

- 32 input features.
- Features `0..5` are generally strong and predictive.
- Features `20..23` are normally useless noise.
- Training includes three transformation families (`seen_A`, `seen_B`, `seen_C`).
- Each seen transformation:
  - corrupts two normally strong features;
  - makes one dormant feature suddenly highly predictive.
- Test uses a fourth transformation, `heldout_D`, that is **never present during training**.
  - It corrupts strong features `0` and `5`.
  - It makes dormant feature `23` highly predictive.

The question is whether the model learns the meta-pattern:

> "When familiar evidence becomes unreliable, some usually weak feature may become important."

rather than memorizing a fixed transform.

## Compared systems

1. `base_linear` — ordinary globally weighted classifier.
2. `plain_mlp` — higher-capacity control model, no explicit judge.
3. `adaptive_zero_shot` — base evidence weights + input-conditioned judge/gates.
4. `adaptive_fewshot` — same adaptive model, but on the unseen transform only a tiny 32-number gate-offset vector is recalibrated using a few labeled samples. The network itself stays frozen.

This fourth system is the closest toy analogue to "temporarily rethink the evidence policy for this new footprint."

## Run on your M4 Pro MacBook

From Terminal:

```bash
cd adaptive_feature_reweighting
./run_macos.sh
```

PyTorch will use Apple's Metal backend through `mps`.

For a faster smoke test first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python experiment.py --device mps --quick --epochs 6 --shots 8 --out results_quick
```

## What to send back

Please send these two files:

- `results/summary.json`
- `results/metrics.csv`

The most important quantities are:

- held-out accuracy of the four systems;
- `heldout_dormant_feature_23_mean_gate_before`;
- `heldout_dormant_feature_23_mean_gate_after`.

A particularly interesting outcome is:

1. base model performs poorly on `heldout_D`;
2. adaptive zero-shot is at least somewhat better;
3. few-shot adaptive improves further;
4. feature 23's gate rises after few-shot adaptation.

That combination would be direct evidence that the model can rapidly change which evidence it trusts for an unseen regime.

## Useful variations

Change support size:

```bash
python experiment.py --device mps --shots 1 --out results_1shot
python experiment.py --device mps --shots 2 --out results_2shot
python experiment.py --device mps --shots 4 --out results_4shot
python experiment.py --device mps --shots 8 --out results_8shot
python experiment.py --device mps --shots 16 --out results_16shot
```

Run several seeds:

```bash
python experiment.py --device mps --seed 1 --out results_seed1
python experiment.py --device mps --seed 2 --out results_seed2
python experiment.py --device mps --seed 3 --out results_seed3
python experiment.py --device mps --seed 4 --out results_seed4
python experiment.py --device mps --seed 5 --out results_seed5
```

Those runs tell us whether the effect is robust or just a lucky initialization.
