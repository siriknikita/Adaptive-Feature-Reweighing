
# Adaptive Feature Reweighting — held-out “glasses” experiment

This is a deliberately small experiment for the idea we were discussing:

> A model may have feature families that are globally weak, yet become useful for a particular input. Instead of permanently pruning those features, learn a per-sample judge/router that can **rapidly reweight** the feature families when the current input violates the model’s usual strategy.

The important part is that the held-out perturbation **`glasses` is never shown during meta-training**.

The experiment asks whether the gate can learn *how to adapt*, rather than merely memorize “glasses => use expert X”.

## Architecture

There are four tiny experts:

1. `rgb` — ordinary RGB signal.
2. `edge` — Sobel edge magnitude.
3. `highpass` — local high-frequency residual.
4. `chroma` — opponent-color differences.

The main experts are trained first and then frozen.

A small gate/judge receives:

- a cheap encoding of the current image;
- each expert's confidence;
- each expert's class-margin;
- each expert's entropy.

It outputs per-sample weights:

\[
w(x)=\operatorname{softmax}(g(x))
\]

and the final prediction is:

\[
z(x)=\sum_i w_i(x)\,z_i(x)
\]

where \(z_i\) is expert \(i\)'s class-logit vector.

The gate is then meta-trained on:

- clean
- blur
- JPEG
- brightness reduction
- bottom occlusion

but **not glasses**.

At test time, the script creates a new glasses-like occlusion. It measures:

- uniform expert averaging;
- one globally learned weight vector;
- adaptive gate with zero glasses examples;
- adaptive gate after 2 / 4 / 8 / 16 / 32 labeled glasses examples.

This directly tests the “rethink the current feature policy” idea without requiring a larger teacher.

---

## Quick start on your M4 Pro MacBook

Open Terminal in this folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_experiment.py --device mps --output results
```

If your PyTorch build does not expose MPS for some reason:

```bash
python run_experiment.py --device cpu --output results_cpu
```

The synthetic experiment is the recommended first run because it requires no dataset download.

Expected runtime is intentionally modest. The exact time depends on your PyTorch version, but this is sized as a laptop experiment rather than a full training run.

---

## What to send back to ChatGPT

Zip the `results/` folder or send these two files:

```text
results/results.json
results/results.csv
```

The key question is **not just whether accuracy rises**.

We want to inspect whether:

1. `global` performs worse on held-out glasses than on clean data;
2. `adaptive_zero_shot` changes the expert weights at all;
3. a tiny support set causes a useful weight shift;
4. accuracy improves monotonically or at least substantially with K;
5. the gate shifts weight toward feature families that remain informative under the held-out corruption.

A particularly interesting result would look conceptually like:

```text
clean:
RGB 0.45, edge 0.25, highpass 0.20, chroma 0.10

glasses before adaptation:
RGB 0.41, edge 0.27, highpass 0.22, chroma 0.10

glasses after 8-shot adaptation:
RGB 0.24, edge 0.18, highpass 0.25, chroma 0.33
```

The exact direction is not predetermined. What matters is whether the gate discovers a stable, useful shift.

---

## Using your own deepfake dataset

Use this directory structure:

```text
my_deepfake_data/
├── train/
│   ├── fake/
│   │   ├── ...
│   └── real/
│       ├── ...
├── val/
│   ├── fake/
│   │   ├── ...
│   └── real/
│       ├── ...
└── test/
    ├── fake/
    │   ├── ...
    └── real/
        ├── ...
```

Then run:

```bash
python run_experiment.py \
  --dataset folder \
  --data "/absolute/path/to/my_deepfake_data" \
  --device mps \
  --image-size 128 \
  --batch-size 48 \
  --expert-epochs 8 \
  --meta-episodes 300 \
  --output deepfake_results
```

The script applies the held-out glasses-like overlay itself, so your original test images should **not** already contain that synthetic overlay.

For a serious paper-quality experiment, do not treat this synthetic overlay as a final benchmark. It is an intervention test: it tells us whether the mechanism is worth pursuing. If it works, the next stage should use naturally occurring eyeglasses/sunglasses, identity-disjoint splits, generator-disjoint splits, and multiple realistic occlusion types.

---

## Controls already included

The experiment contains three important controls:

- **Uniform fusion:** no feature selection.
- **Global learned fusion:** the classical “these features are generally most useful” strategy.
- **Adaptive zero-shot:** gate trained to adapt across other perturbations but given no glasses examples.

The few-shot rows then isolate the value of fast adaptation.

---

## Strong falsification criteria

The idea is *not* supported just because the adaptive system has more parameters.

Treat the mechanism as unsuccessful if any of these happens consistently:

- global weighting performs as well as or better than few-shot adaptation;
- the gate changes weights but accuracy does not improve;
- adaptation helps glasses but badly damages clean examples;
- different random seeds produce contradictory routing patterns;
- gains disappear after controlling for total parameter count;
- the gate simply memorizes class label statistics in the support set.

If the first run is promising, repeat with at least five seeds:

```bash
for s in 11 22 33 44 55; do
  python run_experiment.py \
    --device mps \
    --seed "$s" \
    --output "results_seed_${s}"
done
```

That gives us enough to estimate whether the effect is real rather than one lucky run.
