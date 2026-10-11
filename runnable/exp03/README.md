
# Experiment #3 — Recoverability vs Routing Collapse

This experiment follows directly from Experiment #2.

Experiment #2 showed two qualitatively different regimes:

- some seeds **recovered** after the dominant feature failed;
- some seeds became almost perfectly locked to the dominant expert and **could not escape** even though useful backup experts still existed.

So Experiment #3 asks a narrower question:

> Which routing mechanisms preserve access to low-priority features after the dominant feature becomes unreliable?

This is an **ablation study**. The underlying experts and synthetic world are held fixed; only the routing mechanism changes.

---

## Core hypothesis

A router can retain all experts structurally while still making them functionally inaccessible.

For standard softmax routing,

\[
w_i = \frac{e^{z_i}}{\sum_j e^{z_j}},
\]

if one logit becomes much larger than the others, then:

\[
w_A \rightarrow 1, \qquad w_{B,C} \rightarrow 0.
\]

At that point, gradients toward neglected experts can become too weak for fast adaptation.

Experiment #3 measures whether anti-lock-in mechanisms improve **recoverability** without destroying clean-distribution performance.

---

## Routing methods

The experiment compares:

1. **softmax**
   - ordinary learned routing;
   - serves as the failure-prone baseline.

2. **temperature_softmax**
   - uses a higher temperature:
   \[
   w_i = \operatorname{softmax}(z_i/T), \quad T>1
   \]
   - reduces saturation.

3. **epsilon_floor**
   - guarantees each expert a minimum probability mass:
   \[
   w_i = (1-E\epsilon)\operatorname{softmax}(z)_i+\epsilon
   \]
   - preserves a small amount of access to every expert.

4. **logit_clip**
   - clamps router logits to a bounded range before softmax;
   - prevents arbitrarily extreme routing ratios.

5. **residual_router**
   - starts from global fusion weights and only learns a bounded residual adjustment;
   - prevents the adaptive router from completely overwriting the global prior.

6. **uniform**
   - equal weights; diagnostic only.

7. **global**
   - one fixed learned weight vector; diagnostic baseline.

8. **oracle**
   - directly optimizes fusion logits on the held-out support set;
   - diagnostic upper bound, not a proposed deployment mechanism.

---

## Synthetic world

Same basic setup as Experiment #2:

- **A_eye** — dominant but fragile eye-band cue;
- **B_lower_geometry** — weaker but stable lower-face geometry;
- **C_chroma** — weaker but stable color cue.

The final held-out **glasses** corruption destroys A.

The router never sees glasses during meta-training.

---

## Main metrics

Experiment #3 writes ordinary classification metrics plus explicit routing-plasticity metrics.

### 1. Accuracy recovery

\[
\Delta Acc_K = Acc_K - Acc_{\text{global, glasses}}
\]

### 2. Dominant-cue release

\[
Release_A(K) = w_A^{(0)} - w_A^{(K)}
\]

Positive means adaptation moved away from the failed dominant cue.

### 3. Backup-cue recovery

\[
Recovery_{BC}(K)
=
(w_B^{(K)} + w_C^{(K)})
-
(w_B^{(0)} + w_C^{(0)})
\]

### 4. Routing entropy

\[
H(w) = -\sum_i w_i \log w_i
\]

Very low entropy indicates near one-hot routing.

### 5. Saturation rate

Fraction of evaluation samples for which:

\[
\max_i w_i > 0.99
\]

This is the direct operational signal of routing collapse.

---

## Recommended run on your M4 Pro

This project uses **uv** and supports **MPS**.

```bash
uv sync
uv run python run_experiment.py \
  --device mps \
  --seeds 11 22 33 44 55 \
  --output results
```

A single-seed smoke test:

```bash
uv sync
uv run python run_experiment.py \
  --device mps \
  --seeds 42 \
  --meta-episodes 120 \
  --output smoke_results
```

---

## Output

```text
results/
├── results.csv
├── results.json
├── summary.txt
├── per_method_summary.csv
└── checkpoint_seed_<seed>.pt
```

Send me the whole `results/` directory afterward.

---

## What would count as a strong result?

A useful anti-collapse mechanism should satisfy all three:

1. clean performance stays near the baseline;
2. glasses zero-shot may still fail;
3. after only a few support examples, the router releases A and shifts toward B/C.

The most interesting pattern would be:

```text
softmax:
  clean good
  glasses 0.50
  after 8-shot still ~0.50
  saturation ~1.0

epsilon_floor / residual / temperature:
  clean good
  glasses 0.50
  after 8-shot >0.65
  saturation much lower
  w_A drops, w_B/C rise
```

If that appears consistently across seeds, we have evidence that the important issue is not merely feature retention but **feature recoverability under routing saturation**.
