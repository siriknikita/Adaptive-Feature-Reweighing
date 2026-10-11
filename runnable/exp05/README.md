
# Experiment #5 — Selective Fast-Path Activation

Experiment #4 established the central trade-off:

- a fully independent fast path recovered the held-out regime well;
- applying that fast policy globally damaged clean-distribution behavior;
- a globally mixed slow/fast policy preserved clean behavior but recovered less.

Experiment #5 asks:

> Can the model apply the fast policy **only when the current sample looks like the newly observed regime**?

The goal is to approach the recovery of `fast_only` while retaining the clean accuracy of the slow model.

---

## Controlled world

The evidence families remain:

- **A_eye** — dominant but fragile;
- **B_lower_geometry** — weaker but stable;
- **C_chroma** — weaker but stable.

Synthetic glasses are still completely held out from slow-router training.

All methods within a seed share the same:

- expert bank;
- global fusion;
- slow router;
- support set;
- query set;
- adapted fast policy.

This makes Experiment #5 a paired test of **activation policy**, not a comparison between separately trained models.

---

# Core idea

The slow route is

\[
p_s(x)=\operatorname{softmax}(g_s(x)).
\]

A fresh fast policy is adapted on the K labeled glasses support examples:

\[
p_f=\operatorname{softmax}(a).
\]

The final route becomes

\[
w(x)
=
(1-\alpha(x))p_s(x)
+
\alpha(x)p_f.
\]

Experiment #4 used a single global \(\alpha\).

Experiment #5 makes \(\alpha\) **context dependent**.

---

# Methods

## `slow_sgd`

Directly fine-tune the slow router.

Baseline from the earlier experiments.

## `fast_only`

Use the support-adapted fast policy everywhere.

This is the plasticity-heavy reference.

## `global_mixture`

Learn one scalar \(\alpha\) from the support set.

This reproduces the stability-heavy behavior from Experiment #4.

## `prototype_switch`

No learned neural switch.

The frozen slow router produces a context embedding \(\phi(x)\).

We compute:

- a prototype for the new glasses support regime;
- a prototype for a small stored clean reference set.

Then:

\[
s(x)
=
\cos(\phi(x),\mu_{\text{support}})
-
\cos(\phi(x),\mu_{\text{clean}}).
\]

A threshold is calibrated **only from support + clean-reference samples**, never from query labels.

The fast path activates when the current sample is more support-like than clean-like.

## `memory_switch`

A tiny linear context classifier is fitted on frozen context embeddings:

- glasses support: environment label 1;
- clean memory: environment label 0.

It does not learn the fake/real class.

Its sole job is to estimate:

\[
\alpha(x)=P(\text{new regime}\mid \phi(x)).
\]

The class decision still comes from the expert bank plus slow/fast routing.

## `environment_oracle`

Diagnostic ceiling only.

It uses the fast route for glasses and slow route for clean samples because the experiment itself knows the corruption label.

This is not a deployable method. It shows the best possible stability-plasticity combination if regime detection were perfect.

---

# Why the clean memory is legitimate

The context switch never receives clean **test** samples during adaptation.

It receives a tiny balanced reference memory drawn from the existing clean training distribution.

That is analogous to retaining a small representation of the model's original operating regime.

Default:

```text
128 clean reference samples
```

The glasses query set remains untouched.

---

# New measurements

For each K and method:

- glasses query accuracy;
- clean test accuracy after adaptation;
- harmonic stability-plasticity score;
- minimum of clean/glasses accuracy;
- mean \(\alpha\) on glasses;
- mean \(\alpha\) on clean;
- activation gap;
- new-regime true activation rate;
- clean false activation rate;
- final feature weights on glasses and clean;
- A release and B/C recovery.

The harmonic score is

\[
H
=
\frac{2A_{\text{glasses}}A_{\text{clean}}}
{A_{\text{glasses}}+A_{\text{clean}}}.
\]

It prevents a method from looking good merely by maximizing one side of the trade-off.

---

# Recommended M4 Pro run

The project is `uv`-native and uses MPS.

```bash
uv sync

uv run python run_experiment.py \
  --device mps \
  --seeds 11 22 33 44 55 \
  --output results
```

A small smoke test:

```bash
uv sync

uv run python run_experiment.py \
  --device mps \
  --seeds 42 \
  --expert-epochs 2 \
  --global-epochs 2 \
  --meta-episodes 60 \
  --train-n 1000 \
  --val-n 300 \
  --test-n 400 \
  --k-shots 4 8 \
  --fast-steps 20 \
  --context-steps 80 \
  --output smoke_results
```

---

# What would be a strong result?

Experiment #4 gave us approximately this qualitative frontier:

```text
fast_only:
  glasses high
  clean much lower

global mixture:
  glasses moderate
  clean ~perfect
```

The Experiment #5 target is:

```text
prototype_switch / memory_switch:
  glasses close to fast_only
  clean close to slow model
```

with:

```text
alpha_glasses >> alpha_clean
```

For example, the pattern we want is conceptually:

```text
glasses accuracy: 0.69
clean accuracy:   0.98
alpha glasses:    0.90
alpha clean:      0.05
```

That would show that the system can preserve an independent recoverable path **and invoke it selectively**.

---

# Output

```text
results/
├── results.csv
├── results.json
├── summary.txt
├── per_method_summary.csv
└── checkpoint_seed_<seed>.pt
```

Send the full results directory back after the run.
