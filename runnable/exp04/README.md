
# Experiment #4 — Independent Fast Adaptation Pathways

Experiment #3 showed an important failure mode:

> Keeping a feature present, or even forcing it to retain nonzero probability mass, does **not** guarantee that the model can recover it after routing has saturated.

The next question is therefore narrower:

> Can an **independent fast adaptation pathway** recover useful backup evidence even when the slow router is already overconfident?

This experiment is designed to isolate that question.

---

## The key change from Experiment #3

Experiment #3 trained several different routers. That made the comparison noisier than it needed to be.

Experiment #4 trains **one expert bank, one global fusion vector, and one slow router per seed**.

Every adaptation method then receives:

- the exact same experts;
- the exact same slow router;
- the exact same support examples;
- the exact same query examples;
- the exact same held-out glasses corruption.

So the only thing that changes is **how test-time adaptation gets access to the routing distribution**.

No Python `hash()` is used anywhere. The run is deterministic for a fixed seed.

---

# Synthetic evidence world

As before:

- **A_eye** — dominant, fragile eye-band cue.
- **B_lower_geometry** — weaker but stable lower-face cue.
- **C_chroma** — weaker but stable chromatic cue.

The slow model is trained without glasses.

At final evaluation, synthetic glasses destroy A while leaving B/C available.

---

# Adaptation mechanisms

## 1. `slow_sgd`

The Experiment #2-style baseline.

The entire slow gate is copied and directly fine-tuned on K glasses examples.

If the router has saturated at A, gradients through the softmax can become tiny.

---

## 2. `additive_fast`

Freeze the slow router and create new fast logits:

\[
a=[a_A,a_B,a_C], \qquad a_0=0.
\]

Then use:

\[
w=\operatorname{softmax}(z_{\text{slow}}+a).
\]

This is an important **negative control**.

It has fresh parameters, but adaptation still has to pass through the same saturated final softmax. If the saturation hypothesis is correct, fresh parameters alone should not solve the problem.

---

## 3. `mixture_fixed`

Use an independent fast distribution:

\[
p_{\text{slow}}=\operatorname{softmax}(z_{\text{slow}})
\]

\[
p_{\text{fast}}=\operatorname{softmax}(a)
\]

and after support examples arrive:

\[
w=(1-\alpha)p_{\text{slow}}+\alpha p_{\text{fast}}.
\]

`alpha` is fixed.

The crucial property is that:

\[
\frac{\partial p_{\text{fast}}}{\partial a}
\]

does **not** pass through the saturated slow softmax.

---

## 4. `mixture_learned`

Same independent fast distribution, but the mixture strength is itself learned:

\[
\alpha=\sigma(\beta).
\]

Both `a` and `beta` adapt on the support set.

This lets the system decide how strongly it should temporarily override the slow policy.

---

## 5. `fast_only`

Diagnostic control.

Ignore the slow router after support arrives and optimize only a fresh fast distribution:

\[
w=p_{\text{fast}}.
\]

This is not the intended deployment mechanism. It tells us how much recoverability is available from the support set if the saturated slow policy is completely bypassed.

---

## 6. `oracle`

Longer direct optimization of fusion logits on the support set.

This remains the diagnostic upper bound.

---

# New metric: routing conductance

Accuracy alone is not enough.

For a softmax distribution \(p\), the routing Jacobian is:

\[
J(p)=\operatorname{diag}(p)-pp^\top.
\]

We record its Frobenius norm:

\[
C=\lVert J\rVert_F.
\]

Interpretation:

- large \(C\): routing weights remain responsive to logit changes;
- tiny \(C\): routing is almost locally frozen.

For the independent mixture pathway:

\[
C_{\text{fast}}
=
\alpha
\left\|
J(p_{\text{fast}})
\right\|_F.
\]

The experiment also measures the **actual first-step gradient norm toward backup cues B/C** on the support set.

So Experiment #4 distinguishes:

1. feature availability;
2. nonzero feature weight;
3. local routing conductance;
4. actual gradient access;
5. eventual accuracy recovery.

---

# Main outputs

For every seed, K, and method the experiment records:

- accuracy;
- loss;
- routing weights;
- routing entropy;
- saturation rate;
- dominant-cue release;
- backup-cue recovery;
- routing conductance before adaptation;
- first-step backup-gradient norm;
- learned alpha, where applicable;
- accuracy on clean data after glasses adaptation.

That last metric checks whether adaptation merely fixes glasses by destroying normal behavior.

---

# Run on your M4 Pro

This project is `uv`-native and supports MPS.

```bash
uv sync

uv run python run_experiment.py \
  --device mps \
  --seeds 11 22 33 44 55 \
  --output results
```

For a quick smoke test:

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
  --output smoke_results
```

---

# What would strongly support the hypothesis?

The most informative result would be something like:

```text
slow_sgd
  conductance: ~0
  backup gradient: ~0
  glasses accuracy: 0.50 -> 0.50

additive_fast
  conductance: ~0
  backup gradient: ~0
  glasses accuracy: 0.50 -> 0.50

mixture_learned
  conductance: healthy
  backup gradient: healthy
  alpha rises after support
  w_A drops
  w_B/C rise
  glasses accuracy: 0.50 -> 0.68+
```

while clean accuracy remains reasonably high.

That would be direct evidence for a stronger claim than “avoid saturation”:

> **A model can preserve recoverability by maintaining an adaptation path whose gradients do not depend on the state of its slow routing policy.**

---

# Output files

```text
results/
├── results.csv
├── results.json
├── summary.txt
├── per_method_summary.csv
└── checkpoint_seed_<seed>.pt
```

Send the full `results/` directory back afterward.
