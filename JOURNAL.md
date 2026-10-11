# Experimental Journal

## Experiment 0 — Dormant-feature prototype

**Status:** promising but favorable prototype  
**Historical window:** 2026-10-04 18:28 UTC  
**Evidence:** `historical/exp00/source/`; conversation-reported smoke metrics

### Question

Can an input-conditioned judge discover that a normally useless feature becomes useful under an unseen transformation, and can a few labeled samples amplify it without changing the base network?

### Protocol and implementation

The synthetic vector world used 32 features. Features 0–5 were globally strong; features 20–23 were normally noise. Three transformation families were used during training, while held-out transformation D corrupted strong features 0 and 5 and made feature 23 predictive. Systems were a linear baseline, an MLP, zero-shot adaptive gating, and few-shot gate-offset adaptation.

### Observations

Conversation-reported values were: base linear 59.27%, plain MLP 52.47%, adaptive zero-shot 80.10%, and four-shot adaptation 98.00%. Dormant feature 23's gate reportedly increased from 1.41 to 3.59. These are **reported-only** because the corresponding result directory was not present in the recovered files.

### Conclusion and decision

The mechanism was possible in a favorable world, but the experiment did not test image evidence or a genuinely difficult cue failure. Move to isolated image feature families and a held-out glasses intervention.

---

## Experiment 1 — Overly easy image world

**Status:** negative design result  
**Historical window:** 2026-10-04 18:31–18:36 UTC  
**Evidence:** `historical/exp01/`

### Question

Can a per-image judge reweight RGB, edge, high-pass, and chromatic experts after a glasses-like perturbation withheld from meta-training?

### Observations

All fusion strategies achieved 1.000 accuracy on clean and glasses-corrupted data. The perturbation increased loss but left the edge signal sufficiently predictive. Adaptation was unnecessary, so perfect performance did not support the intended recoverability claim.

### Bug ledger

No result-invalidating software defect was established. The failure was methodological: the synthetic world was too easy.

### Conclusion and decision

Reject Experiment 1 as a useful test of adaptive recovery. Construct a world with a deliberately dominant but fragile eye cue and weaker stable backup cues.

---

## Experiment 2 — Fragile dominance and neglected evidence

**Status:** partial positive result with high seed variance  
**Historical window:** 2026-10-04 18:41–18:46 UTC  
**Evidence:** `historical/exp02/results/results_5seed/`

### Question

When held-out glasses invalidate dominant evidence A, can few-shot adaptation recover weaker evidence B or C?

### Protocol

Five seeds: 11, 22, 33, 44, 55. Cue A was a strong eye-band marker; B a lower-face geometric cue correlated at 0.72; C a chromatic cue correlated at 0.66. Support sizes were 2, 4, 8, 16, and 32.

### Observations

Fixed global fusion fell to 0.5000 glasses accuracy. Adaptive zero-shot was 0.5008 ± 0.0016. Few-shot routing reached 0.6302 ± 0.1064 at K=8 and 0.6300 ± 0.1061 at K=32. A support-optimized diagnostic reached 0.6944 ± 0.0302 at K=32. At K=8 the mean route changed approximately from [A,B,C]=[0.904,0.088,0.008] to [0.451,0.539,0.010].

### Bug ledger

The first uploaded result bundle contained both a one-seed root run and a five-seed subdirectory. This is an archival ambiguity, not a numerical defect; the five-seed directory is the basis of the main conclusion.

### Conclusion and decision

Backup evidence exists and can be recovered, but the high variance reveals seed-dependent lock-in. Next test whether ordinary anti-saturation mechanisms preserve functional access to neglected routes.

---

## Experiment 3 — Output mass versus gradient recoverability

**Status:** informative negative ablation  
**Historical window:** 2026-10-04 18:52–18:59 UTC  
**Evidence:** `historical/exp03/results/`

### Question

Does preventing zero or near-zero expert probability restore fast adaptation?

### Observations

Standard softmax peaked at 0.5870 ± 0.1065; logit clipping peaked at 0.5877 ± 0.1074. The epsilon floor eliminated the measured output-saturation flag but reached only 0.5444 ± 0.0878 at K=32. Temperature softmax and bounded residual routing were also insufficient.

### Conclusion and decision

Nonzero routing mass is not equivalent to gradient access. An epsilon floor changes output weights while derivatives with respect to the slow logits still inherit the saturated softmax Jacobian. Build a fast distribution whose gradients are independent of the slow map.

---

## Experiment 4 — Independent fast adaptation

**Status:** positive recovery with stability cost  
**Historical window:** 2026-10-04 19:09–19:26 UTC  
**Evidence:** `historical/exp04/results/`

### Question

Can an independent fast routing distribution recover backup evidence more reliably than adapting through the slow saturated route?

### Observations

Slow SGD reached 0.6302 ± 0.1064 glasses accuracy at K=8 with clean accuracy 0.9226. Fast-only routing reached 0.6986 ± 0.0259 at K=32, but clean accuracy fell to 0.7158. Its initial routing conductance was 0.4714. A learned mixture preserved 0.9974 clean accuracy at K=4 but recovered only 0.6070 ± 0.0954 glasses accuracy.

### Conclusion and decision

Independent gradient access improves both recovery and seed stability, but applying the fast policy globally creates a stability–plasticity trade-off. The next experiment must determine *when* to activate the fast route.

---

## Experiment 5 — Selective fast-path activation

**Status:** positive controlled mechanism result  
**Historical window:** 2026-10-04 19:49–20:03 UTC  
**Evidence:** `historical/exp05/results/`; paper draft v2

### Question

Can a context switch invoke the support-adapted fast route only for the new regime while retaining the slow route for familiar inputs?

### Observations

At K=32 the memory switch achieved 0.6986 ± 0.0259 glasses accuracy and 1.0000 ± 0.0000 clean accuracy, with harmonic score 0.8223, glasses activation rate 1.000, and clean false-activation rate 0.000. It matched the environment oracle in this controlled world. The prototype switch separated regimes nearly perfectly at threshold level but produced mixture coefficients near 0.5, revealing a calibration rather than discrimination failure.

### Conclusion and decision

Recoverable feature routing requires three distinct properties: evidence retention, a gradient-accessible adaptation path, and selective activation. The synthetic result warrants a real-image pilot but does not establish deepfake performance.

---

## Experiment 6 — Real-data T4 bridge

**Status:** code-only; results missing  
**Historical time:** 2026-10-04 20:17 UTC  
**Evidence:** `historical/exp06/source/`

### Question

Does the retention–recoverability–selection mechanism survive on real deepfake benchmark images under a procedural eyewear intervention?

### Protocol

The pilot uses a public identity-isolated face dataset, four ResNet-18 evidence experts, sequential expert training, cached logits/context embeddings, one representation seed, and five routing seeds. It is designed for a Colab T4.

### Validation and open status

The repaired snapshot resolves dependencies, compiles, and exposes a valid CLI. No historical result archive was recovered, and the full dataset/training run was not performed during archival reconstruction.

### Decision

Do not make a real-data claim. Execute the T4 run, return `results_exp6.zip`, and then create the next results and analysis commits on this preserved branch lineage.

