---
title: "Recoverable Feature Routing under Distribution Shift: Dormant-feature adaptive re-weighting prototype"
author: "Mykyta Siryk"
date: "4 October 2026, 18:28 UTC"
---

### Abstract

A 32-dimensional synthetic task tests whether an input-conditioned judge can reactivate a normally dormant feature after globally strong evidence fails. The recovered conversation reports 59.27% linear accuracy, 52.47% MLP accuracy, 80.10% zero-shot adaptive accuracy, and 98.00% four-shot adaptive accuracy. The gate assigned to feature 23 reportedly rises from 1.41 to 3.59. The raw result bundle was not recovered, so the numerical record is reported-only. The prototype motivates an image-derived falsification environment.

### Historical Manuscript Status

This is the cumulative scientific manuscript at release **v0.0.0**. It includes only experiments completed by 4 October 2026, 18:28 UTC; later findings are intentionally absent.


### 1. Introduction

Neural classifiers usually prioritize features that are most predictive across their training distribution. That behavior is efficient when feature reliability is stable, but brittle when a dominant cue fails while lower-priority evidence remains informative. The motivating deepfake-detection case is a model that trusts one family of texture, frequency, chromatic, geometric, or localized facial evidence until eyewear, occlusion, compression, or generator shift changes its reliability.

The desired response is not necessarily permanent retraining. A system should reconsider which evidence it trusts for the affected regime while preserving its established behavior elsewhere. This manuscript records a sequence of controlled falsification experiments designed to separate apparent adaptation from actual access to neglected evidence.

Each tagged version is a historical scientific snapshot. Negative results, implementation failures, and abandoned explanations are retained because they determine why the next experiment exists.

### 2. Related Work

Mixture-of-experts systems use learned gates for input-dependent expert selection [1,2]. MAML and fast-weight systems formalize rapid adaptation [3,4], while test-time adaptation and continual learning expose complementary distribution-shift and stability problems [5,6]. The present sequence isolates a narrower question: after a learned evidence policy becomes unreliable, does useful backup evidence remain both represented and reachable by a bounded adaptation process?



### 3. Dormant-feature Prototype

#### 3.1 Hypothesis

The initial hypothesis was that an input-conditioned judge could recognize that ordinary evidence had become unreliable and rapidly activate a feature that was globally weak but locally decisive. This was not posed as ordinary feature selection: the target behavior was conditional reconsideration of evidence after an unseen transformation.

#### 3.2 Construction

The binary task used 32-dimensional samples. Features 0--5 were globally predictive. Features 20--23 were normally noise. Three transformation families appeared during training. A held-out transformation corrupted strong features 0 and 5 while making feature 23 predictive. Linear and multilayer baselines were compared with zero-shot adaptive gating and a few-shot gate-offset update.

#### 3.3 Recovered results

| Method | Held-out accuracy | Evidence status |
|---|---:|---|
| Linear baseline | 0.5927 | reported-only |
| MLP baseline | 0.5247 | reported-only |
| Adaptive zero-shot | 0.8010 | reported-only |
| Adaptive four-shot | **0.9800** | reported-only |

The gate assigned to feature 23 reportedly increased from 1.41 to 3.59 after four-shot adaptation. These values were recovered from the conversation, not an original result archive, and therefore are not treated as independently verified measurements.

#### 3.4 Scientific decision

The prototype showed feasibility in a deliberately favorable world. It did not establish robustness, image-level applicability, or recovery under a genuinely difficult failure. The next experiment therefore moved to explicit image-derived evidence families and a held-out eyewear intervention.


### Interpretation at This Milestone

A dormant feature can be reactivated in a favorable construction, but the numerical evidence remains reported-only. The next stage must introduce image-derived evidence and a meaningful held-out intervention.

### Reason for the Next Stage

The limitations and unresolved mechanism at this milestone directly motivate Experiment 1; the next tagged manuscript records that test rather than silently rewriting this version.


### Limitations and Threats to Validity

The controlled worlds are binary, visually regular, and explicitly factorized. The support sets are labeled, only five principal seeds are used after the first stages, and the synthetic glasses intervention may be easier to recognize than natural distribution shifts. Best-operating-point comparisons may benefit from multiple testing. Experiment 0 lacks raw result artifacts, while Experiment 6 lacks completed results. Executable smoke tests establish software integrity, not reproduction of full-scale scientific outcomes.

### Reproducibility and Provenance

The accompanying repository separates byte-preserved historical source and results from repaired runnable snapshots. Each runnable stage provides a `uv` environment, exact entrypoint, bounded smoke test, and validation record. Reconstructed proposal, implementation, results, and analysis commits preserve why the sequence changed. Numerical claims come from stored outputs unless explicitly marked reported-only or results-missing.

### Disclosure of AI Assistance

ChatGPT collaborated in experimental ideation, implementation drafting, debugging, interpretation, manuscript preparation, and archival reconstruction. Git commits credit ChatGPT through co-author trailers. The exact ChatGPT model identifiers and reasoning-effort settings used during the original 4 October 2026 experiment conversations were not exposed in the recovered metadata and have not been inferred. Mykyta Siryk remains responsible for reviewing the code, evidence, claims, and manuscript.


### References

[1] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V. Le, G. E. Hinton, and J. Dean, “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer,” ICLR, 2017.

[2] W. Fedus, B. Zoph, and N. Shazeer, “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[3] C. Finn, P. Abbeel, and S. Levine, “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks,” Proceedings of the 34th International Conference on Machine Learning, PMLR 70, pp. 1126–1135, 2017.

[4] J. Ba, G. Hinton, V. Mnih, J. Z. Leibo, and C. Ionescu, “Using Fast Weights to Attend to the Recent Past,” Advances in Neural Information Processing Systems, 2016.

[5] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, “Tent: Fully Test-Time Adaptation by Entropy Minimization,” International Conference on Learning Representations, 2021.

[6] J. Kirkpatrick et al., “Overcoming Catastrophic Forgetting in Neural Networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, 2017.
