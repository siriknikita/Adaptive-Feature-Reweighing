---
title: "Recoverable Feature Routing under Distribution Shift: Held-out eyewear test and negative design result"
author: "Mykyta Siryk"
date: "4 October 2026, 18:36 UTC"
---

### Abstract

Adaptive fusion is transferred to an image world with RGB, edge, high-pass, and chromatic experts and a held-out glasses intervention. Every evaluated method reaches 1.000 accuracy on both clean and glasses data. This is a negative design result: edge evidence remains sufficient, so adaptation is unnecessary. A valid recovery test must destroy the dominant cue while leaving weaker backup evidence measurable.

### Historical Manuscript Status

This is the cumulative scientific manuscript at release **v0.1.0**. It includes only experiments completed by 4 October 2026, 18:36 UTC; later findings are intentionally absent.


### 1. Introduction

Neural classifiers usually prioritize features that are most predictive across their training distribution. That behavior is efficient when feature reliability is stable, but brittle when a dominant cue fails while lower-priority evidence remains informative. The motivating deepfake-detection case is a model that trusts one family of texture, frequency, chromatic, geometric, or localized facial evidence until eyewear, occlusion, compression, or generator shift changes its reliability.

The desired response is not necessarily permanent retraining. A system should reconsider which evidence it trusts for the affected regime while preserving its established behavior elsewhere. This manuscript records a sequence of controlled falsification experiments designed to separate apparent adaptation from actual access to neglected evidence.

Each tagged version is a historical scientific snapshot. Negative results, implementation failures, and abandoned explanations are retained because they determine why the next experiment exists.

### 2. Related Work

Mixture-of-experts systems use learned gates for input-dependent expert selection [1,2]. MAML and fast-weight systems formalize rapid adaptation [3,4], while test-time adaptation and continual learning expose complementary distribution-shift and stability problems [5,6]. The present sequence isolates a narrower question: after a learned evidence policy becomes unreliable, does useful backup evidence remain both represented and reachable by a bounded adaptation process?



### 3. Experiment 1 Setting

The first image-derived world contains four frozen evidence experts: RGB appearance, edges, high-pass residuals, and chromatic structure. Uniform fusion, a learned global vector, an input-conditioned router, and few-shot routing adaptation are evaluated. Router meta-training uses clean images, blur, JPEG compression, brightness changes, and lower occlusion while withholding a glasses overlay. The intended falsification criterion is a measurable clean-to-glasses failure followed by recovery through changed evidence weights.


### 4. Experiments

#### 4.1 Experiment 1: failure of an overly easy toy world

The first toy construction contained RGB, edge, high-pass, and chromatic expert views. It failed as a falsification environment because all fusion strategies achieved 1.000 accuracy on both clean and glasses-corrupted data. The perturbation increased cross-entropy loss, but the edge expert remained sufficiently diagnostic that the router had no incentive to rediscover weak evidence.

This negative result motivated a stricter construction in which the globally dominant feature is explicitly fragile.

### Interpretation at This Milestone

Universal perfect accuracy does not support recovery. It demonstrates that the intervention failed to invalidate decisive evidence. The next world must force fragile dominance while preserving weaker backups.

### Reason for the Next Stage

The limitations and unresolved mechanism at this milestone directly motivate Experiment 2; the next tagged manuscript records that test rather than silently rewriting this version.


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
