---
title: "Recoverable Feature Routing under Distribution Shift: Fragile dominance and few-shot recovery"
author: "Mykyta Siryk"
date: "4 October 2026, 18:46 UTC"
---

### Abstract

A dominant but fragile eye cue is separated from weaker lower-face and chromatic evidence. Held-out glasses reduce fixed global fusion to 0.5000 accuracy. Across five seeds, few-shot routing reaches 0.6302 +/- 0.1064 at eight shots and shifts routing mass toward backup evidence, but some seeds remain locked. The experiment demonstrates retention and partial recoverability while exposing high variance.

### Historical Manuscript Status

This is the cumulative scientific manuscript at release **v0.2.0**. It includes only experiments completed by 4 October 2026, 18:46 UTC; later findings are intentionally absent.


### 1. Introduction

Neural classifiers usually prioritize features that are most predictive across their training distribution. That behavior is efficient when feature reliability is stable, but brittle when a dominant cue fails while lower-priority evidence remains informative. The motivating deepfake-detection case is a model that trusts one family of texture, frequency, chromatic, geometric, or localized facial evidence until eyewear, occlusion, compression, or generator shift changes its reliability.

The desired response is not necessarily permanent retraining. A system should reconsider which evidence it trusts for the affected regime while preserving its established behavior elsewhere. This manuscript records a sequence of controlled falsification experiments designed to separate apparent adaptation from actual access to neglected evidence.

Each tagged version is a historical scientific snapshot. Negative results, implementation failures, and abandoned explanations are retained because they determine why the next experiment exists.

### 2. Related Work

Mixture-of-experts systems use learned gates for input-dependent expert selection [1,2]. MAML and fast-weight systems formalize rapid adaptation [3,4], while test-time adaptation and continual learning expose complementary distribution-shift and stability problems [5,6]. The present sequence isolates a narrower question: after a learned evidence policy becomes unreliable, does useful backup evidence remain both represented and reachable by a bounded adaptation process?


### 3. Controlled Experimental Setting

#### 3.1 Evidence families

Each synthetic image belongs to one of two classes and contains three separately accessible evidence families:

- **A: eye-region evidence.** A nearly deterministic localized marker makes this the dominant cue on clean data.
- **B: lower-face geometry.** A weaker geometric pattern is correlated with the label with probability 0.72.
- **C: chromatic evidence.** A weaker global color bias is correlated with the label with probability 0.66.

Three small convolutional experts receive feature-isolated views. Expert A sees the eye band, expert B sees grayscale lower-face geometry, and expert C sees an opponent-color representation.

The separation is intentional. It allows us to measure whether a model retains genuinely useful backup information after A fails, rather than inferring feature behavior from a monolithic latent representation.

#### 3.2 Held-out intervention

The final intervention is a synthetic glasses overlay applied to the eye band. It is not used during slow-router training. The perturbation is designed to invalidate A while leaving B and C largely intact.

The router is meta-trained on other perturbations, including eye-band masking, lower-face masking, desaturation, blur, JPEG compression, and clean inputs. Thus, the system is exposed to the general idea that evidence reliability can change, but it does not learn a direct glasses-specific rule.

#### 3.3 Evaluation protocol

The principal runs use five seeds: 11, 22, 33, 44, and 55. Few-shot support sizes are 2, 4, 8, 16, and 32 labeled glasses examples. Support samples are excluded from query evaluation.

The main metrics are shifted-regime accuracy, clean accuracy after adaptation, expert routing weights, routing entropy, and saturation rate. Later experiments add explicit measures of dominant-cue release, backup-cue recovery, routing Jacobian conductance, and initial backup-gradient magnitude.

### 4. Experiments

#### 4.1 Experiment 1: failure of an overly easy toy world

The first toy construction contained RGB, edge, high-pass, and chromatic expert views. It failed as a falsification environment because all fusion strategies achieved 1.000 accuracy on both clean and glasses-corrupted data. The perturbation increased cross-entropy loss, but the edge expert remained sufficiently diagnostic that the router had no incentive to rediscover weak evidence.

This negative result motivated a stricter construction in which the globally dominant feature is explicitly fragile.

#### 4.2 Experiment 2: fragile dominance and few-shot recovery

Experiment 2 introduced evidence families A, B, and C. On clean data, the learned global fusion placed approximately 95.7% of routing mass on A. Under glasses, the fixed global system fell to exactly 0.5000 mean accuracy across five seeds.

The backup experts remained useful. Under glasses, B and C independently retained above-chance classification performance, while A collapsed to chance. A support-optimized fusion reference reached 0.6944 ± 0.0302 accuracy at 32 shots.

The adaptive router partially recovered:

| Method | K | Glasses accuracy |
|---|---:|---:|
| Global fixed fusion | 0 | 0.5000 ± 0.0000 |
| Adaptive zero-shot | 0 | 0.5008 ± 0.0016 |
| Adaptive few-shot | 2 | 0.5872 ± 0.1052 |
| Adaptive few-shot | 8 | **0.6302 ± 0.1064** |
| Adaptive few-shot | 32 | 0.6300 ± 0.1061 |
| Support-optimized fusion reference | 32 | 0.6944 ± 0.0302 |

At 8 shots, mean routing changed from approximately

$$
[A,B,C]=[0.904,0.088,0.008]
$$

before glasses-specific adaptation to

$$
[A,B,C]=[0.451,0.539,0.010].
$$

The mechanism therefore worked in aggregate, but the standard deviation was large. Inspection showed that some seeds escaped from A while others remained almost perfectly locked to it.

### Interpretation at This Milestone

Backup evidence remains represented and is sometimes recoverable, but access is seed-dependent. Feature presence and reliable rapid adaptation are distinct properties.

### Reason for the Next Stage

The limitations and unresolved mechanism at this milestone directly motivate Experiment 3; the next tagged manuscript records that test rather than silently rewriting this version.


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
