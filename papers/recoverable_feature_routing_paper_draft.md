# Recoverable Feature Routing under Distribution Shift:
## A Controlled Study of Gradient Accessibility and Selective Fast Adaptation

**Anonymous draft — October 2026**

### Abstract

Neural classifiers commonly learn to prioritize features that are most predictive on the training distribution. This is efficient when feature reliability is stable, but it can become brittle when a dominant feature fails while lower-priority evidence remains informative. We study this failure mode in a controlled synthetic image setting containing three explicitly isolated evidence families: a dominant but fragile eye-region cue, a weaker lower-face geometric cue, and a weaker chromatic cue. A held-out glasses perturbation invalidates the dominant cue without removing the two backup cues. Across five seeds, fixed global fusion falls to 0.5000 accuracy under this shift, while a few-shot adaptive router recovers to 0.6302 ± 0.1064 at 8 shots. However, recovery is seed-dependent because some routers become effectively locked to the failed feature. Anti-saturation transformations reduce apparent output collapse but do not consistently restore adaptation, showing that nonzero routing mass is not equivalent to gradient accessibility. An independent fast routing path raises held-out accuracy to 0.6986 ± 0.0259 at 32 shots, but applying the fast policy globally reduces clean accuracy to 0.7158. A learned slow-fast mixture instead retains 0.9974 clean accuracy at its best clean-preserving operating point but recovers less of the shifted regime. These results motivate a stability-plasticity formulation in which recoverability requires both an independent adaptation path and a mechanism for selectively invoking it. We therefore introduce a support-keyed selective fast-path experiment that conditions fast-path activation on similarity to a newly observed regime while preserving the original slow policy for familiar inputs.

### 1. Introduction

Feature selection is usually discussed as a question of global utility: features that contribute strongly across a dataset are preserved or amplified, while weak features are downweighted, pruned, or ignored. This view is reasonable when feature reliability is approximately stationary. It is less satisfactory when a feature that is weak on average becomes useful for a specific input regime, or when the feature that is normally strongest becomes unreliable.

The motivating example is deepfake detection. A detector may learn to rely strongly on a particular family of texture, frequency, chromatic, geometric, or localized facial cues. An intervention such as eyewear, occlusion, compression, or a generator-specific artifact can alter the reliability of one family without destroying all remaining evidence. In such a case, the desired behavior is not necessarily permanent retraining. The model should be able to temporarily reconsider which evidence it trusts.

This paper isolates that mechanism in a deliberately controlled synthetic setting before attempting a full deepfake benchmark. The experiments are built around three questions.

1. Can globally weak evidence remain useful after the dominant feature fails?
2. If useful backup evidence still exists, can an adaptive router recover access to it?
3. If a fully independent fast path restores adaptation but damages old-distribution performance, can that path be invoked selectively?

The central distinction that emerges is between **feature retention** and **feature recoverability**. A representation may still exist in the model and may even retain nonzero routing probability, yet remain functionally inaccessible to rapid adaptation because the routing map has entered a low-gradient state.

Our controlled experiments provide four main observations. First, a dominant feature can hide the intended adaptive behavior entirely when the task is too easy. Second, after enforcing a fragile-dominant/stable-backup structure, few-shot routing can recover backup evidence, but recovery is highly seed dependent. Third, output-level anti-collapse heuristics do not necessarily preserve local gradient access. Fourth, a fast routing path whose gradients are independent of the slow routing state substantially improves recovery and reduces seed variance, but creates a stability-plasticity trade-off.

The current fifth experiment tests a support-keyed activation mechanism intended to combine the recovery of a fast path with the retention of the original slow policy.

### 2. Related Work

#### 2.1 Conditional computation and mixture-of-experts routing

Mixture-of-experts models use input-dependent gates to select or combine specialized subnetworks. Sparsely gated MoE layers demonstrated that conditional computation can greatly increase model capacity without activating all parameters for every input [1]. Switch Transformers later simplified sparse routing and highlighted the training instabilities associated with learned expert selection [2]. Our setting is much smaller and is not concerned with scaling model capacity. Instead, we use expert routing as a controlled instrument for studying whether low-priority evidence remains recoverable after the routing policy becomes overconfident.

#### 2.2 Few-shot and fast adaptation

Model-Agnostic Meta-Learning explicitly optimizes a model so that a small number of gradient steps on a new task can produce useful adaptation [3]. Fast-weight approaches similarly distinguish rapidly changing variables from slower learned parameters [4]. Our experiments adopt this slow/fast distinction but focus on a different failure mode: the fast update itself may be ineffective if its gradient must pass through a saturated slow routing map.

#### 2.3 Test-time adaptation and stability-plasticity

Test-time adaptation methods such as Tent update model parameters at inference time to reduce distribution-shift error [5]. Continual-learning work studies the complementary problem of acquiring new behavior without destroying previously learned behavior. Elastic Weight Consolidation preserves old-task performance by constraining changes to important parameters [6]. Our results expose a related but more local trade-off: a fast policy can adapt strongly to a shifted regime, while indiscriminate use of that policy can damage performance on the original regime. We therefore treat selective activation as a stability-plasticity problem rather than merely an optimization problem.

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

\[
[A,B,C]=[0.904,0.088,0.008]
\]

before glasses-specific adaptation to

\[
[A,B,C]=[0.451,0.539,0.010].
\]

The mechanism therefore worked in aggregate, but the standard deviation was large. Inspection showed that some seeds escaped from A while others remained almost perfectly locked to it.

#### 4.3 Experiment 3: output anti-collapse is not gradient recoverability

Experiment 3 compared ordinary softmax routing with temperature scaling, an epsilon probability floor, logit clipping, and bounded residual routing.

The main result was negative but informative. An epsilon floor eliminated measured output saturation while improving glasses accuracy only to 0.5444 ± 0.0878 at 32 shots. Logit clipping reached 0.5877 ± 0.1074 at 8 shots, similar to ordinary softmax. No anti-collapse transformation consistently eliminated the seed-dependent failure.

This shows that a router can assign nonzero mass to a backup expert without making that expert meaningfully accessible to gradient-based adaptation. For an epsilon-floor transformation,

\[
w_i=(1-E\epsilon)\operatorname{softmax}(z)_i+\epsilon,
\]

the output probability may be bounded away from zero while the derivative with respect to the underlying logit remains proportional to the already saturated softmax derivative.

The relevant property is therefore not merely whether \(w_i>0\), but whether the adaptation parameters retain sufficient local influence over \(w_i\).

#### 4.4 Experiment 4: independent fast pathways

Experiment 4 directly tested this gradient-accessibility hypothesis. All methods within a seed shared the exact same expert bank, slow router, support set, and query set.

A standard slow update and an additive fast-logit update both continue to route through the slow softmax geometry. In contrast, the independent mixture uses

\[
p_s=\operatorname{softmax}(z_s), \qquad
p_f=\operatorname{softmax}(a),
\]

and

\[
w=(1-\alpha)p_s+\alpha p_f.
\]

The fast distribution therefore has a routing Jacobian independent of the slow distribution.

The fast-only route produced the strongest shifted-regime recovery:

| Method | Best K | Glasses accuracy | Clean accuracy after adaptation |
|---|---:|---:|---:|
| Slow SGD | 8 | 0.6302 ± 0.1064 | 0.9226 |
| Additive fast logits | 8 | 0.6302 ± 0.1064 | 0.8668 |
| Fixed independent mixture | 32 | 0.5657 ± 0.0805 | 1.0000 |
| Learned independent mixture | 4 | 0.6070 ± 0.0954 | **0.9974** |
| Fast-only independent path | 32 | **0.6986 ± 0.0259** | 0.7158 |

The fast-only route also had markedly lower variance than the slow adaptive router. Its initial routing conductance was 0.4714, compared with approximately 0.14 on average for the slow route and nearly zero on the most strongly locked seeds.

These results support two conclusions.

First, a fresh parameter vector is not sufficient by itself: if it is inserted before the same saturated softmax, adaptation can still fail.

Second, a computationally independent fast distribution restores gradient access, but indiscriminate application of the fast policy creates forgetting on the clean regime.

This produces a clear stability-plasticity frontier.

#### 4.5 Experiment 5: support-keyed selective activation

Experiment 5 is designed to close the gap exposed by Experiment 4.

The expert bank and slow router are frozen. A single fast policy \(p_f\) is adapted on the K glasses support examples and then shared by all selective-activation methods. The experimental variable is therefore not what the fast path learns, but **when the fast path is invoked**.

The final route is

\[
w(x)=(1-\alpha(x))p_s(x)+\alpha(x)p_f.
\]

We compare five deployable strategies:

1. direct slow-router adaptation;
2. fast-only routing;
3. a single learned global mixture coefficient;
4. a support-versus-clean prototype switch;
5. a linear clean-memory context switch.

For the prototype switch, the frozen router provides a context embedding \(\phi(x)\). The activation score is

\[
s(x)=
\cos(\phi(x),\mu_{\mathrm{support}})
-
\cos(\phi(x),\mu_{\mathrm{clean}}).
\]

The threshold is calibrated only on the few-shot support set and a balanced clean reference memory. No query labels are used.

The linear memory switch learns only an environment decision:

\[
\alpha(x)
=
P(\text{new regime}\mid\phi(x)).
\]

Its positive examples are the glasses support set and its negative examples are 128 clean reference samples from the pre-existing validation distribution. It does not receive fake/real class targets.

We evaluate both shifted-regime recovery and post-adaptation clean retention. The principal joint metric is the harmonic mean

\[
H=
\frac{2A_{\mathrm{glasses}}A_{\mathrm{clean}}}
{A_{\mathrm{glasses}}+A_{\mathrm{clean}}}.
\]

The experiment additionally records mean activation on glasses and clean inputs, the activation gap, new-regime true activation rate, and clean false activation rate.

At the time of this draft, Experiment 5 has been implemented and syntax/smoke tested, but its full five-seed results have not yet been generated. The paper therefore makes no empirical claim about selective activation beyond the Experiment 4 motivation.

### 5. Interpretation

The experiments suggest a hierarchy of increasingly strong requirements for adaptive evidence use.

**Feature presence is insufficient.** Backup experts B and C can remain predictive while the model fails under shift.

**Nonzero routing weight is insufficient.** Probability floors can make every route numerically active without restoring effective adaptation.

**Fresh adaptation parameters are insufficient if they inherit the same saturated map.** Additive fast logits can still fail when their influence is mediated by the slow softmax.

**Independent gradient access improves recoverability.** A separate fast distribution substantially increases shifted-regime recovery and reduces seed-dependent failure.

**Independent access alone is insufficient for stability.** Applying the fast distribution to all inputs damages the original regime.

This motivates a stronger design principle:

> An adaptive model should preserve a low-latency path to neglected evidence whose gradients do not depend on the confidence state of the slow policy, and it should condition activation of that path on evidence that the current input belongs to a regime where the slow policy is unreliable.

We use the term **recoverable feature routing** for this combination of representational retention, gradient accessibility, and selective activation.

### 6. Limitations

The present evidence is intentionally narrow.

First, all experiments use synthetic binary images. The construction is valuable because causal roles of A, B, and C are known, but the results do not establish performance on natural images or deepfakes.

Second, the glasses intervention is synthetic and deterministic. Natural eyewear introduces reflection, transparency, pose, lens distortion, frame diversity, and interaction with identity.

Third, the experts are explicitly factorized by design. Real neural representations are entangled, so identifying a recoverable low-priority feature family may be substantially harder.

Fourth, support examples are labeled. A deployment setting may require unlabeled or weakly labeled adaptation.

Fifth, the current experiments study one dominant-feature failure at a time. Multiple simultaneous cue failures may require a richer fast policy than a single global fusion vector.

Finally, Experiment 5 introduces a small clean reference memory. This is a deliberate stability mechanism, but future work should compare reference-memory methods with fully source-free alternatives.

### 7. Path to a Real Deepfake Study

A deepfake follow-up should preserve the controlled logic while removing synthetic assumptions.

A suitable protocol would include identity-disjoint training and testing, generator-disjoint evaluation, naturally occurring glasses and sunglasses, generated eyewear interventions, compression and blur controls, and multiple independently measurable feature families such as spatial texture, frequency residuals, chromatic statistics, geometric consistency, and physically motivated cues.

The first goal should not be a new state-of-the-art detector. It should be to test whether the same phenomenon appears in a real forensic setting:

1. one feature family dominates on the source distribution;
2. a specific condition reduces its reliability;
3. lower-priority evidence remains informative;
4. a conventional adaptive router becomes locked for some inputs or seeds;
5. an independent selective fast path restores access without degrading ordinary inputs.

If that sequence survives a realistic benchmark, recoverability becomes a meaningful architectural property rather than an artifact of the toy world.

### 8. Conclusion

This work began from a simple question: should a model permanently downweight evidence that is weak on average if that evidence may become useful for a particular input?

The controlled experiments show that the deeper issue is not feature weighting alone. A model can retain useful evidence yet lose practical access to it through an overconfident routing policy. Output-level anti-collapse mechanisms do not necessarily solve this because probability mass and gradient accessibility are different properties.

An independent fast path restores much of the lost adaptability, reaching 0.6986 ± 0.0259 accuracy under the held-out shift compared with approximately 0.50 for the unadapted slow policy, while also reducing variance across seeds. The cost is reduced performance on the original regime when the fast path is applied indiscriminately. A globally learned mixture preserves clean accuracy but gives up part of the recovery.

The resulting research question is therefore a stability-plasticity problem: how can a model preserve an independently recoverable evidence path and activate it only when the current input justifies overriding the slow policy?

Experiment 5 directly tests that selective-activation hypothesis.

### References

[1] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V. Le, G. E. Hinton, and J. Dean, “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer,” ICLR, 2017.

[2] W. Fedus, B. Zoph, and N. Shazeer, “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[3] C. Finn, P. Abbeel, and S. Levine, “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks,” Proceedings of the 34th International Conference on Machine Learning, PMLR 70, pp. 1126–1135, 2017.

[4] J. Ba, G. Hinton, V. Mnih, J. Z. Leibo, and C. Ionescu, “Using Fast Weights to Attend to the Recent Past,” Advances in Neural Information Processing Systems, 2016.

[5] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, “Tent: Fully Test-Time Adaptation by Entropy Minimization,” International Conference on Learning Representations, 2021.

[6] J. Kirkpatrick et al., “Overcoming Catastrophic Forgetting in Neural Networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, 2017.
