# Feature Composition in Factual Recall: A Preliminary Investigation

**Model:** GPT-2 Small (117M parameters)  
**Framework:** TransformerLens  
**Dataset:** 22 validated factual recall examples  

---

## Abstract

We study how GPT-2 Small internally represents factual knowledge by identifying which model components (MLP neurons and attention heads) drive factual recall predictions, and asking whether those components work together or independently. We build a curated dataset of 22 "sweet-spot" examples where the model is moderately confident in the correct answer (~55–75%) while still assigning meaningful probability (~10–23%) to a plausible wrong answer. Using direct logit attribution (DLA) and zero-ablation, we rank the top-10 contributing features and then run pairwise compositionality tests across all 45 feature pairs. We find that (1) the top features are almost exclusively specialized for geography, not for factual recall in general; (2) they split cleanly into geography-promoter and geography-suppressor groups; and (3) pairwise interactions are uniformly near-additive (all within ±0.033 of expected), indicating that factual recall is encoded in a distributed, largely non-interacting collection of features rather than a tightly coupled circuit.

---

## 1. Introduction

### Why feature composition matters for AI safety

One of the central goals of mechanistic interpretability is to understand *how* neural networks store and retrieve information. This matters for AI safety for a concrete reason: if we can identify which internal components are responsible for a specific behavior, we can begin to reason about when that behavior will generalize, when it will fail, and how it could be altered or corrected.

A particularly important question is whether internal representations *compose* — whether the model combines information from multiple sources in a predictable, structured way, or whether it relies on a small set of monolithic features that do all the work. A model with compositional, distributed representations may be more robust and interpretable; a model that encodes everything in a few critical components may be brittle and harder to audit.

This paper investigates feature composition in GPT-2 Small's factual recall. We focus on a narrow but well-controlled task: given a prompt that establishes a pattern (e.g., two completed "capital of X is Y" sentences), does the model retrieve the correct next capital city? And if so, which internal features are responsible, and do those features work together or independently?

### What we find (preview)

- Finding 1: The top features identified by attribution are almost entirely geography-specific. They fire strongly for capital-city prompts and are nearly inert on science/physics facts, meaning the attribution ranking reflects a geography circuit rather than general factual recall.

- Finding 2: Within that geography circuit, features split into two opposing groups — promoters that push logits toward the correct capital, and suppressors that push logits toward the most plausible wrong capital. Both groups are necessary to explain why the model is confident but not certain.

- Finding 3: Pairwise compositionality tests show that all 45 top-feature pairs interact near-additively. No pair is synergistic enough to constitute a "circuit" in the strong sense. Factual recall is distributed across many weakly contributing components.

---

## 2. Methods

### 2.1 Model

We use GPT-2 Small (12 transformer layers, 12 attention heads per layer, 768-dimensional residual stream, 3072-dimensional MLP hidden layer) loaded via TransformerLens, which exposes all internal activations and supports hook-based interventions. All inference is done in evaluation mode with no gradient tracking.

### 2.2 Dataset construction

**The sweet-spot constraint.** We require examples where the model is correct but not certain:

- Correct answer first-token probability > 50%  
- Most plausible wrong answer first-token probability > 10%

This constraint is deliberately narrow. It filters out facts the model either knows with near-certainty (>80%, no meaningful competition) or barely knows (<40%, too uncertain to study attribution cleanly). Only examples in this "sweet spot" have the internal tension — competing representations for correct and incorrect answers — that makes feature attribution meaningful.

**Why this is hard to satisfy.** GPT-2 Small's factual recall is nearly bimodal: facts it has encoded with high confidence receive 80–99% probability, while facts it has not encoded get <40%. Finding examples in the 50–75% range required testing over 300 candidate prompts across five exploration scripts. Approximately 7–8% of candidates passed.

**The cap2() prompt format.** Simple prompts ("The capital of France is") give GPT-2 less than 1% probability for "Paris" — the model treats them as open-ended completions rather than factual queries. We use a two-shot priming format:

```
The capital of Italy is Rome. The capital of Spain is Madrid. The capital of France is
```

Two completed seed examples prime the model to recognize the pattern and continue it. Without seeds, confidence is too low; with seeds chosen from the same set as the target, confidence is too high. The specific seed pairs were selected empirically to land in the sweet spot.

**Final dataset.** 22 validated examples across three categories:
- Geography: 13 examples (capital city prompts using the cap2() format)
- Attributes: 7 examples (physics/science facts: chemical composition, temperature scales, planetary ordering)
- History: 2 examples (famous name completions: "Sir Isaac" → Newton, "The Mona Lisa was painted by Leonardo" → da)

Correct probabilities range from 0.507 to 0.751 (mean 0.607). Foil probabilities range from 0.101 to 0.232 (mean 0.148).

**Foil derivation.** Wrong answers ("foils") are derived empirically from the model's own top-k output distribution, not from human priors. Human-specified foils are often wrong — GPT-2 Small does not confuse Paris with Marseille for the same reasons a person would. The model's actual second-choice tokens are what create the internal competition we want to study.

### 2.3 Feature attribution (Direct Logit Attribution)

We score each feature by how much its output pushes the model's final prediction toward the correct answer versus the foil. Formally, for a given (correct, foil) token pair, we compute:

```
u_diff = W_U[:, correct_id] - W_U[:, foil_id]
```

where `W_U` is the unembedding matrix. This is the direction in the model's residual stream that, when projected, increases the correct answer's logit relative to the foil.

For an MLP neuron `(layer l, neuron n)`:
```
attribution = activation[n] × (W_out[l][n] · u_diff)
```

For an attention head `(layer l, head h)`:
```
attribution = head_output[l][h] · u_diff
```

A positive attribution means the feature is pushing the model toward the correct answer; negative means it is pushing toward the foil.

We compute attributions across MLP layers 6–8 and attention layers 9–10 (the late-processing layers most relevant to factual recall in GPT-2 Small), average over all 22 examples by absolute value, and rank the results. The top-20 features are saved; experiments use the top-10.

### 2.4 Feature ablation

To test causal importance rather than correlational attribution, we zero-ablate each top-10 feature individually: a forward hook sets the feature's activation to 0.0 during inference, and we measure the change in logit difference (correct − foil) across all 22 examples. A large positive drop means the feature was genuinely helping; a negative drop (improvement) means it was suppressing performance.

### 2.5 Pairwise compositionality test

For each of the 45 unique pairs of top-10 features, we ablate both simultaneously and measure the combined drop. We define:

```
interaction(A, B) = drop_AB - (drop_A + drop_B)
```

where `drop_A` and `drop_B` come from the single-feature ablation results (Experiment 2), and `drop_AB` is measured by zeroing both features in a single forward pass.

- `interaction > 0`: synergistic — removing both hurts more than expected
- `interaction ≈ 0`: additive/independent — features do separable work  
- `interaction < 0`: redundant — features share their function

When two features occupy the same hook point (e.g., two attention heads in the same layer both modify `hook_z`), we register a single combined hook that zeros both simultaneously, avoiding hook stacking issues.

### 2.6 Feature specialization analysis

To measure category-level specialization, we use the per-example attribution scores already computed in Experiment 1. Each feature has 22 attribution values — one per dataset example, in dataset order. We group these by category and compute the mean for each (feature, category) pair, yielding a 10×3 matrix. No additional model inference is needed.

---

## 3. Results

### 3.1 Finding 1 — The top features are a geography circuit, not a general factual recall circuit

**Figure reference: `experiment4_specialization.png`**

The table below shows mean DLA attribution per feature per category. Positive values push toward the correct answer; negative values push toward the foil.

| Feature | Attributes | Geography | History |
|---|---|---|---|
| L10, Head 0 | −1.97 | −2.45 | +0.16 |
| L10, Head 6 | +0.06 | **+2.86** | −0.21 |
| L8, Neuron 538 | −0.12 | −2.37 | +0.13 |
| L8, Neuron 1028 | −0.04 | +1.59 | +0.03 |
| L9, Head 2 | −0.89 | **+2.04** | −0.14 |
| L8, Neuron 1460 | −0.04 | +1.41 | +0.05 |
| L8, Neuron 2108 | −0.16 | −1.28 | +0.05 |
| L9, Head 5 | +0.24 | −1.31 | −0.33 |
| L8, Neuron 1458 | +0.02 | −1.17 | −0.00 |
| L8, Neuron 1972 | −0.05 | +1.05 | −0.04 |

The geography column dominates by a wide margin. Seven of the ten features have near-zero attributions for attributes (|avg| < 0.25) and near-zero for history (|avg| < 0.34), while their geography attributions are 5–20× larger in magnitude.

The implication is that our top-10 feature ranking is effectively a ranking of *geography-relevant* features. A separate attribution study focused on the 7 attribute examples would likely surface a completely different set of important components. The "most important features for factual recall" depends heavily on which facts are in scope.

### 3.2 Finding 2 — Features split into geography promoters and suppressors

Within the geography-specialized features, a clean two-group structure emerges:

**Geography promoters** (positive DLA → push toward correct capital):
- L10, Head 6: geography avg = +2.86
- L9, Head 2: geography avg = +2.04
- L8, Neuron 1028: geography avg = +1.59
- L8, Neuron 1460: geography avg = +1.41
- L8, Neuron 1972: geography avg = +1.05

**Geography suppressors** (negative DLA → push toward wrong capital):
- L10, Head 0: geography avg = −2.45
- L8, Neuron 538: geography avg = −2.37
- L8, Neuron 2108: geography avg = −1.28
- L9, Head 5: geography avg = −1.31
- L8, Neuron 1458: geography avg = −1.17

This structure explains why the model lands in the sweet spot at all. The promoters encode and amplify the correct capital token; the suppressors encode competition from the most plausible alternative (e.g., Barcelona when predicting the Spanish capital). The correct answer wins because the promoters collectively outweigh the suppressors — but the suppressors are why the foil receives 10–23% probability rather than 0%.

Ablation confirms this split causally. Ablating the largest promoter (L10, Head 6) drops the geography logit difference from +1.24 to +0.98 — a 0.26 unit reduction. Ablating the largest suppressor (L8, Neuron 538) *improves* the geography logit difference from +1.24 to +1.50, confirming that the suppressor was actively working against the correct answer.

### 3.3 Finding 3 — Pairwise feature interactions are near-additive

**Figure reference: `experiment3_compositionality.png`**

We tested all 45 pairs of the top-10 features. The interaction terms — the deviation from perfect additivity — are uniformly small:

| Most extreme pairs | Interaction |
|---|---|
| L9/Head 5 × L8/Neuron 1458 | −0.033 (most redundant) |
| L10/Head 6 × L9/Head 2 | +0.027 (most synergistic) |
| L10/Head 0 × L8/Neuron 1028 | −0.027 (redundant) |
| L8/Neuron 1028 × L9/Head 5 | +0.026 (synergistic) |

All 45 interactions lie within ±0.033 of zero. The baseline logit difference is 1.358, so the largest interaction represents a deviation of roughly 2.4% from the additive prediction. This is within the noise range of a 22-example average.

The largest redundant pair (Head 5 × Neuron 1458) makes sense structurally: both are geography suppressors with similar per-example patterns. Removing either one partly substitutes for removing the other. The largest synergistic pair (Head 6 × Head 2) also makes sense: both are geography promoters. When you remove both simultaneously, the model loses more of its capital-city signal than you would predict from removing each independently — they were jointly building up the correct prediction.

But all of these effects are small. The key conclusion is that no pair of features constitutes a tightly coupled sub-circuit: there is no pair for which removing A makes B suddenly critical (or vice versa). The top-10 features are encoding largely independent, separable contributions to the final prediction.

---

## 4. Discussion

### 4.1 Limitations

**Small dataset, high geography concentration.** With only 22 examples, and 13 of them from the same prompt template (two-shot capital city), all findings about geography should be interpreted as findings about *this prompt format* rather than about capital city knowledge in general. Different seed pairs might activate different circuits. The 7 attribute and 2 history examples are too few to draw strong conclusions from.

**Linear attribution is not causal.** DLA scores measure how each feature's output aligns with the correct-vs-foil direction at a specific activation state. This is a first-order linear approximation. It does not account for nonlinear downstream interactions or compensatory mechanisms that activate when a feature is removed. The frequent disagreement between DLA rankings and ablation effects (e.g., Head 0 has the highest DLA magnitude at 2.06 but a causal ablation effect of just +0.0002) is a reminder that high attribution does not imply causal importance.

**Zero ablation is unnatural.** Setting an activation to exactly zero is an intervention the model was never trained to handle. It may activate compensatory mechanisms elsewhere, inflating or deflating measured effects in ways that do not reflect the feature's normal role. Mean ablation (replacing with the dataset-average activation) would be a cleaner baseline, though computationally heavier.

**Scope of layers analyzed.** We only scored features in MLP layers 6–8 and attention layers 9–10. Earlier layers (responsible for syntactic processing and pattern recognition) and later layers (layer 11, the final MLP) were excluded. The top-10 features found here may not include the most important early-layer components.

**Only two fact categories with meaningful signal.** The history category contains just two examples, making its per-category statistics unreliable. Conclusions about history specialization should be disregarded.

### 4.2 Safety framing

**Distributed encoding is both good and bad for interpretability.**  
The near-additive independence finding (Finding 3) means there is no single small "factual recall circuit" that could be cleanly isolated and analyzed or edited. The information is spread across many components. This is good in one sense — the model is robust to single-component failures — but bad for interpretability: there is no small bottleneck you could patch to correct a factual error or prevent factual confabulation.

**Feature competition explains controlled uncertainty.**  
The promoter/suppressor structure (Finding 2) shows that model uncertainty is not just noise — it is the result of genuine internal competition between representations. The suppressor features are encoding the "runner-up" answer and actively pushing the model toward it. This has implications for factual reliability: if the suppressor features are stronger in some contexts (e.g., different prompt phrasing), they could flip the output from correct to incorrect without any obvious external change. Understanding which contexts strengthen or weaken the suppressor group is a direction worth exploring for understanding hallucination.

**Category specialization means attribution does not generalize.**  
Finding 1 shows that the features identified as most important for capital-city prompts are nearly inert for physics/science prompts. This means that interpretability findings on one task class cannot be assumed to transfer to another, even within the broad label of "factual recall." Any audit of a model's factual behavior would need to be conducted separately for each fact category.

### 4.3 Future work

**Attention pattern analysis.** The two most important geography promoters are attention heads (L10/H6 and L9/H2). The natural next question is: what are these heads attending to? Specifically, are they attending to the seed completions in the prompt (an induction/copying mechanism) or to the target country token (a more abstract retrieval mechanism)? Visualizing their attention patterns on the 13 geography examples would clarify whether the "geography circuit" is pattern-copying or content-addressing.

**Mean ablation.** Replacing zero ablation with mean ablation would give a cleaner causal estimate of each feature's importance, reducing the influence of compensatory mechanisms triggered by unnatural zero-activations.

**Category-specific attribution.** Running the full attribution pipeline restricted to the 7 attribute examples would reveal a completely different set of top features — likely ones encoding physics/chemistry knowledge in earlier MLP layers. Comparing the geography circuit and the attributes circuit would reveal whether GPT-2 Small uses distinct mechanisms for different fact types or a shared retrieval mechanism with fact-type-specific features feeding into it.

**Larger interaction search.** Our compositionality test covered only 45 pairs (top-10 features). Extending to top-50 features (1225 pairs) and using mean ablation might reveal stronger interactions, particularly among features in the same layer that were not individually prominent.

**Prompt sensitivity.** The cap2() prompt format was specifically engineered to produce sweet-spot confidence levels. Testing the same 22 facts with varied prompt formats (different seed countries, one-shot instead of two-shot, Q&A format) would reveal whether the same features activate or whether the circuit is highly format-sensitive.

---

## Summary of experimental outputs

| File | Description |
|---|---|
| `factual_recall_dataset.json` | 22 validated sweet-spot examples |
| `validation_report.txt` | Dataset statistics and pass/fail breakdown |
| `top_features.json` | Top-20 features ranked by DLA, with per-example attribution scores |
| `ablation_results.json` | Single-feature ablation drops for top-10 features, by category |
| `experiment3_compositionality.json` | Pairwise interaction terms for all 45 feature pairs |
| `experiment4_specialization.json` | Per-category mean DLA for top-10 features |
| `experiment1_feature_importance.png` | DLA bar chart and top-10 frequency plot |
| `experiment2_ablation.png` | Ablation drops by feature and category |
| `experiment3_compositionality.png` | 10×10 interaction heatmap + top-pair bar chart |
| `experiment4_specialization.png` | Feature×category heatmap + grouped bar chart |

**Scripts:**
- `create_dataset.py` → generates 50 candidate prompts
- `validate_dataset.py` → filters to 22 sweet-spot examples
- `feature_experiments.py` → Experiments 1 (DLA) and 2 (ablation)
- `composition_experiments.py` → Experiments 3 (pairwise) and 4 (specialization)
