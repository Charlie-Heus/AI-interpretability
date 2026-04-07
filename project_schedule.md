# Project Schedule — Feature Composition in GPT-2 Small

## Days 1-2: Setup & Dataset Creation — COMPLETE
- [x] Install tools (TransformerLens, circuitsvis, plotly)
- [x] `create_dataset.py` — 50 candidates across geography/history/attributes
- [x] `validate_dataset.py` — filters to sweet-spot examples (correct >50%, foil >10%)
- [x] `factual_recall_dataset.json` — 22 validated examples (44% pass rate among candidates; ~7-8% of all 300+ prompts tested)
- [x] `validation_report.txt` — human-readable statistics and per-example breakdown

## Days 3-4: Feature Extraction & Analysis — IN PROGRESS
- [x] `feature_experiments.py` written — Experiment 1 (DLA attribution) and Experiment 2 (zero-ablation)
- [x] Run `python3 feature_experiments.py` to completion
- [x] `top_features.json` generated
- [x] `ablation_results.json` generated
- [x] `experiment1_feature_importance.png` generated
- [x] `experiment2_ablation.png` generated

## Days 5-6: Feature Composition Analysis — COMPLETE
- [x] Experiment 3: pairwise feature interaction (A alone, B alone, A+B — test compositionality)
- [x] Experiment 4: feature specialization heatmap (top-10 features × fact type activation)

## Days 7-8: Visualization & Writeup — NOT STARTED
- [ ] Composition matrix visualization (heatmap of pairwise interaction effects)
- [ ] Feature specialization visualization (grouped bar chart by fact type)
- [ ] Technical writeup: "Feature Composition in Factual Recall: A Preliminary Investigation"
  - [ ] Introduction (why feature composition matters for AI safety)
  - [ ] Methods (model, dataset, experimental setup)
  - [ ] Results (3 findings with visualizations)
  - [ ] Discussion & future work (limitations, safety framing, extensions)
