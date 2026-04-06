# CLAUDE.md
# Project memory, rules, and workflow — read at the start of every session.

## Project Overview

Mechanistic interpretability research on GPT-2 Small. The goal is to study **feature composition** — how the model combines stored factual knowledge to produce next-token predictions — by building a curated dataset of factual recall examples where the correct answer has >50% probability but a semantically related wrong answer ("foil") still gets >10%. This tension between competing internal representations is what makes examples useful for attribution and ablation analysis.

- **Model**: GPT-2 Small (117M params) via TransformerLens
- **Stack**: Python, PyTorch, TransformerLens, matplotlib
- **Final dataset**: `factual_recall_dataset.json` — 22 validated examples (geography, history, attributes)
- **Key finding**: GPT-2 Small's factual recall is "winner-take-all" — ~7-8% of candidate prompts fall in the sweet spot; the rest are either near-certain (>80%) or too uncertain (<40%)

## Key Commands

```bash
# Step 1: Generate 50 candidate prompts (no model required)
python3 create_dataset.py
# → writes factual_recall_raw.json

# Step 2: Validate with GPT-2 Small (downloads ~500MB on first run, then cached)
python3 validate_dataset.py
# → writes factual_recall_dataset.json, validation_report.txt

# Step 3: Run feature attribution (Exp 1) and ablation (Exp 2)
python3 feature_experiments.py
# → writes top_features.json, ablation_results.json
# → writes experiment1_feature_importance.png, experiment2_ablation.png
# → checkpoints saved to intermediate_results/ every 5 examples

# Exploration / probe scripts (historical, safe to delete)
python3 probe_candidates.py   # v1 — 107 candidates
python3 probe_v3.py           # v3 — 79 additional candidates
# probe_v4.py, probe_v5.py follow same pattern
```

## Architecture

**Pipeline:**
```
create_dataset.py  →  factual_recall_raw.json  →  validate_dataset.py
                                                         |
                                          factual_recall_dataset.json
                                                         |
                                          feature_experiments.py
                                          (Exp 1: DLA attribution, Exp 2: ablation)
```

**Key files:**
- `create_dataset.py` — hardcoded 50 candidate prompts (geography/history/attributes); writes `factual_recall_raw.json`
- `validate_dataset.py` — runs GPT-2 on every candidate; filters to sweet-spot examples; derives empirical foils from model's top-k; writes final dataset + report
- `feature_experiments.py` — Experiment 1: direct logit attribution (DLA) across MLP layers 6-8 and attention heads 9-10; Experiment 2: zero-ablation of top-10 features by accuracy drop
- `probe_candidates.py`, `probe_v3.py`–`probe_v5.py` — exploration scripts used to discover working prompt formats; not part of the final pipeline
- `factual_recall_raw.json` — 50 candidate prompts (input to validator)
- `factual_recall_dataset.json` — 22 validated examples with metadata (correct_prob, incorrect_probs, max_incorrect_prob)
- `validation_report.txt` — human-readable statistics and per-example pass/fail breakdown
- `project_writeup.txt` — detailed narrative of all design decisions, issues encountered, and findings

**cap2() prompt format** (the core dataset trick):
```python
def cap2(a, ac, b, bc, target):
    return f"The capital of {a} is {ac}. The capital of {b} is {bc}. The capital of {target} is"
```
Two completed seed examples prime GPT-2 to recognize the pattern and continue it with the target capital at ~55-75% confidence. Without seeds, simple prompts ("The capital of France is") give <1% probability for the correct answer.

**DLA attribution formula** (`feature_experiments.py`):
- MLP neuron `(l, n)`: `attr = act[n] * (W_out[l][n] · u_diff)` where `u_diff = W_U[:, correct_id] - W_U[:, incorrect_id]`
- Attention head `(l, h)`: `attr = head_result[l][h] · u_diff`

## Important Context

**Space-prefixed tokens** — all `correct` and `incorrect` values must start with a space (e.g., `" Paris"` not `"Paris"`). GPT-2 BPE encodes `" Paris"` and `"Paris"` as different tokens; the space-prefixed form is what the model predicts in mid-sentence context. The `create_dataset.py` asserts this at build time.

**First-token-only evaluation** — multi-token answers (e.g., `" Saint Petersburg"`) are evaluated on their first token only (`" Saint"`). This is intentional and consistent throughout all scripts.

**Winner-take-all distribution** — GPT-2 Small is nearly bimodal on factual recall: facts it "knows" get 80-99% probability (no foils survive), facts it doesn't get <40%. The 50-75% sweet spot requires carefully engineered prompts; only ~7-8% of tested candidates pass. Date/year history facts structurally fail (probability spreads thinly across many nearby years — no single foil exceeds 10%).

**Empirical foils vs. human priors** — `validate_dataset.py` derives foils from the model's actual top-k distribution, not from the `incorrect` field in the raw dataset. Human-specified foils are often wrong (the model doesn't confuse European capitals with each other; it confuses Paris with Marseille). The `incorrect` field is retained in the final JSON for reference only.

**TransformerLens cache keys** — activation cache uses dotted path format: `blocks.{layer}.mlp.hook_post` (shape `[batch, seq, d_mlp]`), `blocks.{layer}.attn.hook_result` (shape `[batch, seq, n_heads, d_model]`). Always index `[0, -1, ...]` to get the last token position of the single example.

**Intermediate checkpoints** — `feature_experiments.py` writes checkpoints to `intermediate_results/exp1_checkpoint.json` and `exp2_checkpoint.json` after every 5 examples (Exp 1) or after each ablated feature (Exp 2). Safe to resume from these if a run is interrupted.

---

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for the relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
- **No Orphaned Code**: Never leave dead code, unused imports, or commented-out blocks
- **Type Safety**: Type hints on all Python function signatures. Pydantic models for structured data, not raw dicts.
- **Explicit over Implicit**: Prefer readable, obvious code over clever one-liners

---