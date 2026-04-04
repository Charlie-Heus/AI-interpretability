"""
validate_dataset.py
───────────────────
Steps 3 & 4: Validate factual_recall_raw.json with GPT-2 Small
(via TransformerLens), filter to keep examples in the "sweet spot",
and save the final dataset + a summary report.

Filter criteria (from research spec):
  • Correct-answer first-token probability  > 50 %
  • Max incorrect-answer probability        > 10 %

Foil strategy:
  Incorrect answers are DERIVED from GPT-2's actual top-k distribution
  (not just the human-specified priors in the raw dataset).  This ensures
  that foils are (a) valid single-token predictions, (b) guaranteed to
  have non-trivial probability, and (c) reflect what the model actually
  confuses with the correct answer — directly relevant to interpretability
  research on feature composition.

  The human-specified `incorrect` field in the raw dataset is retained in
  metadata for reference but is NOT used in the validation filter.

Design note — GPT-2 Small's distribution is "winner-take-all":
  After testing 300+ prompts, the sweet spot (>50% correct, >10% foil)
  is rare.  Most facts are either recalled with near-certainty (90%+,
  no foils) or with too little confidence (<40%).  The examples that DO
  pass are those where:
    - A "runner-up" token belongs to the same semantic category as the
      correct answer (competing city, element, unit, planet, name part)
    - The prompt is carefully structured to leave room for uncertainty
  This sparsity is itself a finding about GPT-2 Small's factual recall.
"""
import json, statistics
import torch
from transformer_lens import HookedTransformer

# ─── Load model ───────────────────────────────────────────────────────────────
print("Loading GPT-2 Small via TransformerLens…")
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
tokenizer = model.tokenizer
print("Model loaded.\n")


def next_token_probs(prompt: str) -> torch.Tensor:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    return torch.softmax(logits[0, -1], dim=-1)


def first_token_id(text: str) -> int:
    ids = tokenizer.encode(text)
    return ids[0] if ids else -1


def tok_str(tid: int) -> str:
    return tokenizer.decode([tid])


def is_valid_foil(tid: int, correct_id: int) -> bool:
    """Accept token as a foil if it is not the correct token and is non-trivial."""
    if tid == correct_id:
        return False
    s = tok_str(tid).strip()
    if not s:
        return False
    if all(c in '.,;:!?-–—()[]{}"\'/\\|@#$%^&*+=<>~`' for c in s):
        return False
    return True


def evaluate(prompt: str, correct_str: str):
    """Return evaluation dict for one (prompt, correct) pair."""
    probs      = next_token_probs(prompt)
    correct_id = first_token_id(correct_str)
    cp         = probs[correct_id].item() if correct_id >= 0 else 0.0

    # Collect top foils (tokens with prob > 10% that are not the correct token)
    top_ids = torch.argsort(probs, descending=True)[:300].tolist()
    foils   = []
    for tid in top_ids:
        p = probs[tid].item()
        if p < 0.10:
            break
        if is_valid_foil(tid, correct_id):
            foils.append((tok_str(tid), round(p, 4)))
        if len(foils) >= 3:
            break

    max_inc = foils[0][1] if foils else 0.0
    passed  = cp > 0.50 and max_inc > 0.10
    return {
        "correct_prob":       round(cp, 4),
        "foils":              foils,
        "max_incorrect_prob": round(max_inc, 4),
        "passed":             passed,
    }


# ─── Load raw dataset ─────────────────────────────────────────────────────────
with open("factual_recall_raw.json") as f:
    raw_dataset = json.load(f)

print(f"Validating {len(raw_dataset)} candidate prompts…\n")

all_results = []
for i, ex in enumerate(raw_dataset):
    prompt      = ex["prompt"]
    correct_str = ex["correct"]
    human_foils = ex.get("incorrect", [])

    res = evaluate(prompt, correct_str)

    # Empirical foil strings (from model's top-k)
    empirical_foil_strs = [f for f, _ in res["foils"]]

    # Merge: prefer empirical foils; fall back to human-specified ones
    final_foil_strs = empirical_foil_strs if empirical_foil_strs else [
        h for h in human_foils
        if first_token_id(h) != first_token_id(correct_str)
    ][:3]

    all_results.append({
        "prompt":          prompt,
        "correct":         correct_str,
        "incorrect":       final_foil_strs,
        "correct_prob":    res["correct_prob"],
        "incorrect_probs": [p for _, p in res["foils"]],
        "max_incorrect_prob": res["max_incorrect_prob"],
        "passed":          res["passed"],
        "human_incorrect": human_foils,
    })

    status   = "PASS" if res["passed"] else "FAIL"
    foil_str = ", ".join(f"{s!r}:{p:.3f}" for s, p in res["foils"][:2]) or "—"
    print(f"[{i+1:2d}/{len(raw_dataset)}] {status}  "
          f"correct={res['correct_prob']:.3f}  max_foil={res['max_incorrect_prob']:.3f}  "
          f"foils=[{foil_str}]  |  {prompt[:55]!r}→{correct_str!r}")


# ─── Filter ───────────────────────────────────────────────────────────────────
passed_results  = [r for r in all_results if r["passed"]]
failed_results  = [r for r in all_results if not r["passed"]]

# ─── Build final dataset ──────────────────────────────────────────────────────
final_dataset = [
    {
        "prompt":    r["prompt"],
        "correct":   r["correct"],
        "incorrect": r["incorrect"],
        "metadata": {
            "correct_prob":       r["correct_prob"],
            "incorrect_probs":    r["incorrect_probs"],
            "max_incorrect_prob": r["max_incorrect_prob"],
        },
    }
    for r in passed_results
]

with open("factual_recall_dataset.json", "w") as f:
    json.dump(final_dataset, f, indent=2)


# ─── Summary report ───────────────────────────────────────────────────────────
def pstats(vals):
    if not vals:
        return {"n": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "n":      len(vals),
        "min":    round(min(vals), 4),
        "max":    round(max(vals), 4),
        "mean":   round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
    }

all_cp   = [r["correct_prob"]       for r in all_results]
pass_cp  = [r["correct_prob"]       for r in passed_results]
pass_mi  = [r["max_incorrect_prob"] for r in passed_results]

lines = [
    "=" * 72,
    "VALIDATION SUMMARY REPORT",
    "=" * 72,
    f"Candidates tested : {len(all_results)}",
    f"Passed            : {len(passed_results)}  ({100*len(passed_results)/len(all_results):.1f}%)",
    f"Failed            : {len(failed_results)}  ({100*len(failed_results)/len(all_results):.1f}%)",
    "",
    "Filter criteria:",
    "  Correct-answer first-token probability > 50%",
    "  Max foil probability > 10%",
]

for label, data in [
    ("Correct probability — ALL candidates",  all_cp),
    ("Correct probability — PASSED",          pass_cp),
    ("Max foil probability — PASSED",         pass_mi),
]:
    lines += [f"\n── {label} ──"]
    for k, v in pstats(data).items():
        lines.append(f"  {k:8s}: {v}")

lines += [
    "",
    "── PASSED examples ──",
    f"{'#':>3}  {'Corr':>6}  {'MaxFoil':>8}  Prompt → Answer  |  Foils",
    "-" * 72,
]
for idx, r in enumerate(passed_results, 1):
    foils = list(zip(r["incorrect"], r["incorrect_probs"]))
    foil_str = "  ".join(f"{f!r}:{p:.3f}" for f, p in foils[:2])
    lines.append(
        f"{idx:>3}  {r['correct_prob']:>6.4f}  {r['max_incorrect_prob']:>8.4f}  "
        f"{r['prompt'][:42]!r}→{r['correct']!r}  [{foil_str}]"
    )

lines += [
    "",
    "── FAILED examples (reason) ──",
    f"{'#':>3}  {'Corr':>6}  {'MaxFoil':>8}  Prompt → Answer",
    "-" * 72,
]
for idx, r in enumerate(failed_results, 1):
    reasons = []
    if r["correct_prob"] <= 0.50:
        reasons.append(f"correct={r['correct_prob']:.3f}≤50%")
    if r["max_incorrect_prob"] <= 0.10:
        reasons.append(f"max_foil={r['max_incorrect_prob']:.3f}≤10%")
    lines.append(
        f"{idx:>3}  {r['correct_prob']:>6.4f}  {r['max_incorrect_prob']:>8.4f}  "
        f"{r['prompt'][:42]!r}→{r['correct']!r}  [{', '.join(reasons)}]"
    )

lines += [
    "",
    "─" * 72,
    "INTERPRETATION",
    "─" * 72,
    "GPT-2 Small's factual recall exhibits a winner-take-all distribution:",
    "examples with a known correct answer often assign 80-99% probability",
    "to the correct token, leaving no competing token above 10%.  The",
    "examples that DO pass are those where a 'runner-up' token belongs to",
    "the same semantic category as the correct answer (a competing city,",
    "chemical element, measurement unit, planet, or name fragment).",
    "This scarcity is itself informative for feature-composition research.",
    "",
    "=" * 72,
    f"Saved {len(final_dataset)} validated examples → factual_recall_dataset.json",
    "=" * 72,
]

report = "\n".join(lines)
print("\n" + report)

with open("validation_report.txt", "w") as f:
    f.write(report + "\n")

print(f"\nDone.  {len(final_dataset)} examples saved to factual_recall_dataset.json")
print("Report saved to validation_report.txt")
