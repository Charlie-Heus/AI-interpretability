"""
Feature composition experiments on GPT-2 Small.
  Experiment 1 — Attribution: identify top-20 neurons/heads driving logit difference.
  Experiment 2 — Ablation: zero-ablate top-10 features and measure accuracy drop.

Attribution method: direct logit attribution (DLA).
  For MLP neuron (l, n):  attr = act[n] * (W_out[l][n] · u_diff)
  For attn head (l, h):   attr = head_result[l][h] · u_diff
  where u_diff = W_U[:, correct_id] - W_U[:, incorrect_id]

Ablation: zero out the specific activation via a forward hook, then re-run.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASET_PATH    = "factual_recall_dataset.json"
INTERMEDIATE_DIR = Path("intermediate_results")
INTERMEDIATE_DIR.mkdir(exist_ok=True)

# ── Target components (adjust freely) ────────────────────────────────────────

MLP_LAYERS  = [6, 7, 8]          # layers whose MLP neurons to score
ATTN_LAYERS = [9, 10]            # layers whose attention heads to score

# ── Load model ────────────────────────────────────────────────────────────────

log.info("Loading GPT-2 Small via TransformerLens …")
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
tokenizer = model.tokenizer
N_HEADS   = model.cfg.n_heads   # 12
N_NEURONS = model.cfg.d_mlp     # 3072
log.info("Model ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        data = json.load(f)
    for item in data:
        item["category"] = _infer_category(item)
    return data


def _infer_category(item: dict) -> str:
    p = item["prompt"].lower()
    if "capital" in p:
        return "geography"
    if any(k in p for k in ("water", "atom", "metric", "freezing", "speed",
                             "saturn", "jupiter", "uranus", "planet")):
        return "attributes"
    return "history"


def tok_id(text: str) -> int:
    ids = tokenizer.encode(text)
    return ids[0] if ids else -1


# ── Experiment 1: Direct logit attribution ────────────────────────────────────

def _attribute_example(prompt: str, correct_str: str, incorrect_str: str
                        ) -> tuple[dict, dict]:
    """
    Returns two dicts keyed by (layer, idx):
      mlp_attrs  — per-neuron DLA scores
      attn_attrs — per-head DLA scores
    """
    c_id = tok_id(correct_str)
    i_id = tok_id(incorrect_str)

    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # Unembedding direction for the logit difference
    W_U    = model.W_U.detach()                       # [d_model, vocab]
    u_diff = W_U[:, c_id] - W_U[:, i_id]             # [d_model]

    # MLP neurons
    mlp_attrs: dict[tuple, float] = {}
    for layer in MLP_LAYERS:
        act = cache[f"blocks.{layer}.mlp.hook_post"][0, -1, :]   # [d_mlp]
        W_out = model.blocks[layer].mlp.W_out.detach()            # [d_mlp, d_model]
        contrib = (W_out @ u_diff) * act                          # [d_mlp]
        for n in range(N_NEURONS):
            mlp_attrs[(layer, n)] = contrib[n].item()

    # Attention heads — compute per-head output via hook_z @ W_O (equiv. to hook_result)
    attn_attrs: dict[tuple, float] = {}
    for layer in ATTN_LAYERS:
        z    = cache[f"blocks.{layer}.attn.hook_z"][0, -1, :, :]           # [n_heads, d_head]
        W_O  = model.blocks[layer].attn.W_O.detach()                        # [n_heads, d_head, d_model]
        head_out = torch.einsum("hd,hdm->hm", z, W_O)                      # [n_heads, d_model]
        contrib  = head_out @ u_diff                                         # [n_heads]
        for h in range(N_HEADS):
            attn_attrs[(layer, h)] = contrib[h].item()

    return mlp_attrs, attn_attrs


def run_experiment1(dataset: list[dict]) -> list[dict]:
    log.info("=" * 60)
    log.info("Experiment 1 — Direct Logit Attribution")
    log.info("=" * 60)

    # Accumulators
    accum: dict[tuple, list[float]] = {}   # feature_key -> list of attribution values
    top10_counts: dict[str, int]    = {}   # str(key) -> times in top-10

    for i, item in enumerate(tqdm(dataset, desc="Exp1 attribution")):
        prompt        = item["prompt"]
        correct_str   = item["correct"]
        incorrect_str = item["incorrect"][0]

        log.info(f"  [{i+1:2d}/{len(dataset)}] {prompt[:55]}…")

        mlp_a, attn_a = _attribute_example(prompt, correct_str, incorrect_str)

        example_scores: dict[str, float] = {}
        for (layer, n), v in mlp_a.items():
            key = ("mlp", layer, n)
            accum.setdefault(key, []).append(v)
            example_scores[str(key)] = v
        for (layer, h), v in attn_a.items():
            key = ("attn", layer, h)
            accum.setdefault(key, []).append(v)
            example_scores[str(key)] = v

        # Count top-10 by absolute value for this example
        ranked = sorted(example_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        for _, (k, _) in zip(range(10), ranked):
            top10_counts[k] = top10_counts.get(k, 0) + 1

        # Intermediate checkpoint every 5 examples
        if (i + 1) % 5 == 0:
            ckpt = {
                "processed": i + 1,
                "top10_counts_so_far": {k: v for k, v in
                                        sorted(top10_counts.items(),
                                               key=lambda x: -x[1])[:50]}
            }
            with open(INTERMEDIATE_DIR / "exp1_checkpoint.json", "w") as f:
                json.dump(ckpt, f, indent=2)
            log.info(f"    → checkpoint saved ({i+1} examples done)")

    # Build ranked feature list
    feature_stats: list[dict] = []
    for key, vals in accum.items():
        kind, layer, idx = key
        label = f"Layer {layer}, {'Neuron' if kind == 'mlp' else 'Head'} {idx}"
        feature_stats.append({
            "key":                  str(key),
            "kind":                 kind,
            "layer":                layer,
            "index":                idx,
            "name":                 label,
            "average_attribution":  float(np.mean(vals)),
            "abs_avg":              float(abs(np.mean(vals))),
            "top10_frequency":      top10_counts.get(str(key), 0),
            "all_attributions":     [round(v, 6) for v in vals],
        })

    feature_stats.sort(key=lambda x: x["abs_avg"], reverse=True)
    top20 = feature_stats[:20]

    # Save
    out = {"top_20_features": top20, "total_examples": len(dataset)}
    with open("top_features.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved → top_features.json")

    _plot_experiment1(top20)

    log.info("\nTop-20 features:")
    log.info(f"  {'Rank':4s}  {'Feature':38s}  {'Avg attr':>10s}  {'Top-10 freq':>11s}")
    log.info("  " + "-" * 68)
    for rank, feat in enumerate(top20, 1):
        log.info(f"  {rank:4d}  {feat['name']:38s}  "
                 f"{feat['average_attribution']:+10.4f}  "
                 f"{feat['top10_frequency']:>11d}")

    return top20


def _plot_experiment1(top20: list[dict]) -> None:
    names  = [f["name"] for f in top20]
    values = [f["average_attribution"] for f in top20]
    freqs  = [f["top10_frequency"] for f in top20]
    colors = ["steelblue" if f["kind"] == "mlp" else "coral" for f in top20]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: average attribution
    ax = axes[0]
    ax.barh(range(len(names)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average contribution to logit diff (correct − incorrect)")
    ax.set_title("Average DLA Score")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(handles=[Patch(facecolor="steelblue", label="MLP Neuron"),
                        Patch(facecolor="coral",     label="Attention Head")],
              loc="lower right", fontsize=8)

    # Right: top-10 frequency
    ax2 = axes[1]
    ax2.barh(range(len(names)), freqs, color=colors, alpha=0.85)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel(f"# of examples where feature is in top-10 contributors (out of {len(top20[0]['all_attributions'])})")
    ax2.set_title("Top-10 Frequency")
    ax2.legend(handles=[Patch(facecolor="steelblue", label="MLP Neuron"),
                         Patch(facecolor="coral",     label="Attention Head")],
               loc="lower right", fontsize=8)

    fig.suptitle("Experiment 1: Top-20 Features by Direct Logit Attribution\n"
                 "(GPT-2 Small — factual recall dataset)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("experiment1_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → experiment1_feature_importance.png")


# ── Experiment 2: Activation ablation ─────────────────────────────────────────

def _run_with_zero_ablation(prompt: str, feature: dict
                             ) -> torch.Tensor:
    """Returns logits after zeroing out the specified feature's activation."""
    kind  = feature["kind"]
    layer = feature["layer"]
    idx   = feature["index"]
    tokens = model.to_tokens(prompt)

    if kind == "mlp":
        hook_name = f"blocks.{layer}.mlp.hook_post"
        def hook_fn(value, hook):
            value[:, :, idx] = 0.0
            return value
    else:  # attn head — zero hook_z so head_out = z @ W_O becomes 0
        hook_name = f"blocks.{layer}.attn.hook_z"
        def hook_fn(value, hook):
            value[:, :, idx, :] = 0.0
            return value

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    return logits


def _eval_on_dataset(dataset,
                     logits_fn) -> list[tuple[float, str]]:
    """
    Runs logits_fn on every example and returns (logit_diff, category).
    logit_diff = logit(correct) - logit(foil); positive means correct wins.
    logits_fn(item) must return a logits tensor.
    dataset may be a plain list or a tqdm-wrapped iterable.
    """
    results = []
    for item in dataset:
        c_id = tok_id(item["correct"])
        i_id = tok_id(item["incorrect"][0])
        with torch.no_grad():
            logits = logits_fn(item)
        logit_diff = (logits[0, -1, c_id] - logits[0, -1, i_id]).item()
        results.append((logit_diff, item["category"]))
    return results


def _logit_diff_summary(results: list[tuple[float, str]],
                        categories: list[str]) -> dict:
    overall = float(np.mean([d for d, _ in results]))
    by_cat  = {}
    for cat in categories:
        subset = [d for d, c in results if c == cat]
        by_cat[cat] = float(np.mean(subset)) if subset else 0.0
    return {"overall": overall, "by_category": by_cat}


def run_experiment2(dataset: list[dict], top20: list[dict]) -> dict:
    log.info("=" * 60)
    log.info("Experiment 2 — Feature Ablation")
    log.info("=" * 60)

    top10      = top20[:10]
    categories = sorted(set(item["category"] for item in dataset))

    # ── Baseline ────────────────────────────────────────────────────────────
    log.info("Computing baseline logit differences …")
    baseline_results = _eval_on_dataset(
        tqdm(dataset, desc="Baseline"),
        lambda item: model(model.to_tokens(item["prompt"]))
    )
    baseline = _logit_diff_summary(baseline_results, categories)
    log.info(f"  Baseline mean logit diff: {baseline['overall']:+.4f}")
    for cat in categories:
        log.info(f"    {cat}: {baseline['by_category'][cat]:+.4f}")

    # ── Per-feature ablation ─────────────────────────────────────────────────
    ablation_results: list[dict] = []

    for fi, feat in enumerate(tqdm(top10, desc="Ablation")):
        log.info(f"\n  Ablating [{fi+1}/10]: {feat['name']}")

        ablated_results = _eval_on_dataset(
            tqdm(dataset, desc=f"  {feat['name']}", leave=False),
            lambda item, f=feat: _run_with_zero_ablation(item["prompt"], f)
        )
        ablated = _logit_diff_summary(ablated_results, categories)

        drop_overall = baseline["overall"] - ablated["overall"]
        drop_by_cat  = {cat: baseline["by_category"][cat] - ablated["by_category"][cat]
                        for cat in categories}
        most_affected = max(drop_by_cat, key=lambda c: drop_by_cat[c])

        row = {
            "feature_name":                   feat["name"],
            "kind":                           feat["kind"],
            "layer":                          feat["layer"],
            "index":                          feat["index"],
            "baseline_mean_logit_diff":        round(baseline["overall"], 4),
            "ablated_mean_logit_diff":         round(ablated["overall"], 4),
            "logit_diff_drop":                 round(drop_overall, 4),
            "most_affected_category":          most_affected,
            "drop_by_category":                {c: round(v, 4) for c, v in drop_by_cat.items()},
            "baseline_logit_diff_by_category": {c: round(baseline["by_category"][c], 4) for c in categories},
            "ablated_logit_diff_by_category":  {c: round(ablated["by_category"][c], 4)  for c in categories},
        }
        ablation_results.append(row)
        log.info(f"    logit diff drop: {drop_overall:+.4f}  |  most affected: {most_affected}")
        for cat in categories:
            log.info(f"      {cat}: {drop_by_cat[cat]:+.4f}")

        # Checkpoint after each feature
        with open(INTERMEDIATE_DIR / "exp2_checkpoint.json", "w") as f:
            json.dump(ablation_results, f, indent=2)

    # ── Final output ─────────────────────────────────────────────────────────
    output = {
        "baseline_mean_logit_diff":           round(baseline["overall"], 4),
        "baseline_logit_diff_by_category":    {c: round(baseline["by_category"][c], 4) for c in categories},
        "ablation_results":                   ablation_results,
    }
    with open("ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    log.info("\nSaved → ablation_results.json")

    _plot_experiment2(ablation_results, categories)

    # Print summary table
    log.info("\nAblation summary table:")
    header = f"  {'Feature':38s}  {'Baseline LD':>11s}  {'Ablated LD':>10s}  {'Drop':>8s}  {'Most affected'}"
    log.info(header)
    log.info("  " + "-" * 85)
    for r in ablation_results:
        log.info(f"  {r['feature_name']:38s}  "
                 f"{r['baseline_mean_logit_diff']:+10.4f}  "
                 f"{r['ablated_mean_logit_diff']:+10.4f}  "
                 f"{r['logit_diff_drop']:+8.4f}  "
                 f"{r['most_affected_category']}")

    return output


def _plot_experiment2(ablation_results: list[dict], categories: list[str]) -> None:
    n_feats = len(ablation_results)
    n_cats  = len(categories)
    x       = np.arange(n_feats)
    width   = 0.75 / n_cats

    fig, axes = plt.subplots(2, 1, figsize=(16, 14))

    # ── Top panel: overall logit diff drop ───────────────────────────────────
    ax = axes[0]
    drops = [r["logit_diff_drop"] for r in ablation_results]
    bars  = ax.bar(x, drops, color="steelblue", alpha=0.85)
    for bar, val in zip(bars, drops):
        ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.06
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["feature_name"] for r in ablation_results],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean logit diff drop (correct − foil)")
    ax.set_title("Overall Logit Difference Drop per Ablated Feature")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # ── Bottom panel: breakdown by fact type ──────────────────────────────────
    ax2   = axes[1]
    cmap  = plt.cm.Set2(np.linspace(0, 1, n_cats))
    for ci, cat in enumerate(categories):
        drops_cat = [r["drop_by_category"].get(cat, 0) for r in ablation_results]
        offset    = (ci - n_cats / 2 + 0.5) * width
        ax2.bar(x + offset, drops_cat, width, label=cat.capitalize(),
                color=cmap[ci], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([r["feature_name"] for r in ablation_results],
                        rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Logit diff drop by fact type")
    ax2.set_title("Logit Difference Drop Broken Down by Fact Type")
    ax2.legend(title="Fact type", loc="upper right")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.suptitle("Experiment 2: Feature Ablation Results\n"
                 "(GPT-2 Small — factual recall dataset)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("experiment2_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → experiment2_ablation.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset = load_dataset()
    log.info(f"Loaded {len(dataset)} validated examples.")
    cat_counts = {}
    for item in dataset:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    for cat, n in sorted(cat_counts.items()):
        log.info(f"  {cat}: {n} examples")

    # ── Experiment 1 ─────────────────────────────────────────────────────────
    top20 = run_experiment1(dataset)

    # ── Experiment 2 ─────────────────────────────────────────────────────────
    ablation_output = run_experiment2(dataset, top20)

    log.info("\n" + "=" * 60)
    log.info("All experiments complete. Output files:")
    log.info("  top_features.json                  — ranked feature list")
    log.info("  ablation_results.json              — ablation summary")
    log.info("  experiment1_feature_importance.png — feature importance charts")
    log.info("  experiment2_ablation.png           — ablation bar charts")
    log.info("  intermediate_results/              — checkpoints")
    log.info("=" * 60)
