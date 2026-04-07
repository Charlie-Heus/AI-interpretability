"""
Feature composition experiments on GPT-2 Small — Days 5–6.
  Experiment 3 — Pairwise compositionality: for each pair of top-10 features,
                  measure whether their combined ablation effect is additive,
                  synergistic, or redundant.
  Experiment 4 — Feature specialization: heatmap of mean DLA attribution for
                  each of the top-10 features broken down by fact category.

Compositionality metric (Exp 3):
  interaction(A, B) = drop_AB − (drop_A + drop_B)
    > 0  →  synergistic  (together they matter more than the sum)
    ≈ 0  →  independent / additive
    < 0  →  redundant    (together they matter less — they overlap)

Experiment 4 requires no new model inference: it derives category-level means
from the all_attributions list already stored in top_features.json.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASET_PATH    = Path("factual_recall_dataset.json")
TOP_FEATURES_PATH = Path("top_features.json")
ABLATION_PATH   = Path("ablation_results.json")
INTERMEDIATE_DIR = Path("intermediate_results")
INTERMEDIATE_DIR.mkdir(exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────

log.info("Loading GPT-2 Small via TransformerLens …")
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
tokenizer = model.tokenizer
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


def logit_diff(logits: torch.Tensor, item: dict) -> float:
    """Scalar correct−foil logit difference at the last position."""
    c_id = tok_id(item["correct"])
    i_id = tok_id(item["incorrect"][0])
    return (logits[0, -1, c_id] - logits[0, -1, i_id]).item()


def _hook_name(feat: dict) -> str:
    if feat["kind"] == "mlp":
        return f"blocks.{feat['layer']}.mlp.hook_post"
    return f"blocks.{feat['layer']}.attn.hook_z"


def _make_combined_hook(feats_for_hook: list[dict]):
    """Return a single hook function that zeros out all listed features."""
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        for f in feats_for_hook:
            if f["kind"] == "mlp":
                value[:, :, f["index"]] = 0.0
            else:  # attn
                value[:, :, f["index"], :] = 0.0
        return value
    return hook_fn


def _run_with_ablations(prompt: str, feats: list[dict]) -> torch.Tensor:
    """Zero-ablate any number of features simultaneously; returns full logits."""
    tokens = model.to_tokens(prompt)

    # Group features by hook point to avoid duplicate hooks on the same point.
    hook_map: dict[str, list[dict]] = {}
    for f in feats:
        hook_map.setdefault(_hook_name(f), []).append(f)

    fwd_hooks = [
        (hname, _make_combined_hook(fs))
        for hname, fs in hook_map.items()
    ]
    with torch.no_grad():
        return model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)


def _baseline_logits(prompt: str) -> torch.Tensor:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        return model(tokens)


def _mean_logit_diff(dataset: list[dict],
                     logits_fn) -> dict[str, float]:
    """
    Run logits_fn on every example.
    Returns {'overall': ..., 'by_category': {cat: ...}}.
    """
    all_diffs: list[tuple[float, str]] = []
    for item in dataset:
        ld = logit_diff(logits_fn(item), item)
        all_diffs.append((ld, item["category"]))

    cats = sorted(set(c for _, c in all_diffs))
    by_cat = {
        cat: float(np.mean([d for d, c in all_diffs if c == cat]))
        for cat in cats
    }
    return {
        "overall":     float(np.mean([d for d, _ in all_diffs])),
        "by_category": by_cat,
    }


# ── Experiment 3: Pairwise Compositionality ───────────────────────────────────

def run_experiment3(dataset: list[dict],
                    top10:   list[dict],
                    ablation_data: dict) -> dict:
    log.info("=" * 60)
    log.info("Experiment 3 — Pairwise Compositionality")
    log.info("=" * 60)

    categories = sorted(set(item["category"] for item in dataset))
    n = len(top10)

    # ── Baseline and single-feature drops from Exp 2 ─────────────────────────
    baseline_overall = ablation_data["baseline_mean_logit_diff"]
    baseline_by_cat  = ablation_data["baseline_logit_diff_by_category"]

    single_drops_overall: dict[str, float] = {}
    single_drops_by_cat:  dict[str, dict[str, float]] = {}
    for row in ablation_data["ablation_results"]:
        key = row["feature_name"]
        single_drops_overall[key] = row["logit_diff_drop"]
        single_drops_by_cat[key]  = row["drop_by_category"]

    log.info(f"  Baseline overall logit diff: {baseline_overall:+.4f}")
    log.info(f"  Testing {n*(n-1)//2} feature pairs …")

    # ── Pairwise ablation ─────────────────────────────────────────────────────
    pairs = list(combinations(range(n), 2))
    pair_results: list[dict] = []

    # Try to resume from checkpoint
    ckpt_path = INTERMEDIATE_DIR / "exp3_checkpoint.json"
    done_set: set[tuple[int, int]] = set()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            pair_results = json.load(f)
        done_set = {(r["feat_i_idx"], r["feat_j_idx"]) for r in pair_results}
        log.info(f"  Resumed from checkpoint: {len(done_set)} pairs already done.")

    for i, j in tqdm(pairs, desc="Exp3 pairs"):
        if (i, j) in done_set:
            continue

        feat_a = top10[i]
        feat_b = top10[j]

        ablated = _mean_logit_diff(
            dataset,
            lambda item, fa=feat_a, fb=feat_b: _run_with_ablations(
                item["prompt"], [fa, fb]
            )
        )

        drop_ab_overall = baseline_overall - ablated["overall"]
        drop_ab_by_cat  = {
            cat: baseline_by_cat[cat] - ablated["by_category"][cat]
            for cat in categories
        }

        drop_a = single_drops_overall[feat_a["name"]]
        drop_b = single_drops_overall[feat_b["name"]]
        interaction_overall = drop_ab_overall - (drop_a + drop_b)

        interaction_by_cat = {
            cat: drop_ab_by_cat[cat] - (
                single_drops_by_cat[feat_a["name"]].get(cat, 0.0)
                + single_drops_by_cat[feat_b["name"]].get(cat, 0.0)
            )
            for cat in categories
        }

        row: dict[str, Any] = {
            "feat_i_idx":           i,
            "feat_j_idx":           j,
            "feat_i_name":          feat_a["name"],
            "feat_j_name":          feat_b["name"],
            "drop_ab_overall":      round(drop_ab_overall, 4),
            "drop_a_overall":       round(drop_a, 4),
            "drop_b_overall":       round(drop_b, 4),
            "interaction_overall":  round(interaction_overall, 4),
            "interaction_by_cat":   {c: round(v, 4) for c, v in interaction_by_cat.items()},
            "drop_ab_by_cat":       {c: round(v, 4) for c, v in drop_ab_by_cat.items()},
        }
        pair_results.append(row)
        done_set.add((i, j))

        log.info(
            f"  ({feat_a['name']}) × ({feat_b['name']}): "
            f"drop_A={drop_a:+.3f}  drop_B={drop_b:+.3f}  "
            f"drop_AB={drop_ab_overall:+.3f}  "
            f"interaction={interaction_overall:+.3f}"
        )

        # Checkpoint after every pair
        with open(ckpt_path, "w") as f:
            json.dump(pair_results, f, indent=2)

    # ── Build interaction matrix ──────────────────────────────────────────────
    matrix = np.full((n, n), np.nan)
    for r in pair_results:
        i, j = r["feat_i_idx"], r["feat_j_idx"]
        v = r["interaction_overall"]
        matrix[i, j] = v
        matrix[j, i] = v  # symmetric

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "features":          [f["name"] for f in top10],
        "baseline_overall":  baseline_overall,
        "pair_results":      pair_results,
        "interaction_matrix": [
            [None if np.isnan(v) else round(float(v), 4) for v in row]
            for row in matrix
        ],
    }
    with open("experiment3_compositionality.json", "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved → experiment3_compositionality.json")

    _plot_experiment3(matrix, [f["name"] for f in top10], pair_results, categories)

    # ── Summary ───────────────────────────────────────────────────────────────
    sorted_pairs = sorted(pair_results, key=lambda r: abs(r["interaction_overall"]),
                          reverse=True)
    log.info("\nTop-10 most interactive pairs (by |interaction|):")
    log.info(f"  {'Feature A':28s}  {'Feature B':28s}  {'Drop_A':>7s}  "
             f"{'Drop_B':>7s}  {'Drop_AB':>8s}  {'Interaction':>11s}")
    log.info("  " + "-" * 100)
    for r in sorted_pairs[:10]:
        log.info(
            f"  {r['feat_i_name']:28s}  {r['feat_j_name']:28s}  "
            f"{r['drop_a_overall']:+7.4f}  {r['drop_b_overall']:+7.4f}  "
            f"{r['drop_ab_overall']:+8.4f}  {r['interaction_overall']:+11.4f}"
        )

    return output


def _plot_experiment3(matrix:    np.ndarray,
                      feat_names: list[str],
                      pair_results: list[dict],
                      categories:  list[str]) -> None:
    n = len(feat_names)
    # Short names for axis labels
    short = [n.replace("Layer ", "L").replace(", Neuron ", "N").replace(", Head ", "H")
             for n in feat_names]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # ── Left: overall interaction heatmap ─────────────────────────────────────
    ax = axes[0]
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Interaction (drop_AB − drop_A − drop_B)")

    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_title("Feature Pair Interaction Matrix\n"
                 "(red = synergistic, blue = redundant)", fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > vmax * 0.6 else "black")

    # ── Right: top-10 pairs sorted by |interaction| ───────────────────────────
    ax2 = axes[1]
    top10_pairs = sorted(pair_results, key=lambda r: abs(r["interaction_overall"]),
                         reverse=True)[:10]
    pair_labels = [f"{r['feat_i_name'].replace('Layer ','L').replace(', ','/')}\n"
                   f"× {r['feat_j_name'].replace('Layer ','L').replace(', ','/')}"
                   for r in top10_pairs]
    interactions = [r["interaction_overall"] for r in top10_pairs]
    colors = ["tomato" if v > 0 else "steelblue" for v in interactions]
    bars = ax2.barh(range(len(top10_pairs)), interactions, color=colors, alpha=0.85)
    ax2.set_yticks(range(len(top10_pairs)))
    ax2.set_yticklabels(pair_labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Interaction term (drop_AB − drop_A − drop_B)")
    ax2.set_title("Top-10 Most Interactive Pairs\n"
                  "(red = synergistic, blue = redundant)", fontsize=10)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, interactions):
        xpos = val + (0.005 if val >= 0 else -0.005)
        ax2.text(xpos, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.3f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=7)

    fig.suptitle("Experiment 3: Pairwise Feature Compositionality\n"
                 "(GPT-2 Small — factual recall dataset)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("experiment3_compositionality.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → experiment3_compositionality.png")


# ── Experiment 4: Feature Specialization Heatmap ─────────────────────────────

def run_experiment4(dataset: list[dict], top10: list[dict]) -> dict:
    log.info("=" * 60)
    log.info("Experiment 4 — Feature Specialization Heatmap")
    log.info("=" * 60)
    log.info("  (No new model inference — using all_attributions from top_features.json)")

    categories = sorted(set(item["category"] for item in dataset))

    # Build category index list in same order as dataset (= same order as all_attributions)
    cat_order = [item["category"] for item in dataset]

    # For each feature, compute mean DLA per category
    specialization: list[dict] = []
    for feat in top10:
        attrs = feat["all_attributions"]  # len == len(dataset)
        assert len(attrs) == len(dataset), (
            f"Attribution count mismatch for {feat['name']}: "
            f"{len(attrs)} vs {len(dataset)}"
        )
        by_cat: dict[str, float] = {}
        for cat in categories:
            vals = [a for a, c in zip(attrs, cat_order) if c == cat]
            by_cat[cat] = float(np.mean(vals)) if vals else 0.0

        log.info(f"  {feat['name']:32s}: " +
                 "  ".join(f"{cat}={by_cat[cat]:+.3f}" for cat in categories))

        specialization.append({
            "feature_name":      feat["name"],
            "kind":              feat["kind"],
            "layer":             feat["layer"],
            "index":             feat["index"],
            "avg_attr_by_cat":   {c: round(v, 4) for c, v in by_cat.items()},
            "avg_attr_overall":  round(float(np.mean(attrs)), 4),
        })

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "categories":     categories,
        "features":       [s["feature_name"] for s in specialization],
        "specialization": specialization,
    }
    with open("experiment4_specialization.json", "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved → experiment4_specialization.json")

    _plot_experiment4(specialization, categories)

    return output


def _plot_experiment4(specialization: list[dict], categories: list[str]) -> None:
    n_feats = len(specialization)
    n_cats  = len(categories)

    # Build matrix: rows = features (top→bottom), cols = categories
    matrix = np.array([
        [s["avg_attr_by_cat"][cat] for cat in categories]
        for s in specialization
    ])

    feat_labels = [s["feature_name"] for s in specialization]
    cat_labels  = [c.capitalize() for c in categories]
    short_feat  = [l.replace("Layer ", "L").replace(", Neuron ", "/N").replace(", Head ", "/H")
                   for l in feat_labels]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: heatmap ─────────────────────────────────────────────────────────
    ax = axes[0]
    vmax = np.max(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean DLA attribution (correct − foil direction)")

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_title("Feature Specialization by Fact Type\n"
                 "(mean DLA attribution per category)", fontsize=10)
    ax.set_xlabel("Fact category")
    ax.set_ylabel("Feature (rank order from Exp 1)")

    # Annotate cells
    for i in range(n_feats):
        for j in range(n_cats):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > vmax * 0.65 else "black")

    # ── Right: grouped bar chart ───────────────────────────────────────────────
    ax2  = axes[1]
    x    = np.arange(n_feats)
    w    = 0.7 / n_cats
    cmap = plt.cm.Set2(np.linspace(0, 1, n_cats))

    for ci, cat in enumerate(categories):
        vals   = matrix[:, ci]
        offset = (ci - n_cats / 2 + 0.5) * w
        ax2.bar(x + offset, vals, w, label=cat.capitalize(), color=cmap[ci], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_feat, rotation=35, ha="right", fontsize=7)
    ax2.set_ylabel("Mean DLA attribution")
    ax2.set_title("Mean DLA Attribution by Fact Category\n"
                  "(grouped by feature, ordered by overall rank)", fontsize=10)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.legend(title="Fact type", fontsize=9)

    fig.suptitle("Experiment 4: Feature Specialization — Top-10 Features × Fact Category\n"
                 "(GPT-2 Small — factual recall dataset)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("experiment4_specialization.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → experiment4_specialization.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    dataset = load_dataset()
    log.info(f"Loaded {len(dataset)} validated examples.")
    cat_counts: dict[str, int] = {}
    for item in dataset:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    for cat, cnt in sorted(cat_counts.items()):
        log.info(f"  {cat}: {cnt} examples")

    with open(TOP_FEATURES_PATH) as f:
        top_features_data = json.load(f)
    top10 = top_features_data["top_20_features"][:10]
    log.info(f"\nUsing top-10 features from {TOP_FEATURES_PATH}:")
    for i, feat in enumerate(top10, 1):
        log.info(f"  {i:2d}. {feat['name']}  (abs_avg={feat['abs_avg']:.4f})")

    with open(ABLATION_PATH) as f:
        ablation_data = json.load(f)
    log.info(f"\nLoaded ablation baseline from {ABLATION_PATH}  "
             f"(baseline LD={ablation_data['baseline_mean_logit_diff']:+.4f})")

    # ── Experiment 3 ─────────────────────────────────────────────────────────
    exp3_out = run_experiment3(dataset, top10, ablation_data)

    # ── Experiment 4 ─────────────────────────────────────────────────────────
    exp4_out = run_experiment4(dataset, top10)

    log.info("\n" + "=" * 60)
    log.info("Days 5–6 experiments complete. Output files:")
    log.info("  experiment3_compositionality.json — pairwise interaction table")
    log.info("  experiment3_compositionality.png  — interaction heatmap + top pairs")
    log.info("  experiment4_specialization.json   — per-category attribution table")
    log.info("  experiment4_specialization.png    — specialization heatmap")
    log.info("=" * 60)
