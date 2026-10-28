#!/usr/bin/env python3
"""
analyze_results.py — Deep statistical analysis and visualization for WOAH 2026.

Generates 8 publication-quality figures covering:
  1.  TP/FP/FN/TN confusion breakdown per model × label
  2.  Precision / Recall / F1 heatmaps
  3.  Intent detection rate distribution (model vs gold)
  4.  Per-email label count distributions (how many intents per email?)
  5.  Error pattern analysis — where do models systematically disagree?
  6.  Flat vs hierarchical Δ heatmap (RQ2)
  7.  Model agreement matrix — pairwise agreement on every email
  8.  Over/under-detection radar chart per model

Usage:
  python analyze_results.py --results-dir results --gold gold_standard.json
  python analyze_results.py --results-dir results --gold gold_standard.json --out figures/
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# ── Config ────────────────────────────────────────────────────────────────────

EXPLICIT_LABELS = [
    "information_manipulation", "resource_acquisition",
    "trust_establishment", "call_to_action",
]
IMPLICIT_LABELS = [
    "authority", "social_proof", "liking_similarity_deception",
    "distraction", "commitment_integrity_reciprocation",
]
ALL_LABELS = EXPLICIT_LABELS + IMPLICIT_LABELS

LABEL_SHORT = {
    "information_manipulation":           "Info\nManip",
    "resource_acquisition":               "Resource\nAcq",
    "trust_establishment":                "Trust\nEstab",
    "call_to_action":                     "Call to\nAction",
    "authority":                          "Authority",
    "social_proof":                       "Social\nProof",
    "liking_similarity_deception":        "Liking/\nDecept",
    "distraction":                        "Distraction",
    "commitment_integrity_reciprocation": "Commit/\nRecip",
}

LABEL_FULL = {
    "information_manipulation":           "1.1 Information Manipulation",
    "resource_acquisition":               "1.2 Resource Acquisition",
    "trust_establishment":                "1.3 Trust Establishment",
    "call_to_action":                     "1.4 Call to Action",
    "authority":                          "2.1 Authority",
    "social_proof":                       "2.2 Social Proof",
    "liking_similarity_deception":        "2.3 Liking, Similarity & Deception",
    "distraction":                        "2.4 Distraction",
    "commitment_integrity_reciprocation": "2.5 Commitment/Integrity/Recip.",
}

MODEL_DISPLAY = {
    "claude_sonnet4":   "Claude Sonnet 4",
    "gpt4o":            "GPT-4o",
    "llama3_3_70b":     "Llama 3.3-70B",
}
MODEL_COLORS = {
    "claude_sonnet4":   "#534AB7",
    "gpt4o":            "#1D9E75",
    "llama3_3_70b":     "#EF9F27",
}

HUMAN_KAPPA = {"explicit": 0.577, "implicit": 0.111}

# Aesthetic palette
PALETTE = {
    "bg":        "#0F1117",
    "bg2":       "#1A1D27",
    "border":    "#2A2D3A",
    "text":      "#E8E9F0",
    "subtext":   "#8B8FA8",
    "tp":        "#1D9E75",
    "fp":        "#E24B4A",
    "fn":        "#EF9F27",
    "tn":        "#3B4066",
    "explicit":  "#7C6FD4",
    "implicit":  "#4AADA8",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg2"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["border"],
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gold(path):
    with open(path) as f:
        raw = json.load(f)
    gold = {}
    for eid, data in raw.items():
        labels = {}
        labels.update(data.get("explicit", {}))
        labels.update(data.get("implicit", {}))
        gold[str(eid)] = labels
    return gold


def load_results(results_dir):
    """Returns dict: (model, strategy) → {email_id: {label: bool}}"""
    runs = {}
    results_dir = Path(results_dir)
    for fpath in sorted(results_dir.glob("*.json")):
        if fpath.stem == "manifest":
            continue
        stem = fpath.stem  # e.g. claude_sonnet4_flat
        # parse model and strategy
        for strat in ("hierarchical", "flat"):
            if stem.endswith(f"_{strat}"):
                model = stem[: -(len(strat) + 1)]
                break
        else:
            continue
        with open(fpath) as f:
            data = json.load(f)
        annotations = {}
        for record in data:
            if "annotation_error" in record:
                continue
            eid = str(record.get("id", ""))
            ann = record.get("annotations", {})
            labels = {}
            labels.update(ann.get("explicit", {}))
            labels.update(ann.get("implicit", {}))
            annotations[eid] = labels
        runs[(model, strat)] = annotations
        print(f"  Loaded ({model}, {strat}): {len(annotations)} emails")
    return runs


def confusion(pred_dict, gold_dict):
    """Per-label TP/FP/FN/TN across all emails."""
    counts = {lbl: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for lbl in ALL_LABELS}
    eids = set(pred_dict) & set(gold_dict)
    for eid in eids:
        for lbl in ALL_LABELS:
            p = pred_dict[eid].get(lbl, False)
            g = gold_dict[eid].get(lbl, False)
            if p and g:     counts[lbl]["tp"] += 1
            elif p and not g: counts[lbl]["fp"] += 1
            elif not p and g: counts[lbl]["fn"] += 1
            else:           counts[lbl]["tn"] += 1
    return counts


def prf(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# ── Figure helpers ────────────────────────────────────────────────────────────

def save(fig, path, name):
    out = Path(path) / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def label_bar(ax, bars, fmt="{:.0f}", color=None):
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                fmt.format(h), ha="center", va="bottom",
                fontsize=7, color=color or PALETTE["subtext"])


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Stacked TP/FP/FN/TN per model × label (flat strategy)
# ─────────────────────────────────────────────────────────────────────────────

def fig_confusion_stacked(runs, gold, out):
    flat_runs = {m: a for (m, s), a in runs.items() if s == "flat"}
    models = [m for m in ("claude_sonnet4", "gpt4o", "llama3_3_70b") if m in flat_runs]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6),
                             facecolor=PALETTE["bg"])
    fig.suptitle("Confusion Breakdown per Label — Flat Strategy",
                 fontsize=13, fontweight="bold", y=1.01, color=PALETTE["text"])

    x = np.arange(len(ALL_LABELS))
    w = 0.6
    colors = [PALETTE["tp"], PALETTE["fp"], PALETTE["fn"], PALETTE["tn"]]
    keys   = ["tp", "fp", "fn", "tn"]
    labels = ["TP", "FP", "FN", "TN"]

    for ax, model in zip(axes, models):
        c = confusion(flat_runs[model], gold)
        bottoms = np.zeros(len(ALL_LABELS))
        for key, col, lbl in zip(keys, colors, labels):
            vals = np.array([c[l][key] for l in ALL_LABELS], dtype=float)
            ax.bar(x, vals, bottom=bottoms, color=col, label=lbl,
                   width=w, alpha=0.9, zorder=3)
            # label non-trivial segments
            for i, (v, b) in enumerate(zip(vals, bottoms)):
                if v >= 3:
                    ax.text(x[i], b + v / 2, str(int(v)), ha="center",
                            va="center", fontsize=7, fontweight="bold",
                            color=PALETTE["bg"])
            bottoms += vals

        # vertical divider between explicit/implicit
        ax.axvline(3.5, color=PALETTE["border"], lw=1.5, ls="--", zorder=4)
        ax.text(1.5, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 55,
                "EXPLICIT", ha="center", fontsize=8, color=PALETTE["explicit"], alpha=0.8)
        ax.text(6.5, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 55,
                "IMPLICIT", ha="center", fontsize=8, color=PALETTE["implicit"], alpha=0.8)

        ax.set_title(MODEL_DISPLAY.get(model, model), color=PALETTE["text"])
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_SHORT[l] for l in ALL_LABELS],
                           fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Email count (n=60)")
        ax.set_ylim(0, 65)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.2,
                      facecolor=PALETTE["bg2"], edgecolor=PALETTE["border"])

    fig.tight_layout()
    return save(fig, out, "fig1_confusion_stacked.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Precision / Recall / F1 heatmaps (both strategies)
# ─────────────────────────────────────────────────────────────────────────────

def fig_prf_heatmap(runs, gold, out):
    strategies = ["flat", "hierarchical"]
    models     = ["claude_sonnet4", "gpt4o", "llama3_3_70b"]
    metrics    = ["Precision", "Recall", "F1"]

    cmap = LinearSegmentedColormap.from_list(
        "prf", ["#1A1D27", "#4AADA8", "#E8E9F0"])

    fig, big_axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])
    fig.suptitle("Precision · Recall · F1 per Label", fontsize=13,
                 fontweight="bold", color=PALETTE["text"])

    for ax_col, strat in zip(big_axes, strategies):
        ax_col.set_visible(False)
        # create 3 sub-axes manually
        pos = ax_col.get_position()
        sub_width = pos.width / 3.2
        gap = pos.width * 0.05

        for mi, metric in enumerate(metrics):
            left = pos.x0 + mi * (sub_width + gap)
            sub_ax = fig.add_axes([left, pos.y0, sub_width, pos.height])

            matrix = np.zeros((len(models), len(ALL_LABELS)))
            for ri, model in enumerate(models):
                key = (model, strat)
                if key not in runs:
                    continue
                c = confusion(runs[key], gold)
                for ci, lbl in enumerate(ALL_LABELS):
                    p, r, f = prf(c[lbl])
                    matrix[ri, ci] = [p, r, f][mi]

            im = sub_ax.imshow(matrix, aspect="auto", cmap=cmap,
                               vmin=0, vmax=1)
            sub_ax.set_xticks(range(len(ALL_LABELS)))
            sub_ax.set_xticklabels(
                [LABEL_SHORT[l] for l in ALL_LABELS],
                fontsize=6.5, rotation=40, ha="right")
            if mi == 0:
                sub_ax.set_yticks(range(len(models)))
                sub_ax.set_yticklabels(
                    [MODEL_DISPLAY.get(m, m) for m in models], fontsize=8)
            else:
                sub_ax.set_yticks([])
            sub_ax.set_title(
                f"{metric}\n({strat})", fontsize=9, color=PALETTE["text"])

            # annotate cells
            for ri in range(len(models)):
                for ci in range(len(ALL_LABELS)):
                    v = matrix[ri, ci]
                    sub_ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 0.6 else PALETTE["bg"],
                                fontweight="bold")

            # explicit/implicit divider
            sub_ax.axvline(3.5, color=PALETTE["border"], lw=1.5, ls="--")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, out, "fig2_prf_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Intent detection rates: model vs gold (flat)
# ─────────────────────────────────────────────────────────────────────────────

def fig_detection_rates(runs, gold, out):
    flat_runs = {m: a for (m, s), a in runs.items() if s == "flat"}
    models = [m for m in ("claude_sonnet4", "gpt4o", "llama3_3_70b") if m in flat_runs]
    n = 60

    gold_rates = {lbl: sum(1 for g in gold.values() if g.get(lbl)) / n
                  for lbl in ALL_LABELS}

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    fig.suptitle("Intent Detection Rate: Models vs Gold Standard (Flat Strategy)",
                 fontsize=13, fontweight="bold", color=PALETTE["text"])

    x = np.arange(len(ALL_LABELS))
    total = len(models) + 1  # +1 for gold
    w = 0.7 / total
    offsets = np.linspace(-(total - 1) / 2 * w, (total - 1) / 2 * w, total)

    # gold bars
    gold_vals = [gold_rates[l] for l in ALL_LABELS]
    ax.bar(x + offsets[0], gold_vals, width=w,
           color="#E8E9F0", alpha=0.35, label="Gold (human)", zorder=3,
           hatch="//", edgecolor=PALETTE["subtext"], linewidth=0.5)

    for i, model in enumerate(models):
        pred_rates = {lbl: sum(1 for e in flat_runs[model].values()
                               if e.get(lbl)) / n for lbl in ALL_LABELS}
        vals = [pred_rates[l] for l in ALL_LABELS]
        bars = ax.bar(x + offsets[i + 1], vals, width=w,
                      color=MODEL_COLORS[model], alpha=0.85,
                      label=MODEL_DISPLAY[model], zorder=3)

    ax.axvline(3.5, color=PALETTE["border"], lw=1.5, ls="--", zorder=4)
    ax.text(1.75, 1.04, "EXPLICIT", ha="center", fontsize=9,
            color=PALETTE["explicit"], transform=ax.get_xaxis_transform())
    ax.text(6.5, 1.04, "IMPLICIT", ha="center", fontsize=9,
            color=PALETTE["implicit"], transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_FULL[l] for l in ALL_LABELS],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Detection rate (proportion of 60 emails)")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.15, facecolor=PALETTE["bg2"],
              edgecolor=PALETTE["border"])

    fig.tight_layout()
    return save(fig, out, "fig3_detection_rates.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Per-email label count distributions
# ─────────────────────────────────────────────────────────────────────────────

def fig_label_count_dist(runs, gold, out):
    flat_runs = {m: a for (m, s), a in runs.items() if s == "flat"}
    models = [m for m in ("claude_sonnet4", "gpt4o", "llama3_3_70b") if m in flat_runs]

    def counts_per_email(ann, layer_labels):
        return [sum(1 for l in layer_labels if ann.get(eid, {}).get(l, False))
                for eid in sorted(gold.keys())]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor=PALETTE["bg"])
    fig.suptitle("Per-Email Intent Count Distributions", fontsize=13,
                 fontweight="bold", color=PALETTE["text"])

    layers = [
        ("All labels (0–9)",    ALL_LABELS,      range(10)),
        ("Explicit only (0–4)", EXPLICIT_LABELS, range(5)),
        ("Implicit only (0–5)", IMPLICIT_LABELS, range(6)),
    ]

    gold_all   = counts_per_email(gold, ALL_LABELS)
    gold_exp   = counts_per_email(gold, EXPLICIT_LABELS)
    gold_imp   = counts_per_email(gold, IMPLICIT_LABELS)
    gold_by_layer = [gold_all, gold_exp, gold_imp]

    sources = [("Gold", "#E8E9F0", gold)] + \
              [(MODEL_DISPLAY[m], MODEL_COLORS[m], flat_runs[m]) for m in models]

    for col, (title, layer_labels, bins) in enumerate(layers):
        for row, (name, color, ann) in enumerate(sources):
            if row >= 2 and col >= 3:
                continue
            ax = axes[row % 2][col if row < 2 else col]
            if row == 0:
                axes[0][col].set_title(title, fontsize=9, color=PALETTE["text"])

        # violin/hist per column
        all_data = {}
        for name, color, ann in sources:
            all_data[name] = counts_per_email(ann, layer_labels)

        ax = axes[0][col]
        ax.set_title(title, fontsize=9, color=PALETTE["text"])
        positions = range(len(sources))
        parts = ax.violinplot(
            [all_data[n] for n, _, _ in sources],
            positions=list(positions), showmedians=True, showextrema=True)
        for i, (pc, (_, color, _)) in enumerate(zip(parts["bodies"], sources)):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color(PALETTE["text"])
        parts["cmaxes"].set_color(PALETTE["subtext"])
        parts["cmins"].set_color(PALETTE["subtext"])
        parts["cbars"].set_color(PALETTE["subtext"])
        ax.set_xticks(list(positions))
        ax.set_xticklabels([n for n, _, _ in sources], fontsize=7, rotation=20)
        ax.set_ylabel("# labels per email")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

        # mean lines
        for i, (name, color, _) in enumerate(sources):
            mean = np.mean(all_data[name])
            ax.hlines(mean, i - 0.15, i + 0.15, colors=color, lw=2, zorder=5)
            ax.text(i + 0.18, mean, f"{mean:.1f}", fontsize=7,
                    color=color, va="center")

    # 4th column: mean label count comparison bar chart
    for row in range(2):
        axes[row][3].set_visible(False)
    ax4 = fig.add_subplot(1, 4, 4, facecolor=PALETTE["bg2"])
    ax4.set_facecolor(PALETTE["bg2"])

    source_names = [n for n, _, _ in sources]
    all_means  = [np.mean(counts_per_email(ann, ALL_LABELS))   for _, _, ann in sources]
    exp_means  = [np.mean(counts_per_email(ann, EXPLICIT_LABELS)) for _, _, ann in sources]
    imp_means  = [np.mean(counts_per_email(ann, IMPLICIT_LABELS)) for _, _, ann in sources]

    y = np.arange(len(sources))
    h = 0.25
    colors_list = [c for _, c, _ in sources]

    ax4.barh(y + h,   all_means, h, color=colors_list, alpha=0.9, label="All")
    ax4.barh(y,       exp_means, h, color=colors_list, alpha=0.6, hatch="//",
             edgecolor="white", linewidth=0.3, label="Explicit")
    ax4.barh(y - h,   imp_means, h, color=colors_list, alpha=0.4,
             edgecolor="white", linewidth=0.3, label="Implicit")

    ax4.set_yticks(y)
    ax4.set_yticklabels(source_names, fontsize=8)
    ax4.set_xlabel("Mean # labels per email")
    ax4.set_title("Mean label counts", fontsize=9, color=PALETTE["text"])
    ax4.legend(fontsize=7, framealpha=0.2)
    ax4.grid(axis="x", zorder=0)
    ax4.set_axisbelow(True)

    fig.tight_layout()
    return save(fig, out, "fig4_label_count_dist.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Error pattern: per-email disagreement heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_error_heatmap(runs, gold, out):
    flat_runs = {m: a for (m, s), a in runs.items() if s == "flat"}
    models = [m for m in ("claude_sonnet4", "gpt4o", "llama3_3_70b") if m in flat_runs]
    eids   = sorted(gold.keys(), key=lambda x: int(x) if x.isdigit() else x)

    # For each (model, email, label): 1=correct, -1=FP, -2=FN, 0=TN
    def error_matrix(model_ann):
        mat = np.zeros((len(eids), len(ALL_LABELS)))
        for ei, eid in enumerate(eids):
            for li, lbl in enumerate(ALL_LABELS):
                p = model_ann.get(eid, {}).get(lbl, False)
                g = gold.get(eid, {}).get(lbl, False)
                if p and g:       mat[ei, li] =  1   # TP
                elif p and not g: mat[ei, li] = -1   # FP
                elif not p and g: mat[ei, li] = -2   # FN
                else:             mat[ei, li] =  0   # TN
        return mat

    cmap = LinearSegmentedColormap.from_list(
        "err", [PALETTE["fn"], PALETTE["bg2"], PALETTE["tn"], PALETTE["tp"]])

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 9),
                             facecolor=PALETTE["bg"])
    fig.suptitle("Per-Email Error Pattern (Flat Strategy)\nTP=green · FP=red · FN=orange · TN=dark",
                 fontsize=12, fontweight="bold", color=PALETTE["text"])

    for ax, model in zip(axes, models):
        mat = error_matrix(flat_runs[model])
        im = ax.imshow(mat.T, aspect="auto", cmap=cmap, vmin=-2, vmax=1,
                       interpolation="nearest")
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=10,
                     color=PALETTE["text"])
        ax.set_yticks(range(len(ALL_LABELS)))
        ax.set_yticklabels([LABEL_SHORT[l].replace("\n", " ")
                            for l in ALL_LABELS], fontsize=7)
        ax.set_xlabel("Email index", fontsize=8)
        ax.axhline(3.5, color=PALETTE["border"], lw=2, ls="--")
        ax.text(-3, 1.8, "EXPLICIT", fontsize=7, color=PALETTE["explicit"],
                rotation=90, va="center")
        ax.text(-3, 6.5, "IMPLICIT", fontsize=7, color=PALETTE["implicit"],
                rotation=90, va="center")

        # aggregate error counts on right side
        for li, lbl in enumerate(ALL_LABELS):
            col = mat[:, li]
            fp = int((col == -1).sum())
            fn = int((col == -2).sum())
            ax.text(len(eids) + 0.5, li, f"FP:{fp} FN:{fn}",
                    fontsize=6, va="center", color=PALETTE["subtext"])

        ax.set_xlim(-4, len(eids) + 8)

    fig.tight_layout()
    return save(fig, out, "fig5_error_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Flat vs Hierarchical delta heatmap (RQ2)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rq2_delta(runs, gold, out):
    models = ["claude_sonnet4", "gpt4o", "llama3_3_70b"]

    def kappa(pred_ann, lbl):
        eids = list(set(pred_ann) & set(gold))
        if not eids:
            return 0.0
        p_arr = np.array([pred_ann[e].get(lbl, False) for e in eids], dtype=float)
        g_arr = np.array([gold[e].get(lbl, False) for e in eids], dtype=float)
        n = len(eids)
        po = np.mean(p_arr == g_arr)
        pe = np.mean(p_arr) * np.mean(g_arr) + (1 - np.mean(p_arr)) * (1 - np.mean(g_arr))
        return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0

    delta = np.zeros((len(models), len(ALL_LABELS)))
    for mi, model in enumerate(models):
        flat_key = (model, "flat")
        hier_key = (model, "hierarchical")
        if flat_key not in runs or hier_key not in runs:
            continue
        for li, lbl in enumerate(ALL_LABELS):
            kf = kappa(runs[flat_key], lbl)
            kh = kappa(runs[hier_key], lbl)
            delta[mi, li] = kh - kf

    cmap = LinearSegmentedColormap.from_list(
        "delta", [PALETTE["fp"], PALETTE["bg2"], PALETTE["tp"]])

    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor=PALETTE["bg"])
    fig.suptitle("RQ2 — Δκ: Hierarchical − Flat (per model × label)\nGreen = hierarchical improves  ·  Red = flat is better",
                 fontsize=12, fontweight="bold", color=PALETTE["text"])

    im = ax.imshow(delta, aspect="auto", cmap=cmap,
                   vmin=-0.4, vmax=0.4, interpolation="nearest")

    for mi in range(len(models)):
        for li in range(len(ALL_LABELS)):
            v = delta[mi, li]
            ax.text(li, mi, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if abs(v) < 0.2 else PALETTE["bg"])

    ax.set_xticks(range(len(ALL_LABELS)))
    ax.set_xticklabels([LABEL_SHORT[l].replace("\n", " ") for l in ALL_LABELS],
                       fontsize=8, rotation=25, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=9)
    ax.axvline(3.5, color=PALETTE["border"], lw=2, ls="--")
    ax.text(1.75, -0.7, "EXPLICIT", ha="center", fontsize=8,
            color=PALETTE["explicit"], transform=ax.transData)
    ax.text(6.5, -0.7, "IMPLICIT", ha="center", fontsize=8,
            color=PALETTE["implicit"], transform=ax.transData)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Δκ", color=PALETTE["text"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["subtext"])

    fig.tight_layout()
    return save(fig, out, "fig6_rq2_delta.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Inter-model agreement per email (where do models always agree/disagree?)
# ─────────────────────────────────────────────────────────────────────────────

def fig_model_agreement(runs, gold, out):
    flat_runs = {m: a for (m, s), a in runs.items() if s == "flat"}
    models = [m for m in ("claude_sonnet4", "gpt4o", "llama3_3_70b") if m in flat_runs]
    eids   = sorted(gold.keys(), key=lambda x: int(x) if x.isdigit() else x)

    # consensus score per email: fraction of (model, label) pairs that agree with gold
    consensus = []
    all_agree_with_gold = []
    for eid in eids:
        total = 0; correct = 0
        model_votes = []
        for lbl in ALL_LABELS:
            g = gold.get(eid, {}).get(lbl, False)
            votes = [flat_runs[m].get(eid, {}).get(lbl, False) for m in models]
            majority = sum(votes) >= 2
            correct += int(majority == g)
            total   += 1
            model_votes.append(votes)
        consensus.append(correct / total)
        # agreement: all 3 models same AND correct
        ag = sum(1 for lbl in ALL_LABELS
                 if len(set(flat_runs[m].get(eid, {}).get(lbl, False)
                            for m in models)) == 1
                 and flat_runs[models[0]].get(eid, {}).get(lbl, False) ==
                     gold.get(eid, {}).get(lbl, False))
        all_agree_with_gold.append(ag / len(ALL_LABELS))

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), facecolor=PALETTE["bg"])
    fig.suptitle("Inter-Model Agreement Analysis (Flat Strategy)",
                 fontsize=13, fontweight="bold", color=PALETTE["text"])

    # Top: consensus accuracy per email
    ax = axes[0]
    eid_nums = [int(e) if e.isdigit() else i for i, e in enumerate(eids)]
    colors_bar = [PALETTE["tp"] if v >= 0.7 else
                  PALETTE["fn"] if v >= 0.5 else
                  PALETTE["fp"] for v in consensus]
    bars = ax.bar(range(len(eids)), consensus, color=colors_bar, alpha=0.85, zorder=3)
    ax.axhline(np.mean(consensus), color=PALETTE["text"], lw=1.5, ls="--",
               label=f"Mean: {np.mean(consensus):.2f}")
    ax.set_xticks(range(0, len(eids), 5))
    ax.set_xticklabels(eids[::5], fontsize=7)
    ax.set_ylabel("Majority-vote accuracy\n(fraction of labels correct)")
    ax.set_title("Per-Email Majority-Vote Accuracy vs Gold", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)

    # Bottom: per-email unanimous correct rate
    ax2 = axes[1]
    ax2.fill_between(range(len(eids)), all_agree_with_gold,
                     alpha=0.4, color=PALETTE["implicit"])
    ax2.plot(range(len(eids)), all_agree_with_gold,
             color=PALETTE["implicit"], lw=1.5, zorder=3)
    ax2.axhline(np.mean(all_agree_with_gold), color=PALETTE["text"],
                lw=1.5, ls="--",
                label=f"Mean: {np.mean(all_agree_with_gold):.2f}")
    ax2.set_xticks(range(0, len(eids), 5))
    ax2.set_xticklabels(eids[::5], fontsize=7)
    ax2.set_ylabel("Fraction of labels where\nall 3 models agree AND correct")
    ax2.set_title("Unanimous Correct Agreement per Email", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.legend(fontsize=9, framealpha=0.2)
    ax2.grid(axis="y", zorder=0); ax2.set_axisbelow(True)

    fig.tight_layout()
    return save(fig, out, "fig7_model_agreement.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Over/under-detection radar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_over_under_detection(runs, gold, out):
    flat_runs = {m: a for (m, s), a in runs.items() if s == "flat"}
    models = [m for m in ("claude_sonnet4", "gpt4o", "llama3_3_70b") if m in flat_runs]
    n = len(gold)
    labels_radar = ALL_LABELS
    N = len(labels_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    gold_rate = np.array([
        sum(1 for g in gold.values() if g.get(l, False)) / n
        for l in labels_radar
    ])

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5),
                             subplot_kw=dict(polar=True),
                             facecolor=PALETTE["bg"])
    fig.suptitle("Over/Under-Detection Radar: Model Rate vs Gold Rate (Flat)",
                 fontsize=12, fontweight="bold", color=PALETTE["text"])

    for ax, model in zip(axes, models):
        pred_rate = np.array([
            sum(1 for e in flat_runs[model].values() if e.get(l, False)) / n
            for l in labels_radar
        ])
        delta = pred_rate - gold_rate  # positive = over-detection

        delta_closed = np.concatenate([delta, [delta[0]]])
        zero_closed  = np.zeros(N + 1)
        gold_closed  = np.concatenate([gold_rate, [gold_rate[0]]])
        pred_closed  = np.concatenate([pred_rate,  [pred_rate[0]]])

        ax.set_facecolor(PALETTE["bg2"])
        ax.spines["polar"].set_color(PALETTE["border"])

        # gold baseline
        ax.plot(angles, gold_closed, color=PALETTE["text"], lw=1.5,
                ls="--", alpha=0.5, label="Gold")
        ax.fill(angles, gold_closed, color=PALETTE["text"], alpha=0.05)

        # model
        ax.plot(angles, pred_closed, color=MODEL_COLORS[model], lw=2,
                label=MODEL_DISPLAY[model])
        ax.fill(angles, pred_closed, color=MODEL_COLORS[model], alpha=0.15)

        # color positive/negative delta
        for i in range(N):
            a1, a2 = angles[i], angles[(i + 1) % N]
            if delta[i] > 0.02:
                ax.fill([a1, a1, a2, a2], [gold_rate[i], pred_rate[i],
                                            pred_rate[(i + 1) % N], gold_rate[(i + 1) % N]],
                        color=PALETTE["fp"], alpha=0.3)
            elif delta[i] < -0.02:
                ax.fill([a1, a1, a2, a2], [pred_rate[i], gold_rate[i],
                                            gold_rate[(i + 1) % N], pred_rate[(i + 1) % N]],
                        color=PALETTE["fn"], alpha=0.3)

        short_labels = [LABEL_SHORT[l].replace("\n", " ") for l in labels_radar]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_labels, size=7, color=PALETTE["text"])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.set_title(MODEL_DISPLAY.get(model, model), pad=15,
                     color=PALETTE["text"], fontsize=10)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.15,
                  facecolor=PALETTE["bg2"])

        # annotate explicit/implicit ring
        ax.set_rticks([0.25, 0.5, 0.75])
        ax.grid(color=PALETTE["border"], alpha=0.5)

    fig.tight_layout()
    return save(fig, out, "fig8_radar_over_under.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WOAH 2026 — Deep result analysis")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gold",        default="gold_standard.json")
    parser.add_argument("--out",         default="figures")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  WOAH 2026 — Result Analysis")
    print("=" * 60)

    print("\nLoading data...")
    gold = load_gold(args.gold)
    runs = load_results(args.results_dir)
    print(f"  Gold: {len(gold)} emails | Runs: {len(runs)}")

    print("\nGenerating figures...")
    outputs = []
    outputs.append(fig_confusion_stacked(runs, gold, args.out))
    outputs.append(fig_prf_heatmap(runs, gold, args.out))
    outputs.append(fig_detection_rates(runs, gold, args.out))
    outputs.append(fig_label_count_dist(runs, gold, args.out))
    outputs.append(fig_error_heatmap(runs, gold, args.out))
    outputs.append(fig_rq2_delta(runs, gold, args.out))
    outputs.append(fig_model_agreement(runs, gold, args.out))
    outputs.append(fig_over_under_detection(runs, gold, args.out))

    print(f"\n{'=' * 60}")
    print(f"  ✓ {len(outputs)} figures saved to {args.out}/")
    print(f"{'=' * 60}")

    # Print a quick summary table
    print("\n  QUICK STATS SUMMARY")
    print(f"  {'Run':<35} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  Acc")
    print(f"  {'─' * 62}")
    flat_runs = {m: a for (m, s), a in runs.items() if s == "flat"}
    for model in ("claude_sonnet4", "gpt4o", "llama3_3_70b"):
        if model not in flat_runs:
            continue
        c_all = {k: 0 for k in ("tp","fp","fn","tn")}
        for lbl in ALL_LABELS:
            c = confusion(flat_runs[model], gold)[lbl]
            for k in c_all:
                c_all[k] += c[k]
        total = sum(c_all.values())
        acc = (c_all["tp"] + c_all["tn"]) / total if total > 0 else 0.0
        name = MODEL_DISPLAY.get(model, model)
        print(f"  {name + ' (flat)':<35} {c_all['tp']:>5} {c_all['fp']:>5} "
              f"{c_all['fn']:>5} {c_all['tn']:>5}  {acc:.1%}")


if __name__ == "__main__":
    main()
