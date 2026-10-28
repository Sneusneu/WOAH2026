#!/usr/bin/env python3
"""
plot_annotations.py — Generate analysis graphs for human vs LLM annotation comparison.

Produces 6 figures saved to figures_annotation/ directory.

Usage:
  python3 plot_annotations.py \
    --human woah-project/woah-annotate/annotations/annotator1.json \
    --llm1 annotations_claude_sublabels.json \
    --llm2 annotations_gemma_sublabels.json \
    --llm3 annotations_qwen_sublabels.json \
    --out figures_annotation
"""

import json
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import combinations
from collections import Counter

# ── Style ─────────────────────────────────────────────────────────────────────

BG    = "#0D0F14"
SURF  = "#161820"
SURF2 = "#1C1F2A"
BORD  = "#252836"
TEXT  = "#E2E6F0"
TEXT2 = "#8B90A8"

COLORS = {
    "Human":         "#E2E6F0",
    "claude_sonnet4":"#534AB7",
    "gemma3_27b":    "#1D9E75",
    "qwen2_5_7b":    "#EF9F27",
}
DISPLAY = {
    "Human":         "Human",
    "claude_sonnet4":"Claude Sonnet 4",
    "gemma3_27b":    "Gemma 3-27B",
    "qwen2_5_7b":    "Qwen 2.5-7B",
}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURF,
    "axes.edgecolor":    BORD,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT2,
    "ytick.color":       TEXT2,
    "text.color":        TEXT,
    "grid.color":        BORD,
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
})

# ── Taxonomy ──────────────────────────────────────────────────────────────────

INTENT_TO_FIELD = {
    "Explicit / Information Manipulation / Information Gathering":          "information_gathering",
    "Explicit / Information Manipulation / Information Control":            "information_control",
    "Explicit / Resource Acquisition / Direct Financial Gain":              "direct_financial_gain",
    "Explicit / Resource Acquisition / Indirect Resource Acquisition":      "indirect_resource_acquisition",
    "Explicit / Trust Establishment / Identity Establishment":              "identity_establishment",
    "Explicit / Trust Establishment / Legitimacy Creation":                 "legitimacy_creation",
    "Explicit / Call to Action / Direct Action Requests":                   "direct_action_requests",
    "Explicit / Call to Action / Indirect Action Facilitation":             "indirect_action_facilitation",
    "Implicit / Authority":                                                  "authority",
    "Implicit / Social Proof / Herd Mentality":                             "herd_mentality",
    "Implicit / Social Proof / Diffusion of Responsibility":                "diffusion_of_responsibility",
    "Implicit / Social Proof / Moral Duty":                                 "moral_duty",
    "Implicit / Liking, Similarity & Deception / General Deception":       "general_deception",
    "Implicit / Liking, Similarity & Deception / Deceptive Relationships":  "deceptive_relationships",
    "Implicit / Liking, Similarity & Deception / Liking & Similarity":     "liking_and_similarity",
    "Implicit / Distraction / Scarcity":                                    "scarcity",
    "Implicit / Distraction / Overloading":                                 "overloading",
    "Implicit / Distraction / Strong Affect":                               "strong_affect",
    "Implicit / Distraction / Need & Greed":                                "need_and_greed",
    "Implicit / Commitment, Integrity & Reciprocation / Integrity":         "integrity",
    "Implicit / Commitment, Integrity & Reciprocation / Consistency":       "consistency",
    "Implicit / Commitment, Integrity & Reciprocation / Commitment":        "commitment",
    "Implicit / Commitment, Integrity & Reciprocation / Reciprocation":     "reciprocation",
}

SUB_EXPLICIT = ["information_gathering","information_control","direct_financial_gain",
                "indirect_resource_acquisition","identity_establishment","legitimacy_creation",
                "direct_action_requests","indirect_action_facilitation"]
SUB_IMPLICIT = ["authority","herd_mentality","diffusion_of_responsibility","moral_duty",
                "general_deception","deceptive_relationships","liking_and_similarity",
                "scarcity","overloading","strong_affect","need_and_greed",
                "integrity","consistency","commitment","reciprocation"]
ALL_FIELDS = SUB_EXPLICIT + SUB_IMPLICIT

SHORT = {
    "information_gathering":         "Info Gathering",
    "information_control":           "Info Control",
    "direct_financial_gain":         "Direct Financial",
    "indirect_resource_acquisition": "Indirect Resource",
    "identity_establishment":        "Identity Estab.",
    "legitimacy_creation":           "Legitimacy",
    "direct_action_requests":        "Direct Action",
    "indirect_action_facilitation":  "Indirect Action",
    "authority":                     "Authority",
    "herd_mentality":                "Herd Mentality",
    "diffusion_of_responsibility":   "Diffusion Resp.",
    "moral_duty":                    "Moral Duty",
    "general_deception":             "Gen. Deception",
    "deceptive_relationships":       "Decep. Relations",
    "liking_and_similarity":         "Liking & Sim.",
    "scarcity":                      "Scarcity",
    "overloading":                   "Overloading",
    "strong_affect":                 "Strong Affect",
    "need_and_greed":                "Need & Greed",
    "integrity":                     "Integrity",
    "consistency":                   "Consistency",
    "commitment":                    "Commitment",
    "reciprocation":                 "Reciprocation",
}

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_human(path):
    with open(path) as f:
        d = json.load(f)
    result = {}
    for eid, ann in d.get("annotations", {}).items():
        labels = {}
        for intent_name, val in ann.get("intents", {}).items():
            field = INTENT_TO_FIELD.get(intent_name)
            if field:
                labels[field] = bool(val)
        result[eid] = labels
    return result, "Human"

def load_llm(path):
    with open(path) as f:
        d = json.load(f)
    name = d.get("annotator_id", path)
    result = {}
    for ann in d.get("annotations", []):
        if ann.get("status") == "error":
            continue
        eid = ann["annotation_id"]
        labels = {}
        labels.update(ann.get("explicit", {}))
        labels.update(ann.get("implicit", {}))
        result[eid] = {k: bool(v) for k, v in labels.items() if k in ALL_FIELDS}
    return result, name

# ── Metrics ───────────────────────────────────────────────────────────────────

def cohen_kappa(a, b, fields, eids):
    kappas = []
    for f in fields:
        av = np.array([int(a.get(e, {}).get(f, False)) for e in eids])
        bv = np.array([int(b.get(e, {}).get(f, False)) for e in eids])
        po = np.mean(av == bv)
        pe = np.mean(av)*np.mean(bv) + (1-np.mean(av))*(1-np.mean(bv))
        k = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0
        kappas.append(k)
    return float(np.mean(kappas))

def detection_rate(anns, field, eids):
    return sum(1 for e in eids if anns.get(e, {}).get(field, False)) / len(eids) * 100

def save(fig, out, name):
    path = Path(out) / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {path}")

# ── Figures ───────────────────────────────────────────────────────────────────

def fig_overall_kappa(all_data, eids, out):
    """Fig 1 — Overall κ: explicit / implicit / all, per model"""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    fig.suptitle("Human vs LLM — Overall Cohen's κ", fontsize=13, fontweight="bold")

    llm_names = [n for n in all_data if n != "Human"]
    x = np.arange(3)
    w = 0.25
    metrics = ["Explicit κ", "Implicit κ", "Overall κ"]

    for i, name in enumerate(llm_names):
        vals = [
            cohen_kappa(all_data["Human"], all_data[name], SUB_EXPLICIT, eids),
            cohen_kappa(all_data["Human"], all_data[name], SUB_IMPLICIT, eids),
            cohen_kappa(all_data["Human"], all_data[name], ALL_FIELDS,   eids),
        ]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, color=COLORS[name]+"cc",
                      edgecolor=COLORS[name], linewidth=1,
                      label=DISPLAY[name])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT2)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Cohen's κ")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.2, facecolor=SURF2, edgecolor=BORD)
    fig.tight_layout()
    save(fig, out, "fig1_overall_kappa.png")


def fig_per_label_kappa(all_data, eids, out):
    """Fig 2 — Per-label κ horizontal bar chart"""
    llm_names = [n for n in all_data if n != "Human"]
    n_labels = len(ALL_FIELDS)
    fig, ax = plt.subplots(figsize=(11, 12), facecolor=BG)
    fig.suptitle("Per-label Cohen's κ — Human vs LLM", fontsize=13, fontweight="bold")

    y = np.arange(n_labels)
    w = 0.25

    for i, name in enumerate(llm_names):
        kappas = [cohen_kappa(all_data["Human"], all_data[name], [f], eids) for f in ALL_FIELDS]
        offset = (i - 1) * w
        ax.barh(y + offset, kappas, w,
                color=COLORS[name]+"cc", edgecolor=COLORS[name],
                linewidth=0.8, label=DISPLAY[name])

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[f] for f in ALL_FIELDS], fontsize=9)
    ax.axvline(0, color=BORD, linewidth=1)
    ax.axvline(0.8, color="#534AB7", linewidth=1, linestyle="--", alpha=0.5, label="κ=0.8 threshold")
    ax.set_xlim(-0.15, 1.1)
    ax.set_xlabel("Cohen's κ")
    ax.axhline(7.5, color=BORD, linewidth=1.5, linestyle="--")
    ax.text(1.05, 3.75, "EXPLICIT", ha="left", va="center", fontsize=8,
            color="#534AB7", rotation=90)
    ax.text(1.05, 11.5, "IMPLICIT", ha="left", va="center", fontsize=8,
            color="#1D9E75", rotation=90)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.2, facecolor=SURF2, edgecolor=BORD, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, out, "fig2_per_label_kappa.png")


def fig_detection_rates(all_data, eids, out):
    """Fig 3 — Detection rates: human vs LLMs"""
    n_labels = len(ALL_FIELDS)
    fig, ax = plt.subplots(figsize=(11, 12), facecolor=BG)
    fig.suptitle("Detection Rate per Sub-label (% True)", fontsize=13, fontweight="bold")

    y = np.arange(n_labels)
    names_ordered = ["Human"] + [n for n in all_data if n != "Human"]
    w = 0.2

    for i, name in enumerate(names_ordered):
        rates = [detection_rate(all_data[name], f, eids) for f in ALL_FIELDS]
        offset = (i - 1.5) * w
        hatch = "//" if name == "Human" else None
        ax.barh(y + offset, rates, w,
                color=COLORS[name]+"bb", edgecolor=COLORS[name],
                linewidth=0.8, hatch=hatch,
                label=DISPLAY[name])

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[f] for f in ALL_FIELDS], fontsize=9)
    ax.set_xlabel("Detection rate (%)")
    ax.set_xlim(0, 110)
    ax.axhline(7.5, color=BORD, linewidth=1.5, linestyle="--")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.2, facecolor=SURF2, edgecolor=BORD)
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, out, "fig3_detection_rates.png")


def fig_confusion(all_data, eids, out):
    """Fig 4 — TP/FP/FN/TN stacked bars, human as gold"""
    llm_names = [n for n in all_data if n != "Human"]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    fig.suptitle("Confusion Summary — Human as Gold", fontsize=13, fontweight="bold")

    tp_c = {"TP": "#1D9E75", "FP": "#E24B4A", "FN": "#EF9F27", "TN": "#252836"}
    x = np.arange(len(llm_names))

    tps, fps, fns, tns = [], [], [], []
    for name in llm_names:
        tp = fp = fn = tn = 0
        for f in ALL_FIELDS:
            for eid in eids:
                hv = int(all_data["Human"].get(eid, {}).get(f, False))
                lv = int(all_data[name].get(eid, {}).get(f, False))
                if hv and lv:     tp += 1
                elif not hv and lv: fp += 1
                elif hv and not lv: fn += 1
                else:               tn += 1
        tps.append(tp); fps.append(fp); fns.append(fn); tns.append(tn)

    ax.bar(x, tps, color=tp_c["TP"], label="TP")
    ax.bar(x, fps, bottom=tps, color=tp_c["FP"], label="FP")
    ax.bar(x, fns, bottom=np.array(tps)+np.array(fps), color=tp_c["FN"], label="FN")

    # Annotate
    for i, (tp, fp, fn, tn) in enumerate(zip(tps, fps, fns, tns)):
        total = tp+fp+fn+tn
        f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
        ax.text(i, tp+fp+fn+20, f"F1={f1:.3f}", ha="center", fontsize=9, color=TEXT2)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[n] for n in llm_names])
    ax.set_ylabel("Label count")
    ax.legend(fontsize=9, framealpha=0.2, facecolor=SURF2, edgecolor=BORD)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "fig4_confusion.png")


def fig_disagreement_heatmap(all_data, eids, out):
    """Fig 5 — Disagreement heatmap: FP and FN counts per label per model"""
    llm_names = [n for n in all_data if n != "Human"]
    n_labels = len(ALL_FIELDS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), facecolor=BG)
    fig.suptitle("Disagreement Patterns — FP and FN per Label", fontsize=13, fontweight="bold")

    for ax, dtype, title, cmap in zip(
        axes,
        ["FP", "FN"],
        ["False Positives (LLM=True, Human=False)", "False Negatives (LLM=False, Human=True)"],
        ["Reds", "Oranges"]
    ):
        matrix = np.zeros((n_labels, len(llm_names)))
        for j, name in enumerate(llm_names):
            for i, f in enumerate(ALL_FIELDS):
                for eid in eids:
                    hv = int(all_data["Human"].get(eid, {}).get(f, False))
                    lv = int(all_data[name].get(eid, {}).get(f, False))
                    if dtype == "FP" and not hv and lv:
                        matrix[i, j] += 1
                    elif dtype == "FN" and hv and not lv:
                        matrix[i, j] += 1

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0)
        ax.set_xticks(range(len(llm_names)))
        ax.set_xticklabels([DISPLAY[n].split()[0] for n in llm_names], fontsize=9)
        ax.set_yticks(range(n_labels))
        ax.set_yticklabels([SHORT[f] for f in ALL_FIELDS], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.axhline(7.5, color=TEXT2, linewidth=1.5, linestyle="--")

        for i in range(n_labels):
            for j in range(len(llm_names)):
                v = int(matrix[i, j])
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if v > 10 else "black")

        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    save(fig, out, "fig5_disagreement_heatmap.png")


def fig_source_breakdown(all_data, eids, out, source_map):
    """Fig 6 — Avg labels per email by corpus source"""
    llm_names = [n for n in all_data if n != "Human"]
    sources = ["TExtPhish", "Nazario", "Enron"]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    fig.suptitle("Average Labels per Email by Source Corpus", fontsize=13, fontweight="bold")

    x = np.arange(len(sources))
    w = 0.2

    all_names = ["Human"] + llm_names
    for i, name in enumerate(all_names):
        avgs = []
        for src in sources:
            src_eids = [e for e in eids if source_map.get(e, "") == src]
            if not src_eids:
                avgs.append(0)
                continue
            total = sum(1 for eid in src_eids for f in ALL_FIELDS
                        if all_data[name].get(eid, {}).get(f, False))
            avgs.append(total / len(src_eids))
        offset = (i - 1.5) * w
        hatch = "//" if name == "Human" else None
        ax.bar(x + offset, avgs, w,
               color=COLORS[name]+"bb", edgecolor=COLORS[name],
               linewidth=0.8, hatch=hatch, label=DISPLAY[name])
        for j, v in enumerate(avgs):
            ax.text(x[j] + offset, v + 0.05, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7, color=TEXT2)

    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Avg labels / email")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.2, facecolor=SURF2, edgecolor=BORD)
    fig.tight_layout()
    save(fig, out, "fig6_source_breakdown.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human",  required=True)
    parser.add_argument("--llm1",   required=True)
    parser.add_argument("--llm2",   default=None)
    parser.add_argument("--llm3",   default=None)
    parser.add_argument("--out",    default="figures_annotation")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    human_data, _ = load_human(args.human)
    all_data = {"Human": human_data}

    for path in [args.llm1, args.llm2, args.llm3]:
        if path:
            data, name = load_llm(path)
            all_data[name] = data
            print(f"  Loaded {name}: {len(data)} emails")

    # Source names are stored in the ID-only manifest, not repeated in predictions.
    manifest = Path(__file__).resolve().parents[1] / "corpus" / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as f:
        source_map = {row["message_id"]: row["source"] for row in csv.DictReader(f)}

    eids = sorted(set.intersection(*[set(d.keys()) for d in all_data.values()]))
    print(f"  Common emails: {len(eids)}")

    print("\nGenerating figures...")
    fig_overall_kappa(all_data, eids, args.out)
    fig_per_label_kappa(all_data, eids, args.out)
    fig_detection_rates(all_data, eids, args.out)
    fig_confusion(all_data, eids, args.out)
    fig_disagreement_heatmap(all_data, eids, args.out)
    fig_source_breakdown(all_data, eids, args.out, source_map)

    print(f"\nDone — 6 figures saved to {args.out}/")


if __name__ == "__main__":
    main()
