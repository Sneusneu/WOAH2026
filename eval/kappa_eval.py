#!/usr/bin/env python3
"""
kappa_eval.py — Evaluation for WOAH 2026 paper RQ1–RQ3.

Computes per (model, strategy) combination:
  • Cohen's κ (LLM vs human majority-vote gold)    → RQ1, RQ3
  • F1, precision, recall per intent category      → RQ1
  • Explicit vs implicit split                     → RQ1, RQ2
  • Inter-LLM pairwise agreement                   → RQ3
  • Flat vs hierarchical strategy comparison       → RQ2

Human IAA baselines (DIDECO Appendix D, 169 emails, 2+ annotators):
  Krippendorff's α (overall): 0.407
  Mean pairwise Cohen's κ:    0.355
  κ — Explicit intents:       0.577   ← primary RQ1 target for explicit
  κ — Implicit intents:       0.111   ← primary RQ1 target for implicit

Outputs:
  Console report | eval_results.json | tables/*.tex (--latex flag)

Usage:
  python kappa_eval.py --results-dir results --gold data/gold_standard.json
  python kappa_eval.py --results-dir results --gold data/gold_standard.json --latex
"""

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations


# ── Label definitions (DIDECO taxonomy, Popovic et al. 2026) ─────────────────

EXPLICIT_LABELS = [
    "information_manipulation", "resource_acquisition",
    "trust_establishment", "call_to_action",
]
IMPLICIT_LABELS = [
    "authority", "social_proof", "liking_similarity_deception",
    "distraction", "commitment_integrity_reciprocation",
]
ALL_LABELS = EXPLICIT_LABELS + IMPLICIT_LABELS

# Display names matching paper Table 1
LABEL_DISPLAY = {
    "information_manipulation":           "1.1 Information Manipulation",
    "resource_acquisition":               "1.2 Resource Acquisition",
    "trust_establishment":                "1.3 Trust Establishment",
    "call_to_action":                     "1.4 Call to Action",
    "authority":                          "2.1 Authority",
    "social_proof":                       "2.2 Social Proof",
    "liking_similarity_deception":        "2.3 Liking, Similarity & Deception",
    "distraction":                        "2.4 Distraction",
    "commitment_integrity_reciprocation": "2.5 Commitment, Integrity & Reciprocation",
}

LABEL_SHORT = {
    "information_manipulation":           "InfoManip",
    "resource_acquisition":               "ResourceAcq",
    "trust_establishment":                "TrustEstab",
    "call_to_action":                     "CallToAct",
    "authority":                          "Authority",
    "social_proof":                       "SocialProof",
    "liking_similarity_deception":        "Liking/Decept",
    "distraction":                        "Distraction",
    "commitment_integrity_reciprocation": "Commit/Recip",
}

# Human IAA baselines from DIDECO Appendix D
HUMAN_BASELINES = {
    "krippendorff_alpha":   0.407,
    "mean_pairwise_kappa":  0.355,
    "kappa_explicit":       0.577,
    "kappa_implicit":       0.111,
    "source":               "Popovic et al. (LREC 2026), Appendix D — 169 emails, 2+ annotators",
}

MODEL_DISPLAY = {
    "gpt4o":         "GPT-4o",
    "claude_sonnet4":"Claude Sonnet~4",
    "llama3_3_70b":  "Llama~3.3-70B",
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def cohen_kappa(y_pred: list, y_true: list) -> float:
    n = len(y_pred)
    if n == 0:
        return float("nan")
    tp = sum(1 for p, t in zip(y_pred, y_true) if p and t)
    tn = sum(1 for p, t in zip(y_pred, y_true) if not p and not t)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p and not t)
    fn = sum(1 for p, t in zip(y_pred, y_true) if not p and t)
    po = (tp + tn) / n
    pred_pos = (tp + fp) / n
    true_pos = (tp + fn) / n
    pe = pred_pos * true_pos + (1 - pred_pos) * (1 - true_pos)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def f1_pr(y_pred: list, y_true: list) -> dict:
    tp = sum(1 for p, t in zip(y_pred, y_true) if p and t)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p and not t)
    fn = sum(1 for p, t in zip(y_pred, y_true) if not p and t)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"f1": f1, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}


def macro_kappa(kappas: dict) -> float:
    vals = [v for v in kappas.values() if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


# ── Loading ───────────────────────────────────────────────────────────────────

def load_gold(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_results(results_dir: str) -> dict:
    """Returns {(model_short, strategy): {email_id: {explicit:{...}, implicit:{...}}}}"""
    results = {}
    for fpath in sorted(Path(results_dir).glob("*.json")):
        if fpath.name in ("manifest.json",):
            continue
        stem = fpath.stem
        with open(fpath, encoding="utf-8") as f:
            emails = json.load(f)

        indexed = {}
        for email in emails:
            if "annotation_error" in email or "annotations" not in email:
                continue
            eid = str(email.get("id", ""))
            ann = email["annotations"]
            indexed[eid] = {
                "explicit": ann.get("explicit", {}),
                "implicit": ann.get("implicit", {}),
            }

        # Infer model + strategy from filename
        if "_hierarchical" in stem:
            strategy = "hierarchical"
            model = stem.replace("_hierarchical", "")
        elif "_flat" in stem:
            strategy = "flat"
            model = stem.replace("_flat", "")
        else:
            strategy = "unknown"
            model = stem

        results[(model, strategy)] = indexed
        print(f"  Loaded ({model}, {strategy}): {len(indexed)} emails")

    return results


def get_vectors(result_index, gold, label, category):
    y_pred, y_true = [], []
    for eid, g in gold.items():
        if eid not in result_index:
            continue
        y_pred.append(bool(result_index[eid].get(category, {}).get(label, False)))
        y_true.append(bool(g.get(category, {}).get(label, False)))
    return y_pred, y_true


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_run(result_index: dict, gold: dict) -> dict:
    label_metrics = {}
    for category, labels in [("explicit", EXPLICIT_LABELS), ("implicit", IMPLICIT_LABELS)]:
        for label in labels:
            y_pred, y_true = get_vectors(result_index, gold, label, category)
            label_metrics[label] = {
                "category": category,
                "kappa": cohen_kappa(y_pred, y_true),
                **f1_pr(y_pred, y_true),
                "n": len(y_pred),
                "n_gold_pos": sum(y_true),
                "n_pred_pos": sum(y_pred),
            }

    exp_kappas = {l: label_metrics[l]["kappa"] for l in EXPLICIT_LABELS}
    imp_kappas = {l: label_metrics[l]["kappa"] for l in IMPLICIT_LABELS}

    return {
        "per_label": label_metrics,
        "macro_kappa_explicit": macro_kappa(exp_kappas),
        "macro_kappa_implicit": macro_kappa(imp_kappas),
        "macro_kappa_all":      macro_kappa({**exp_kappas, **imp_kappas}),
        "macro_f1_explicit": sum(label_metrics[l]["f1"] for l in EXPLICIT_LABELS) / len(EXPLICIT_LABELS),
        "macro_f1_implicit": sum(label_metrics[l]["f1"] for l in IMPLICIT_LABELS) / len(IMPLICIT_LABELS),
        "n_evaluated": sum(1 for eid in gold if eid in result_index),
    }


def compute_inter_llm(all_results: dict, gold: dict = None) -> dict:
    pairwise = {}
    for (k1, k2) in combinations(sorted(all_results.keys()), 2):
        r1, r2 = all_results[k1], all_results[k2]
        common_ids = set(r1) & set(r2)
        if gold:
            common_ids &= set(gold)
        common = sorted(common_ids)
        label_kappas = {}
        for cat, labels in [("explicit", EXPLICIT_LABELS), ("implicit", IMPLICIT_LABELS)]:
            for label in labels:
                y1 = [bool(r1[e].get(cat, {}).get(label, False)) for e in common]
                y2 = [bool(r2[e].get(cat, {}).get(label, False)) for e in common]
                label_kappas[label] = cohen_kappa(y1, y2)
        pairwise[f"{k1[0]}_{k1[1]} vs {k2[0]}_{k2[1]}"] = {
            "macro_kappa":          macro_kappa(label_kappas),
            "macro_kappa_explicit": macro_kappa({l: label_kappas[l] for l in EXPLICIT_LABELS}),
            "macro_kappa_implicit": macro_kappa({l: label_kappas[l] for l in IMPLICIT_LABELS}),
            "per_label": label_kappas,
            "n_common": len(common),
        }
    return pairwise


# ── Console report ────────────────────────────────────────────────────────────

def fk(k): return f"{k:+.3f}" if not math.isnan(k) else "   NaN"
def ff(f): return f"{f:.3f}"


def print_report(eval_results: dict, inter_llm: dict, gold: dict):
    print("\n" + "=" * 72)
    print("  WOAH 2026 — EVALUATION REPORT")
    print("  Dataset: DIDECO (Popovic et al., LREC 2026)")
    print("=" * 72)

    # RQ1 + RQ3 overview
    print(f"\n{'─'*72}")
    print("  RQ1/RQ3: MACRO KAPPA vs GOLD  |  Human baselines: κ_exp=0.577  κ_imp=0.111")
    print(f"{'─'*72}")
    print(f"  {'Run':<38} {'κ(exp)':<9} {'κ(imp)':<9} {'κ(all)':<9} {'F1(exp)':<9} {'F1(imp)':<9} {'n'}")
    print(f"  {'─'*72}")
    for (model, strat), ev in sorted(eval_results.items()):
        name = f"{model}_{strat}"
        print(f"  {name:<38} {fk(ev['macro_kappa_explicit']):<9} "
              f"{fk(ev['macro_kappa_implicit']):<9} {fk(ev['macro_kappa_all']):<9} "
              f"{ff(ev['macro_f1_explicit']):<9} {ff(ev['macro_f1_implicit']):<9} "
              f"{ev['n_evaluated']}")
    print(f"\n  {'Human IAA (Appendix D)':<38} {fk(0.577):<9} {fk(0.111):<9} "
          f"{'α=0.407':<9} {'—':<9} {'—':<9} 169")

    # RQ2: Strategy comparison
    print(f"\n{'─'*72}")
    print("  RQ2: HIERARCHICAL vs FLAT — Δκ per model")
    print(f"{'─'*72}")
    models = set(m for m, _ in eval_results)
    for model in sorted(models):
        flat_k = (model, "flat")
        hier_k = (model, "hierarchical")
        if flat_k in eval_results and hier_k in eval_results:
            fe, fi = eval_results[flat_k]["macro_kappa_explicit"], eval_results[flat_k]["macro_kappa_implicit"]
            he, hi = eval_results[hier_k]["macro_kappa_explicit"], eval_results[hier_k]["macro_kappa_implicit"]
            print(f"\n  {model}:")
            print(f"    Δκ explicit  (hier−flat): {he-fe:+.3f}  "
                  f"[flat={fk(fe)}, hier={fk(he)}]")
            print(f"    Δκ implicit  (hier−flat): {hi-fi:+.3f}  "
                  f"[flat={fk(fi)}, hier={fk(hi)}]")
            winner = "Hierarchical" if (he+hi) > (fe+fi) else "Flat" if (fe+fi) > (he+hi) else "Tie"
            print(f"    → {winner} better overall")

    # Per-label kappa table
    print(f"\n{'─'*72}")
    print("  PER-LABEL COHEN'S κ (all runs vs human gold)")
    print(f"{'─'*72}")
    run_keys = sorted(eval_results.keys())
    run_names = [f"{m[:8]}_{s[:4]}" for m, s in run_keys]
    header = f"  {'Label (DIDECO §3)':<30}" + "".join(f" {rn:<16}" for rn in run_names)
    print(header)
    print(f"  {'─'*72}")
    for layer, labels in [("EXPLICIT (Speech Act Theory)", EXPLICIT_LABELS),
                           ("IMPLICIT (Ferreira & Teles 2019)", IMPLICIT_LABELS)]:
        print(f"\n  [{layer}]")
        for label in labels:
            row = f"  {LABEL_DISPLAY.get(label, label):<30}"
            for key in run_keys:
                k = eval_results[key]["per_label"][label]["kappa"]
                row += f" {fk(k):<16}"
            print(row)

    # RQ3: Inter-LLM
    if inter_llm:
        print(f"\n{'─'*72}")
        print("  RQ3: INTER-LLM AGREEMENT (pairwise κ)")
        print(f"{'─'*72}")
        print(f"  {'Pair':<48} {'κ(all)':<9} {'κ(exp)':<9} {'κ(imp)':<9} {'n'}")
        print(f"  {'─'*72}")
        for pair, d in sorted(inter_llm.items()):
            print(f"  {pair:<48} {fk(d['macro_kappa']):<9} "
                  f"{fk(d['macro_kappa_explicit']):<9} {fk(d['macro_kappa_implicit']):<9} "
                  f"{d['n_common']}")

    print("\n" + "=" * 72)
    print("  END OF REPORT")
    print("=" * 72 + "\n")


# ── LaTeX tables ─────────────────────────────────────────────────────────────

def generate_latex(eval_results: dict, inter_llm: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Main results
    run_order = sorted(eval_results.keys())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLM annotation agreement with human gold standard (DIDECO, "
        r"Popovic et al.\ 2026). Human IAA baselines from Appendix~D: "
        r"$\kappa_{\text{exp}}=0.577$, $\kappa_{\text{imp}}=0.111$, "
        r"$\alpha_{\text{Kripp}}=0.407$.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Strategy & $\kappa_{\text{exp}}$ & $\kappa_{\text{imp}}$ "
        r"& $\kappa_{\text{all}}$ & $F1_{\text{exp}}$ & $F1_{\text{imp}}$ \\",
        r"\midrule",
    ]
    for (model, strat), ev in run_order_items(run_order, eval_results):
        m = MODEL_DISPLAY.get(model, model)
        s = "Hierarchical" if strat == "hierarchical" else "Flat"
        lines.append(
            f"{m} & {s} & {fmt_cell(ev['macro_kappa_explicit'])} "
            f"& {fmt_cell(ev['macro_kappa_implicit'])} "
            f"& {fmt_cell(ev['macro_kappa_all'])} "
            f"& {fmt_cell(ev['macro_f1_explicit'])} "
            f"& {fmt_cell(ev['macro_f1_implicit'])} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Human IAA (Appendix D)}} "
        r"& 0.577 & 0.111 & --- & --- & --- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    p = output_dir / "table_main_results.tex"
    p.write_text("\n".join(lines))
    print(f"  ✓ {p}")

    # Table 2: Per-label kappa
    col_spec = "l" + "c" * len(run_order)
    run_headers = [f"{MODEL_DISPLAY.get(m, m)[:8]} {s[:4]}" for m, s in run_order]
    lines2 = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-category Cohen's $\kappa$ (LLM vs.\ human gold). "
        r"Taxonomy: Popovic et al.\ (2026). "
        r"Hier = hierarchical prompting, Flat = flat prompting.}",
        r"\label{tab:per_label}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Category & " + " & ".join(run_headers) + r" \\",
        r"\midrule",
    ]
    for layer, layer_key, labels in [
        ("Explicit (\\S3.2)", "explicit", EXPLICIT_LABELS),
        ("Implicit (\\S3.3)", "implicit", IMPLICIT_LABELS),
    ]:
        lines2.append(
            r"\multicolumn{" + str(len(run_order)+1) + r"}{l}{\textit{" + layer + r"}} \\"
        )
        for label in labels:
            vals = [fmt_cell(eval_results[k]["per_label"][label]["kappa"]) for k in run_order]
            lines2.append(f"\\quad {LABEL_SHORT.get(label, label)} & " + " & ".join(vals) + r" \\")
    lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    p2 = output_dir / "table_per_label_kappa.tex"
    p2.write_text("\n".join(lines2))
    print(f"  ✓ {p2}")


def run_order_items(run_order, eval_results):
    return [(k, eval_results[k]) for k in run_order]


def fmt_cell(v):
    return f"{v:.3f}" if not math.isnan(v) else r"\textemdash"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate WOAH annotation experiments")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gold", default="data/gold_standard.json")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--no-gold", action="store_true",
                        help="Skip LLM-vs-gold metrics (for external datasets like Nazario)")
    args = parser.parse_args()

    print("=" * 72)
    print("  KAPPA EVALUATION — WOAH 2026")
    print("=" * 72)

    gold = None if args.no_gold else load_gold(args.gold)
    if gold:
        print(f"\n  Gold: {len(gold)} emails | Baselines: κ_exp={HUMAN_BASELINES['kappa_explicit']} "
              f"κ_imp={HUMAN_BASELINES['kappa_implicit']} α={HUMAN_BASELINES['krippendorff_alpha']}")
    else:
        print("\n  ⚠  No-gold mode: inter-LLM agreement only (no LLM-vs-gold κ)")

    print("\n  Loading results...")
    all_results = load_results(args.results_dir)
    if not all_results:
        print("\n  ⚠  No results found. Run woah_experiment_runner.py first.")
        return

    print("\n  Computing metrics...")
    eval_results = {k: evaluate_run(v, gold) for k, v in all_results.items()} if gold else {}
    inter_llm = compute_inter_llm(all_results, gold)

    print_report(eval_results, inter_llm, gold)

    # Save JSON
    def clean(v):
        return None if isinstance(v, float) and math.isnan(v) else v

    serializable = {
        "dataset": "DIDECO (Popovic et al., LREC 2026)",
        "human_iaa_baselines": HUMAN_BASELINES,
        "runs": {
            f"{m}_{s}": {
                "model": m, "strategy": s,
                **{k: clean(v) for k, v in ev.items() if k != "per_label"},
                "per_label": {
                    label: {mk: clean(mv) for mk, mv in metrics.items()}
                    for label, metrics in ev["per_label"].items()
                }
            }
            for (m, s), ev in eval_results.items()
        },
        "inter_llm_agreement": {
            pair: {k: clean(v) for k, v in d.items() if k != "per_label"}
            for pair, d in inter_llm.items()
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"  ✓ Saved → {args.output}")

    if args.latex:
        generate_latex(eval_results, inter_llm, Path("tables"))


if __name__ == "__main__":
    main()
