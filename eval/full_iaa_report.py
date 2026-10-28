#!/usr/bin/env python3
"""
full_iaa_report.py — Complete inter-annotator agreement report.

Computes three comparison types:
  1. Human vs Human   — pairwise kappa across all human annotators
  2. LLM vs Human      — each LLM against majority-vote human gold
  3. LLM vs LLM         — pairwise kappa across all LLM annotators

Auto-detects file format (TUI session dict, or annotate_sublabels.py list).

Usage:
  python3 full_iaa_report.py \
    --human annotations/annotator1.json annotations/annotator2.json session_colleague3.json \
    --llm annotations_claude_sublabels.json annotations_gemma_sublabels.json annotations_qwen_sublabels.json
"""

import json
import argparse
import numpy as np
from itertools import combinations
from collections import Counter

# ── Intent name -> field name mapping ─────────────────────────────────────────

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


# ── Loader (auto-detect format) ───────────────────────────────────────────────

def load_annotations(path: str):
    """
    Returns ({email_id: {field: bool}}, display_name)
    Supports TUI session format (dict) and annotate_sublabels.py format (list).
    """
    with open(path) as f:
        d = json.load(f)

    anns = d.get("annotations", {})

    if isinstance(anns, dict):
        # TUI session format
        name = d.get("annotator", path)
        result = {}
        for eid, ann in anns.items():
            labels = {}
            for intent_name, val in ann.get("intents", {}).items():
                field = INTENT_TO_FIELD.get(intent_name)
                if field:
                    labels[field] = bool(val)
            result[eid] = labels
        return result, name

    # annotate_sublabels.py format
    name = d.get("annotator_id", path)
    result = {}
    for ann in anns:
        if not isinstance(ann, dict) or ann.get("status") == "error":
            continue
        eid = ann["annotation_id"]
        labels = {}
        labels.update(ann.get("explicit", {}))
        labels.update(ann.get("implicit", {}))
        result[eid] = {k: bool(v) for k, v in labels.items() if k in ALL_FIELDS}
    return result, name


# ── Metrics ───────────────────────────────────────────────────────────────────

def cohen_kappa(a: dict, b: dict, fields: list, eids: list) -> float:
    kappas = []
    for f in fields:
        av = np.array([int(a.get(e, {}).get(f, False)) for e in eids])
        bv = np.array([int(b.get(e, {}).get(f, False)) for e in eids])
        po = np.mean(av == bv)
        pe = np.mean(av)*np.mean(bv) + (1-np.mean(av))*(1-np.mean(bv))
        k = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0
        kappas.append(k)
    return float(np.mean(kappas))


def majority_vote(annotators: list, eids: list, fields: list) -> dict:
    """Build majority-vote gold from a list of {eid: {field: bool}} dicts."""
    gold = {}
    n = len(annotators)
    for eid in eids:
        gold[eid] = {}
        for f in fields:
            votes = sum(1 for a in annotators if a.get(eid, {}).get(f, False))
            gold[eid][f] = votes > n / 2
    return gold


def krippendorff_alpha_binary(annotators: list, eids: list, fields: list) -> float:
    """
    Simplified Krippendorff's alpha for binary nominal data across
    multiple annotators (treats each (email, field) as one item).
    """
    # Build coincidence matrix approach (simplified for binary case)
    all_values = []
    for eid in eids:
        for f in fields:
            votes = [int(a.get(eid, {}).get(f, False)) for a in annotators]
            all_values.append(votes)

    all_values = np.array(all_values)  # shape (n_items, n_annotators)
    n_items, n_annot = all_values.shape

    # Observed disagreement
    Do = 0.0
    pair_count = 0
    for row in all_values:
        for i in range(n_annot):
            for j in range(i+1, n_annot):
                Do += (row[i] != row[j])
                pair_count += 1
    Do = Do / pair_count if pair_count else 0.0

    # Expected disagreement (from overall proportion of 1s)
    p1 = np.mean(all_values)
    De = 2 * p1 * (1 - p1)

    if De < 1e-9:
        return 1.0  # no variance, perfect agreement by definition
    return 1 - (Do / De)


def confusion(gold: dict, pred: dict, fields: list, eids: list):
    tp = fp = fn = tn = 0
    for eid in eids:
        for f in fields:
            g = gold.get(eid, {}).get(f, False)
            p = pred.get(eid, {}).get(f, False)
            if g and p: tp += 1
            elif not g and p: fp += 1
            elif g and not p: fn += 1
            else: tn += 1
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    return prec, rec, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full IAA report: human-human, LLM-human, LLM-LLM")
    parser.add_argument("--human", nargs="+", required=True, help="Human annotation session files")
    parser.add_argument("--llm",   nargs="+", default=[],     help="LLM annotation files")
    args = parser.parse_args()

    # Load all
    humans = {}
    for path in args.human:
        data, name = load_annotations(path)
        humans[name] = data
        print(f"  Loaded human: {name} ({len(data)} emails)")

    llms = {}
    for path in args.llm:
        data, name = load_annotations(path)
        llms[name] = data
        print(f"  Loaded LLM: {name} ({len(data)} emails)")

    all_sets = [set(d.keys()) for d in list(humans.values()) + list(llms.values())]
    eids = sorted(set.intersection(*all_sets))
    N = len(eids)

    print(f"\n{'='*70}")
    print(f"  FULL IAA REPORT")
    print(f"{'='*70}")
    print(f"  Common emails: {N}")
    print(f"  Human annotators: {len(humans)} | LLM annotators: {len(llms)}")

    # ── 1. HUMAN vs HUMAN ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  1. HUMAN vs HUMAN")
    print(f"{'─'*70}")
    human_names = list(humans.keys())

    if len(human_names) >= 2:
        print(f"\n  {'Pair':<35} {'κ_all':>8} {'κ_exp':>8} {'κ_imp':>8}")
        print(f"  {'─'*60}")
        human_kappas = []
        for n1, n2 in combinations(human_names, 2):
            ka = cohen_kappa(humans[n1], humans[n2], ALL_FIELDS,   eids)
            ke = cohen_kappa(humans[n1], humans[n2], SUB_EXPLICIT, eids)
            ki = cohen_kappa(humans[n1], humans[n2], SUB_IMPLICIT, eids)
            human_kappas.append(ka)
            print(f"  {n1+' vs '+n2:<35} {ka:>8.3f} {ke:>8.3f} {ki:>8.3f}")

        if len(human_names) >= 2:
            avg_k = np.mean(human_kappas)
            print(f"\n  Average pairwise κ: {avg_k:.3f}")

        if len(human_names) >= 3:
            human_list = [humans[n] for n in human_names]
            alpha = krippendorff_alpha_binary(human_list, eids, ALL_FIELDS)
            alpha_exp = krippendorff_alpha_binary(human_list, eids, SUB_EXPLICIT)
            alpha_imp = krippendorff_alpha_binary(human_list, eids, SUB_IMPLICIT)
            print(f"\n  Krippendorff's α (all {len(human_names)} annotators):")
            print(f"    α_all={alpha:.3f}  α_exp={alpha_exp:.3f}  α_imp={alpha_imp:.3f}")
    else:
        print("  Only 1 human annotator provided — need at least 2 for human-human IAA")

    # ── Build majority-vote human gold ────────────────────────────────────────
    human_gold = None
    if len(human_names) >= 2:
        human_list = [humans[n] for n in human_names]
        human_gold = majority_vote(human_list, eids, ALL_FIELDS)
        print(f"\n  ✓ Majority-vote human gold built from {len(human_names)} annotators")

    # ── 2. LLM vs HUMAN GOLD ──────────────────────────────────────────────────
    if llms and human_gold:
        print(f"\n{'─'*70}")
        print(f"  2. LLM vs HUMAN GOLD (majority vote, n={len(human_names)} annotators)")
        print(f"{'─'*70}")
        print(f"\n  {'Model':<25} {'κ_all':>8} {'κ_exp':>8} {'κ_imp':>8} {'F1':>8}")
        print(f"  {'─'*60}")
        for name, llm in llms.items():
            ka = cohen_kappa(human_gold, llm, ALL_FIELDS,   eids)
            ke = cohen_kappa(human_gold, llm, SUB_EXPLICIT, eids)
            ki = cohen_kappa(human_gold, llm, SUB_IMPLICIT, eids)
            tp, fp, fn, tn = confusion(human_gold, llm, ALL_FIELDS, eids)
            _, _, f1 = prf(tp, fp, fn)
            print(f"  {name:<25} {ka:>8.3f} {ke:>8.3f} {ki:>8.3f} {f1:>8.3f}")

        # Per-label breakdown
        print(f"\n  Per-label κ vs human gold:")
        header = f"  {'Label':<38}"
        for n in llms: header += f" {n[:10]:>10}"
        print(header)
        print(f"  {'─'*55}")
        print("  EXPLICIT")
        for f in SUB_EXPLICIT:
            row = f"    {f:<36}"
            for name, llm in llms.items():
                k = cohen_kappa(human_gold, llm, [f], eids)
                row += f" {k:>10.3f}"
            print(row)
        print("  IMPLICIT")
        for f in SUB_IMPLICIT:
            row = f"    {f:<36}"
            for name, llm in llms.items():
                k = cohen_kappa(human_gold, llm, [f], eids)
                row += f" {k:>10.3f}"
            print(row)

    elif llms and len(human_names) == 1:
        print(f"\n{'─'*70}")
        print(f"  2. LLM vs HUMAN (single annotator — not gold standard)")
        print(f"{'─'*70}")
        single_human = humans[human_names[0]]
        print(f"\n  {'Model':<25} {'κ_all':>8} {'κ_exp':>8} {'κ_imp':>8}")
        for name, llm in llms.items():
            ka = cohen_kappa(single_human, llm, ALL_FIELDS,   eids)
            ke = cohen_kappa(single_human, llm, SUB_EXPLICIT, eids)
            ki = cohen_kappa(single_human, llm, SUB_IMPLICIT, eids)
            print(f"  {name:<25} {ka:>8.3f} {ke:>8.3f} {ki:>8.3f}")

    # ── 3. LLM vs LLM ─────────────────────────────────────────────────────────
    if len(llms) >= 2:
        print(f"\n{'─'*70}")
        print(f"  3. LLM vs LLM")
        print(f"{'─'*70}")
        llm_names = list(llms.keys())
        print(f"\n  {'Pair':<35} {'κ_all':>8} {'κ_exp':>8} {'κ_imp':>8}")
        print(f"  {'─'*60}")
        llm_kappas = []
        for n1, n2 in combinations(llm_names, 2):
            ka = cohen_kappa(llms[n1], llms[n2], ALL_FIELDS,   eids)
            ke = cohen_kappa(llms[n1], llms[n2], SUB_EXPLICIT, eids)
            ki = cohen_kappa(llms[n1], llms[n2], SUB_IMPLICIT, eids)
            llm_kappas.append(ka)
            print(f"  {n1+' vs '+n2:<35} {ka:>8.3f} {ke:>8.3f} {ki:>8.3f}")
        print(f"\n  Average pairwise κ: {np.mean(llm_kappas):.3f}")

    # ── Summary comparison ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    if len(human_names) >= 2:
        print(f"  Human-human avg κ:  {np.mean(human_kappas):.3f}")
    if len(llms) >= 2:
        print(f"  LLM-LLM avg κ:      {np.mean(llm_kappas):.3f}")
    if llms and human_gold:
        llm_gold_kappas = [cohen_kappa(human_gold, llm, ALL_FIELDS, eids) for llm in llms.values()]
        print(f"  LLM-vs-gold avg κ:  {np.mean(llm_gold_kappas):.3f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
