#!/usr/bin/env python3
"""
held_out_comparison.py — Fair comparison: held-out human vs majority-of-others.

This produces the benchmark that is directly comparable to Claude's
"vs. 3-annotator majority vote" number (kappa=0.469), instead of comparing
Claude against an aggregate while only comparing humans pairwise.

For each annotator, this holds them out, builds a 2-person consensus from
the other two, and computes that held-out annotator's kappa against it.
This is the same "one vs. aggregate" structure used for Claude, so the
resulting numbers are now apples-to-apples.

Usage:
  python3 held_out_comparison.py \
    --human annotations/annotator1.json \
    --human annotations/annotator2.json \
    --human annotations/annotator3.json
"""

import json
import argparse
import numpy as np
from itertools import combinations


# ── Intent name -> field name mapping (same as full_iaa_report.py) ──────────

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

SUB_EXPLICIT = ["information_gathering", "information_control", "direct_financial_gain",
                "indirect_resource_acquisition", "identity_establishment", "legitimacy_creation",
                "direct_action_requests", "indirect_action_facilitation"]
SUB_IMPLICIT = ["authority", "herd_mentality", "diffusion_of_responsibility", "moral_duty",
                "general_deception", "deceptive_relationships", "liking_and_similarity",
                "scarcity", "overloading", "strong_affect", "need_and_greed",
                "integrity", "consistency", "commitment", "reciprocation"]
ALL_FIELDS = SUB_EXPLICIT + SUB_IMPLICIT


# ── Loader (auto-detects TUI session format vs annotate_sublabels.py format) ─

def load_annotations(path: str):
    """Returns ({email_id: {field: bool}}, display_name)."""
    with open(path) as f:
        d = json.load(f)

    anns = d.get("annotations", {})

    if isinstance(anns, dict):
        # TUI session format — dict keyed by email id
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

    # annotate_sublabels.py format — list of annotation dicts
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
        pe = np.mean(av) * np.mean(bv) + (1 - np.mean(av)) * (1 - np.mean(bv))
        k = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0
        kappas.append(k)
    return float(np.mean(kappas))


def macro_f1(pred: dict, gold: dict, fields: list, eids: list) -> float:
    """
    Macro-averaged F1 treating `gold` as ground truth and `pred` as
    predictions, computed per-email per-category, same unit as kappa.
    """
    tp = fp = fn = 0
    for eid in eids:
        for f in fields:
            p = pred.get(eid, {}).get(f, False)
            g = gold.get(eid, {}).get(f, False)
            if p and g: tp += 1
            elif p and not g: fp += 1
            elif not p and g: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def majority_vote_of_two(a: dict, b: dict, eids: list, fields: list, rule: str = "and") -> dict:
    """
    Build a 2-person consensus label.
      rule='and' -> label True only if BOTH annotators said True (unanimous, conservative)
      rule='or'  -> label True if EITHER annotator said True (liberal)
    """
    gold = {}
    for eid in eids:
        gold[eid] = {}
        for f in fields:
            va = a.get(eid, {}).get(f, False)
            vb = b.get(eid, {}).get(f, False)
            if rule == "and":
                gold[eid][f] = bool(va and vb)
            else:
                gold[eid][f] = bool(va or vb)
    return gold


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Held-out human vs majority-of-others — fair benchmark for model comparison")
    parser.add_argument("--human", action="append", required=True,
                        help="Path to a human annotation file. Pass 3 times, once per annotator.")
    parser.add_argument("--claude-kappa", type=float, default=0.469,
                        help="Claude's kappa vs 3-annotator majority vote, for reference (default: 0.469)")
    args = parser.parse_args()

    if len(args.human) != 3:
        print(f"ERROR: need exactly 3 human files, got {len(args.human)}")
        return

    # Load all three annotators
    humans = {}
    for path in args.human:
        data, name = load_annotations(path)
        humans[name] = data
        print(f"  Loaded: {name} ({len(data)} emails)")

    names = list(humans.keys())
    eids = sorted(set.intersection(*[set(d.keys()) for d in humans.values()]))
    print(f"\n  Common emails: {len(eids)}")

    print(f"\n{'='*65}")
    print(f"  HELD-OUT HUMAN vs MAJORITY-OF-OTHER-TWO")
    print(f"  (fair benchmark — same 'one vs aggregate' structure as Claude)")
    print(f"{'='*65}")

    for rule in ["and", "or"]:
        rule_label = "unanimous (AND)" if rule == "and" else "either (OR)"
        print(f"\n  Tie-break rule: {rule_label}")
        print(f"  {'-'*69}")
        print(f"  {'Held-out':<12} {'vs majority of':<22} {'k_all':>8} {'k_exp':>8} {'k_imp':>8} {'F1':>8}")
        print(f"  {'-'*69}")

        held_out_kappas = []
        held_out_f1s = []
        for held_out in names:
            others = [n for n in names if n != held_out]
            majority = majority_vote_of_two(humans[others[0]], humans[others[1]],
                                            eids, ALL_FIELDS, rule=rule)
            k_all = cohen_kappa(humans[held_out], majority, ALL_FIELDS, eids)
            k_exp = cohen_kappa(humans[held_out], majority, SUB_EXPLICIT, eids)
            k_imp = cohen_kappa(humans[held_out], majority, SUB_IMPLICIT, eids)
            f1 = macro_f1(humans[held_out], majority, ALL_FIELDS, eids)
            held_out_kappas.append(k_all)
            held_out_f1s.append(f1)
            others_str = f"({others[0]}, {others[1]})"
            print(f"  {held_out:<12} {others_str:<22} {k_all:>8.3f} {k_exp:>8.3f} {k_imp:>8.3f} {f1:>8.3f}")

        avg = np.mean(held_out_kappas)
        avg_f1 = np.mean(held_out_f1s)
        print(f"  {'-'*69}")
        print(f"  {'Average':<12} {'':<22} {avg:>8.3f} {'':>8} {'':>8} {avg_f1:>8.3f}")

    print(f"\n{'='*65}")
    print(f"  COMPARISON")
    print(f"{'='*65}")
    print(f"  Claude vs 3-annotator majority vote:        k = {args.claude_kappa:.3f}")
    print(f"  (see above for the fair held-out human benchmark under each rule)")
    print(f"\n  If Claude's k exceeds the held-out human average under the")
    print(f"  'unanimous' rule, that is the more defensible version of the claim.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
