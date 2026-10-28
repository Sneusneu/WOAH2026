#!/usr/bin/env python3
"""Recompute the mixed-100 headline agreement metrics from ID-only labels."""
import argparse
import json
from pathlib import Path

from full_iaa_report import (
    ALL_FIELDS, SUB_EXPLICIT, SUB_IMPLICIT, cohen_kappa,
    krippendorff_alpha_binary, load_annotations, majority_vote, prf,
)

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    "information_manipulation": ["information_gathering", "information_control"],
    "resource_acquisition": ["direct_financial_gain", "indirect_resource_acquisition"],
    "trust_establishment": ["identity_establishment", "legitimacy_creation"],
    "call_to_action": ["direct_action_requests", "indirect_action_facilitation"],
    "authority": ["authority"],
    "social_proof": ["herd_mentality", "diffusion_of_responsibility", "moral_duty"],
    "liking_similarity_deception": ["general_deception", "deceptive_relationships", "liking_and_similarity"],
    "distraction": ["scarcity", "overloading", "strong_affect", "need_and_greed"],
    "commitment_integrity_reciprocation": ["integrity", "consistency", "commitment", "reciprocation"],
}


def derive_top(labels):
    return {name: any(labels.get(field, False) for field in fields)
            for name, fields in TOP_LEVEL.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results" / "computed_metrics.json"))
    args = parser.parse_args()

    human_files = [ROOT / "annotations" / f"annotator{i}.json" for i in range(1, 4)]
    humans = [load_annotations(str(path))[0] for path in human_files]
    ids = sorted(set.intersection(*(set(h) for h in humans)))
    human_gold = majority_vote(humans, ids, ALL_FIELDS)
    human_top = [{eid: derive_top(labels) for eid, labels in h.items() if eid in ids}
                 for h in humans]
    top_gold = {eid: {label: sum(h[eid][label] for h in human_top) > 1
                      for label in TOP_LEVEL} for eid in ids}

    model_files = sorted((ROOT / "results" / "mixed100").glob("*.json"))
    metrics = {
        "n_emails": len(ids),
        "human_krippendorff_alpha": {
            "all": krippendorff_alpha_binary(humans, ids, ALL_FIELDS),
            "explicit": krippendorff_alpha_binary(humans, ids, SUB_EXPLICIT),
            "implicit": krippendorff_alpha_binary(humans, ids, SUB_IMPLICIT),
        },
        "models": {},
    }
    for path in model_files:
        pred, _ = load_annotations(str(path))
        pred_top = {eid: derive_top(pred.get(eid, {})) for eid in ids}
        top_kappas = {label: cohen_kappa(top_gold, pred_top, [label], ids)
                      for label in TOP_LEVEL}
        tp = fp = fn = 0
        for eid in ids:
            for label in ALL_FIELDS:
                p, g = pred.get(eid, {}).get(label, False), human_gold[eid].get(label, False)
                tp += bool(p and g)
                fp += bool(p and not g)
                fn += bool(not p and g)
        precision, recall, f1 = prf(tp, fp, fn)
        metrics["models"][path.stem] = {
            "top_level_macro_kappa": sum(top_kappas.values()) / len(top_kappas),
            "top_level_per_label_kappa": top_kappas,
            "sublabel_macro_kappa": cohen_kappa(human_gold, pred, ALL_FIELDS, ids),
            "sublabel_explicit_macro_kappa": cohen_kappa(human_gold, pred, SUB_EXPLICIT, ids),
            "sublabel_implicit_macro_kappa": cohen_kappa(human_gold, pred, SUB_IMPLICIT, ids),
            "sublabel_micro_f1": f1,
            "sublabel_micro_precision": precision,
            "sublabel_micro_recall": recall,
        }
    Path(args.out).write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"n_emails": metrics["n_emails"],
                      "human_alpha": metrics["human_krippendorff_alpha"],
                      "top_level_macro_kappa": {k: v["top_level_macro_kappa"]
                                                 for k, v in metrics["models"].items()},
                      "sublabel_macro_kappa": {k: v["sublabel_macro_kappa"]
                                               for k, v in metrics["models"].items()}}, indent=2))
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
