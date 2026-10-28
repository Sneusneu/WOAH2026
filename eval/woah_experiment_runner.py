#!/usr/bin/env python3
"""
woah_experiment_runner.py — WOAH 2026 annotation experiments.

Runs 3 LLMs × 2 prompting strategies on the DIDECO email set
(Popovic et al., LREC 2026 — 220 LLM-generated spear-phishing emails).

Models:
  GPT-4o        (OpenAI, via OpenRouter)
  Claude Sonnet 4 (Anthropic, via OpenRouter)
  Llama 3.3-70B   (Meta, via OpenRouter or local Ollama)

Strategies:
  flat         — all 9 high-level intents in a single DSPy call (RQ2 baseline)
  hierarchical — two-step: explicit then implicit, with fine-grained follow-up (RQ2 treatment)

Research questions addressed:
  RQ1: Do LLMs match/exceed human IAA (κ_explicit=0.577, κ_implicit=0.111)?
  RQ2: Does hierarchical prompting outperform flat prompting?
  RQ3: How does reliability vary across LLM families?

Usage:
  export OPENROUTER_API_KEY
  python woah_experiment_runner.py --input data/emails.json
  python woah_experiment_runner.py --input data/emails.json --models gpt4o --strategies flat
  python woah_experiment_runner.py --dry-run
"""

import json
import time
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
import dspy


# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "gpt4o": {
        "openrouter_id": "openrouter/openai/gpt-4o",
        "display_name":  "GPT-4o",
        "short":         "gpt4o",
    },
    "claude": {
        "openrouter_id": "openrouter/anthropic/claude-haiku-4-5",
        "display_name":  "Claude Haiku 4.5",
        "short":         "claude_haiku4",
    },
    "llama": {
        "openrouter_id": "openrouter/meta-llama/llama-3.3-70b-instruct",
        "ollama_id":     "ollama/llama3.3",
        "display_name":  "Llama 3.3-70B",
        "short":         "llama3_3_70b",
    },
}

STRATEGIES = ["flat", "hierarchical"]


# ── Flat strategy ─────────────────────────────────────────────────────────────

def run_flat_strategy(lm, emails: list) -> list:
    """
    Flat strategy: classify all 9 high-level intents in a single DSPy call.
    Uses DIDECO taxonomy category descriptions directly in field docstrings.
    RQ2 baseline — compared against hierarchical two-step strategy.
    """
    from datetime import datetime

    class FlatIntentSignature(dspy.Signature):
        """
        Classify all explicit and implicit intents present in this spear-phishing email
        using the DIDECO taxonomy (Popovic et al. 2026).

        EXPLICIT intents (Speech Act Theory — directly expressed communicative goals):
          Mark True only when clear textual evidence supports the category.

        IMPLICIT intents (Ferreira & Teles 2019 framework — psychological persuasion mechanisms):
          Apply STRICT criteria. Human IAA on implicit categories is κ=0.111, reflecting
          genuine difficulty. Only annotate when specific mechanism evidence is present.
          Surface-level professional language does NOT qualify.
        """
        email_text: str = dspy.InputField(desc="Full email text to classify")

        # 1. Explicit intents
        information_manipulation: bool = dspy.OutputField(
            desc="1.1 — Direct requests for data/credentials OR instructions to manage system access"
        )
        resource_acquisition: bool = dspy.OutputField(
            desc="1.2 — Requests for money/bank details (Direct Financial Gain) OR gift cards/admin rights/equipment (Indirect)"
        )
        trust_establishment: bool = dspy.OutputField(
            desc="1.3 — Sender asserts identity/role (Identity Establishment) OR invokes org/brand/official process (Legitimacy Creation)"
        )
        call_to_action: bool = dspy.OutputField(
            desc="1.4 — Explicit immediate action instruction (Direct) OR preparatory step for a future action (Indirect Facilitation)"
        )

        # 2. Implicit intents
        authority: bool = dspy.OutputField(
            desc="2.1 — Exploits hierarchical authority, uses credentials directively, OR implies consequences for non-compliance (Cialdini 1993)"
        )
        social_proof: bool = dspy.OutputField(
            desc="2.2 — Herd mentality ('everyone did X'), diffusion of responsibility ('all employees'), OR moral duty (compliance protects others)"
        )
        liking_similarity_deception: bool = dspy.OutputField(
            desc="2.3 — General deception (fabricated scenarios), deceptive relationships (fake prior bond), OR liking/similarity (shared traits exploited for rapport)"
        )
        distraction: bool = dspy.OutputField(
            desc="2.4 — Scarcity (time/resource urgency), overloading, strong affect (fear/alarm/excitement), OR need and greed (exceptional reward appeal)"
        )
        commitment_integrity_reciprocation: bool = dspy.OutputField(
            desc="2.5 — Integrity signal (sender claims honesty), consistency (invokes past behavior/role), commitment (obligation/coercion), OR reciprocation (prior favor creates debt)"
        )

    dspy.configure(lm=lm)
    predictor = dspy.Predict(FlatIntentSignature)
    results = []
    n = len(emails)

    for i, email in enumerate(emails, 1):
        eid = email.get("id", f"email_{i}")
        text = email.get("text") or f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}".strip()
        print(f"  [{i}/{n}] {eid}", end="", flush=True)
        try:
            r = predictor(email_text=text)
            annotations = {
                "explicit": {
                    "information_manipulation": bool(r.information_manipulation),
                    "resource_acquisition":     bool(r.resource_acquisition),
                    "trust_establishment":      bool(r.trust_establishment),
                    "call_to_action":           bool(r.call_to_action),
                },
                "implicit": {
                    "authority":                          bool(r.authority),
                    "social_proof":                       bool(r.social_proof),
                    "liking_similarity_deception":        bool(r.liking_similarity_deception),
                    "distraction":                        bool(r.distraction),
                    "commitment_integrity_reciprocation": bool(r.commitment_integrity_reciprocation),
                }
            }
            exp_pos = sum(1 for v in annotations["explicit"].values() if v)
            imp_pos = sum(1 for v in annotations["implicit"].values() if v)
            print(f" → explicit={exp_pos}/4  implicit={imp_pos}/5")
            result = {"id": eid, "annotations": annotations}
        except Exception as e:
            print(f" ERROR: {e}")
            result = {"id": eid, "annotation_error": True}
        results.append(result)

    return results


def run_hierarchical_strategy(lm, emails: list) -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompts"))
    from hierarchical import HierarchicalAnnotator
    annotator = HierarchicalAnnotator(lm=lm, fine_grained=True)
    return annotator.annotate_dataset(emails, verbose=True)


# ── Model init ────────────────────────────────────────────────────────────────

def init_model(model_key: str, use_ollama: bool = False) -> dspy.LM:
    cfg = MODELS[model_key]
    kwargs = {"temperature": 0.0, "max_tokens": 1000}

    if use_ollama and "ollama_id" in cfg:
        model_id = cfg["ollama_id"]
        kwargs["api_base"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    else:
        model_id = cfg["openrouter_id"]
        kwargs["api_base"] = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        kwargs["api_key"] = os.environ["OPENROUTER_API_KEY"]

    print(f"  Initializing {cfg['display_name']} ({model_id})")
    return dspy.LM(model_id, **kwargs)


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(model_key, strategy, emails, output_dir, use_ollama=False):
    cfg = MODELS[model_key]
    short_name = f"{cfg['short']}_{strategy}"
    output_path = output_dir / f"{short_name}.json"

    if output_path.exists():
        print(f"  ⏭  Skipping {short_name} — results exist ({output_path})")
        with open(output_path) as f:
            return {"model": model_key, "strategy": strategy, "skipped": True,
                    "results": json.load(f)}

    print(f"\n{'='*65}")
    print(f"  {cfg['display_name']} × {strategy.upper()}")
    print(f"  Output: {output_path}")
    print(f"{'='*65}")

    start = time.time()
    try:
        lm = init_model(model_key, use_ollama=use_ollama)
        results = (run_flat_strategy if strategy == "flat" else run_hierarchical_strategy)(lm, emails)
        elapsed = time.time() - start
        successful = sum(1 for r in results if "annotation_error" not in r)
        print(f"\n  ✓ {successful}/{len(emails)} successful in {elapsed:.1f}s")

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved → {output_path}")

        return {"model": model_key, "strategy": strategy, "results": results,
                "elapsed": elapsed, "successful": successful, "skipped": False}
    except Exception as e:
        print(f"\n  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()
        return {"model": model_key, "strategy": strategy, "error": str(e), "skipped": False}


def print_summary(run_results, emails):
    print(f"\n{'='*65}\n  EXPERIMENT SUMMARY\n{'='*65}")
    print(f"  {'Combination':<35} {'Status':<10} {'Success':<10} {'Time'}")
    print(f"  {'─'*62}")
    for r in run_results:
        name = f"{MODELS[r['model']]['short']}_{r['strategy']}"
        if r.get("skipped"):   status, success, elapsed = "SKIPPED", "—", "—"
        elif "error" in r:     status, success, elapsed = "FAILED", "—", "—"
        else:
            status = "OK"
            success = f"{r['successful']}/{len(emails)}"
            elapsed = f"{r.get('elapsed',0):.0f}s"
        print(f"  {name:<35} {status:<10} {success:<10} {elapsed}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="WOAH 2026 annotation experiments: 3 LLMs × 2 strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python woah_experiment_runner.py --input data/emails.json
  python woah_experiment_runner.py --input data/emails.json --models gpt4o --strategies flat
  python woah_experiment_runner.py --dry-run
  python woah_experiment_runner.py --models llama --use-ollama
        """
    )
    parser.add_argument("--input", default="data/emails.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()), default=list(MODELS.keys()))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  WOAH 2026 ANNOTATION EXPERIMENTS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  Dataset: DIDECO (Popovic et al. 2026) — spear-phishing emails")
    print("  IAA targets: κ_explicit=0.577 / κ_implicit=0.111")
    print("=" * 65)

    with open(args.input, encoding="utf-8") as f:
        emails = json.load(f)
    if args.limit:
        emails = emails[:args.limit]
        print(f"  ⚠  Limited to {args.limit} emails")

    print(f"\n  Emails: {len(emails)} | Models: {args.models} | Strategies: {args.strategies}")

    if args.dry_run:
        print("\n  DRY RUN — no API calls")
        for m in args.models:
            for s in args.strategies:
                print(f"    Would run: {MODELS[m]['short']}_{s}")
        return

    output_dir = Path(args.output_dir)
    run_results = []
    for model_key in args.models:
        for strategy in args.strategies:
            result = run_experiment(model_key, strategy, emails, output_dir,
                                    args.use_ollama)
            run_results.append(result)

    print_summary(run_results, emails)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "DIDECO (Popovic et al. 2026)",
        "n_emails": len(emails),
        "models": args.models,
        "strategies": args.strategies,
        "human_iaa_baselines": {
            "krippendorff_alpha": 0.407,
            "mean_pairwise_kappa": 0.355,
            "kappa_explicit": 0.577,
            "kappa_implicit": 0.111,
        },
        "runs": [{"model": r["model"], "strategy": r["strategy"],
                  "status": "skipped" if r.get("skipped") else ("ok" if "error" not in r else "failed"),
                  "successful": r.get("successful"), "elapsed": r.get("elapsed")}
                 for r in run_results],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest → {output_dir}/manifest.json")
    print(f"\n  Next: python kappa_eval.py --results-dir results --gold data/gold_standard.json --latex")


if __name__ == "__main__":
    main()
