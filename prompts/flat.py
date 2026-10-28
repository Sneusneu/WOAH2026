#!/usr/bin/env python3
"""
annotate_sublabels.py — Annotate emails at sub-label level (23 fields)
using the DIDECO taxonomy from the annotation guidelines PDF.

Matches exactly the output format of the woah-annotate TUI tool.

Usage:
  export OPENROUTER_API_KEY
  python3 annotate_sublabels.py \
    --input annotation_sample.json \
    --output annotations_claude.json \
    --model claude \
    --annotator claude_sonnet4
"""

import json
import time
import argparse
import os
from datetime import datetime
from pathlib import Path

import dspy


# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "claude": {
        "openrouter_id": "openrouter/anthropic/claude-sonnet-4-5",
        "display_name":  "Claude Sonnet 4",
        "short":         "claude_sonnet4",
    },
    "gpt4o": {
        "openrouter_id": "openrouter/openai/gpt-4o",
        "display_name":  "GPT-4o",
        "short":         "gpt4o",
    },
    "llama": {
        "openrouter_id": "openrouter/meta-llama/llama-3.3-70b-instruct",
        "display_name":  "Llama 3.3-70B",
        "short":         "llama3_3_70b",
    },
    "qwen": {
        "openrouter_id": "openrouter/qwen/qwen-2.5-7b-instruct",
        "display_name":  "Qwen 2.5-7B",
        "short":         "qwen2_5_7b",
    },
    "gemma": {
        "openrouter_id": "openrouter/google/gemma-3-27b-it",
        "display_name":  "Gemma 3-27B",
        "short":         "gemma3_27b",
    },
}


# ── DSPy signature — 23 sub-labels ───────────────────────────────────────────

class SubLabelSignature(dspy.Signature):
    """
    Annotate this email using the DIDECO persuasion intent taxonomy.
    Answer True or False for each sub-label based on textual evidence only.

    EXPLICIT INTENTS (Speech Act Theory — directly expressed communicative goals):
    Mark True only when clear textual evidence supports the sub-label.
    Default to False if uncertain.

    IMPLICIT INTENTS (Cialdini 1993 via Ferreira & Teles 2019 — psychological persuasion):
    Apply STRICT criteria. Only annotate when specific mechanism evidence is present.
    Surface-level professional language does NOT qualify.
    Human IAA on implicit categories is κ=0.111 — be conservative.
    """

    email_text: str = dspy.InputField(desc="Full email text to annotate")

    # ── EXPLICIT — 1.1 Information Manipulation ───────────────────────────────
    information_gathering: bool = dspy.OutputField(
        desc="1.1.1 — Direct or indirect requests for data, credentials, or identity confirmation. "
             "YES: 'Can you confirm your banking details?', 'Click here to verify your identity'. "
             "NO: Mentioning information in passing, providing info without requesting it."
    )
    information_control: bool = dspy.OutputField(
        desc="1.1.2 — Instructions to manage system access, permissions, or credential configurations. "
             "YES: 'Please modify your account preferences', 'Assign a username to this person'. "
             "NO: Generic IT updates without specific instructions."
    )

    # ── EXPLICIT — 1.2 Resource Acquisition ──────────────────────────────────
    direct_financial_gain: bool = dspy.OutputField(
        desc="1.2.1 — Explicit requests for money transfers, bank account details, or payment credentials. "
             "YES: 'Wire the funds to this account today', 'Please provide your bank details'. "
             "NO: Discussing invoices without requesting action."
    )
    indirect_resource_acquisition: bool = dspy.OutputField(
        desc="1.2.2 — Requests for non-monetary resources with clear value: gift cards, equipment, admin rights. "
             "YES: 'Purchase gift cards for an employee event', 'Grant admin access to this account'. "
             "NO: Requesting information (not resources), asking for a meeting."
    )

    # ── EXPLICIT — 1.3 Trust Establishment ───────────────────────────────────
    identity_establishment: bool = dspy.OutputField(
        desc="1.3.1 — Sender explicitly asserts their name, role, title, or position to establish credibility. "
             "YES: 'I am Prof. X, Manager at Central Bank', 'I am the personal attorney to Mr. Y'. "
             "NO: Generic email signatures, institutional logos without personal identity claim."
    )
    legitimacy_creation: bool = dspy.OutputField(
        desc="1.3.2 — Email invokes a known organisation, brand, official process, or compliance requirement. "
             "YES: 'We detected unusual activity on your PayPal account', 'As required by GDPR compliance'. "
             "NO: Generic professional tone, mentioning company name naturally."
    )

    # ── EXPLICIT — 1.4 Call to Action ────────────────────────────────────────
    direct_action_requests: bool = dspy.OutputField(
        desc="1.4.1 — Explicit instruction for an immediate specific action: click, fill, send, confirm. "
             "YES: 'Click here to verify your identity immediately', 'Wire the funds today'. "
             "NO: Vague requests like 'let me know your thoughts', passive information sharing."
    )
    indirect_action_facilitation: bool = dspy.OutputField(
        desc="1.4.2 — Sets up conditions for a future required action without issuing the direct command yet. "
             "YES: 'Keep these credentials for your next login', 'Reply urgently so I can inform you of the next step'. "
             "NO: Direct action requests (use 1.4.1), general information with no future action implied."
    )

    # ── IMPLICIT — 2.1 Authority (no sub-labels) ─────────────────────────────
    authority: bool = dspy.OutputField(
        desc="2.1 — Exploits hierarchical power, invokes institutional authority, or threatens consequences for non-compliance. "
             "YES: 'The director has approved this and requires your immediate action', 'Failure to update will result in account suspension'. "
             "NO: Generic professional titles in a signature, polite requests without power dynamic."
    )

    # ── IMPLICIT — 2.2 Social Proof ──────────────────────────────────────────
    herd_mentality: bool = dspy.OutputField(
        desc="2.2.1 — Claims that most or everyone has already complied, implying the recipient should follow. "
             "YES: 'Most of your colleagues have already completed this', 'Over 500 employees have already updated'. "
             "NO: Collective pronouns alone without a behavioral norm claim."
    )
    diffusion_of_responsibility: bool = dspy.OutputField(
        desc="2.2.2 — Uses collective framing to dilute individual agency — implies this applies to everyone. "
             "YES: 'We kindly request ALL employees, including yourself, to review this'. "
             "NO: 'We ask for your collaboration' — individual request with 'we' as subject only."
    )
    moral_duty: bool = dspy.OutputField(
        desc="2.2.3 — Appeals to prosocial values — frames compliance as protecting or benefiting others. "
             "YES: 'Your action is critical to protect your colleagues from this threat'. "
             "NO: 'This is important for the company' — importance claim without altruistic framing."
    )

    # ── IMPLICIT — 2.3 Liking, Similarity & Deception ────────────────────────
    general_deception: bool = dspy.OutputField(
        desc="2.3.1 — Presents a scenario that is objectively false or fabricated. "
             "YES: 'USD$15M in an abandoned account', fabricated security vulnerability. "
             "NO: Plausible but unverified claims — deception requires objective falsity."
    )
    deceptive_relationships: bool = dspy.OutputField(
        desc="2.3.2 — Falsifies a prior relationship, fakes a referral, or claims false familiarity. "
             "YES: 'Your colleague recommended I contact you', 'As per our previous conversation last Tuesday'. "
             "NO: Generic corporate relationship — 'We have worked with your team'."
    )
    liking_and_similarity: bool = dspy.OutputField(
        desc="2.3.3 — Exploits genuine or claimed shared characteristics, expertise, or identity to build rapport. "
             "YES: 'As a cybersecurity expert like yourself, I understand the urgency'. "
             "NO: Generic professional courtesy — 'We appreciate your continued collaboration'."
    )

    # ── IMPLICIT — 2.4 Distraction ────────────────────────────────────────────
    scarcity: bool = dspy.OutputField(
        desc="2.4.1 — Creates artificial urgency through time pressure (deadlines, 'immediately', 'last chance') "
             "OR resource scarcity (limited availability, exclusive access). "
             "YES: 'Confirm within 24 hours', 'Only available to a select group'. "
             "NO: 'Please respond at your earliest convenience' — polite, no pressure."
    )
    overloading: bool = dspy.OutputField(
        desc="2.4.2 — Deliberately overwhelms with multiple simultaneous demands or dense complex information. "
             "YES: 'Review the 25-page protocol and complete tasks A, B, C before end of day'. "
             "NO: Detailed but functional information — length alone is not overloading."
    )
    strong_affect: bool = dspy.OutputField(
        desc="2.4.3 — Deliberately triggers fear, alarm, or excitement to bypass rational analysis. "
             "YES: 'Your account has been locked' (fear), 'You have been selected for an exclusive reward' (excitement). "
             "NO: 'This is an important update' — importance claim without emotional trigger."
    )
    need_and_greed: bool = dspy.OutputField(
        desc="2.4.4 — Appeals to exceptional financial reward, exclusive status, or personal enrichment. "
             "YES: 'You will receive 30% of USD$15M upon completion'. "
             "NO: 'This will benefit your professional development' — career benefit is not greed."
    )

    # ── IMPLICIT — 2.5 Commitment, Integrity & Reciprocation ─────────────────
    integrity: bool = dspy.OutputField(
        desc="2.5.1 — Sender explicitly signals their own honesty, transparency, or good faith. "
             "YES: 'I am revealing this in complete confidence and good faith', 'please accept my apologies in advance'. "
             "NO: 'We take your privacy seriously' — institutional policy statement."
    )
    consistency: bool = dspy.OutputField(
        desc="2.5.2 — Invokes recipient's past behaviour, stated values, or established role to create obligation. "
             "YES: 'As someone who has always prioritized security, you'll understand the urgency'. "
             "NO: Role reference for relevance only, without consistency pressure."
    )
    commitment: bool = dspy.OutputField(
        desc="2.5.3 — Creates obligation through coercion, prior agreements, or role-based expectations with consequences. "
             "YES: 'If you do not confirm, your account will remain limited', 'Failure to respond will trigger escalation'. "
             "NO: 'We ask for your prompt response' — polite request without obligation framing."
    )
    reciprocation: bool = dspy.OutputField(
        desc="2.5.4 — Implies a prior favour was performed, creating a felt obligation to return it. "
             "YES: 'I am sharing confidential information and trust you will reciprocate'. "
             "NO: 'Thank you for your collaboration' — gratitude without creating obligation."
    )


# ── OR-derivation of top-level categories ────────────────────────────────────

def derive_top_level(sub: dict) -> dict:
    return {
        "information_manipulation":           sub["information_gathering"] or sub["information_control"],
        "resource_acquisition":               sub["direct_financial_gain"] or sub["indirect_resource_acquisition"],
        "trust_establishment":                sub["identity_establishment"] or sub["legitimacy_creation"],
        "call_to_action":                     sub["direct_action_requests"] or sub["indirect_action_facilitation"],
        "authority":                          sub["authority"],
        "social_proof":                       sub["herd_mentality"] or sub["diffusion_of_responsibility"] or sub["moral_duty"],
        "liking_similarity_deception":        sub["general_deception"] or sub["deceptive_relationships"] or sub["liking_and_similarity"],
        "distraction":                        sub["scarcity"] or sub["overloading"] or sub["strong_affect"] or sub["need_and_greed"],
        "commitment_integrity_reciprocation": sub["integrity"] or sub["consistency"] or sub["commitment"] or sub["reciprocation"],
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_annotation(emails: list, lm, annotator_name: str) -> dict:
    dspy.configure(lm=lm)
    predictor = dspy.Predict(SubLabelSignature)

    annotations = []
    n = len(emails)
    start_total = time.time()
    success = 0

    for i, email in enumerate(emails, 1):
        eid = email.get("annotation_id") or email.get("id", f"email_{i}")
        text = email.get("text") or f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}".strip()

        print(f"  [{i}/{n}] {eid}", end="", flush=True)
        t0 = time.time()

        try:
            r = predictor(email_text=text)

            sub = {
                # explicit
                "information_gathering":          bool(r.information_gathering),
                "information_control":            bool(r.information_control),
                "direct_financial_gain":          bool(r.direct_financial_gain),
                "indirect_resource_acquisition":  bool(r.indirect_resource_acquisition),
                "identity_establishment":         bool(r.identity_establishment),
                "legitimacy_creation":            bool(r.legitimacy_creation),
                "direct_action_requests":         bool(r.direct_action_requests),
                "indirect_action_facilitation":   bool(r.indirect_action_facilitation),
                # implicit
                "authority":                      bool(r.authority),
                "herd_mentality":                 bool(r.herd_mentality),
                "diffusion_of_responsibility":    bool(r.diffusion_of_responsibility),
                "moral_duty":                     bool(r.moral_duty),
                "general_deception":              bool(r.general_deception),
                "deceptive_relationships":        bool(r.deceptive_relationships),
                "liking_and_similarity":          bool(r.liking_and_similarity),
                "scarcity":                       bool(r.scarcity),
                "overloading":                    bool(r.overloading),
                "strong_affect":                  bool(r.strong_affect),
                "need_and_greed":                 bool(r.need_and_greed),
                "integrity":                      bool(r.integrity),
                "consistency":                    bool(r.consistency),
                "commitment":                     bool(r.commitment),
                "reciprocation":                  bool(r.reciprocation),
            }

            top = derive_top_level(sub)
            elapsed = int((time.time() - t0) * 1000)

            # Count yes labels
            exp_yes = sum(1 for k in ["information_gathering","information_control",
                                       "direct_financial_gain","indirect_resource_acquisition",
                                       "identity_establishment","legitimacy_creation",
                                       "direct_action_requests","indirect_action_facilitation"] if sub[k])
            imp_yes = sum(1 for k in ["authority","herd_mentality","diffusion_of_responsibility",
                                       "moral_duty","general_deception","deceptive_relationships",
                                       "liking_and_similarity","scarcity","overloading",
                                       "strong_affect","need_and_greed","integrity",
                                       "consistency","commitment","reciprocation"] if sub[k])

            print(f" → exp={exp_yes}/8  imp={imp_yes}/15  ({elapsed}ms)")

            annotations.append({
                "annotation_id":      eid,
                "explicit": {
                    # top-level derived
                    "information_manipulation":  top["information_manipulation"],
                    "resource_acquisition":      top["resource_acquisition"],
                    "trust_establishment":       top["trust_establishment"],
                    "call_to_action":            top["call_to_action"],
                    # sub-labels
                    "information_gathering":         sub["information_gathering"],
                    "information_control":           sub["information_control"],
                    "direct_financial_gain":         sub["direct_financial_gain"],
                    "indirect_resource_acquisition": sub["indirect_resource_acquisition"],
                    "identity_establishment":        sub["identity_establishment"],
                    "legitimacy_creation":           sub["legitimacy_creation"],
                    "direct_action_requests":        sub["direct_action_requests"],
                    "indirect_action_facilitation":  sub["indirect_action_facilitation"],
                },
                "implicit": {
                    # top-level derived
                    "authority":                          top["authority"],
                    "social_proof":                       top["social_proof"],
                    "liking_similarity_deception":        top["liking_similarity_deception"],
                    "distraction":                        top["distraction"],
                    "commitment_integrity_reciprocation": top["commitment_integrity_reciprocation"],
                    # sub-labels
                    "herd_mentality":              sub["herd_mentality"],
                    "diffusion_of_responsibility": sub["diffusion_of_responsibility"],
                    "moral_duty":                  sub["moral_duty"],
                    "general_deception":           sub["general_deception"],
                    "deceptive_relationships":     sub["deceptive_relationships"],
                    "liking_and_similarity":       sub["liking_and_similarity"],
                    "scarcity":                    sub["scarcity"],
                    "overloading":                 sub["overloading"],
                    "strong_affect":               sub["strong_affect"],
                    "need_and_greed":              sub["need_and_greed"],
                    "integrity":                   sub["integrity"],
                    "consistency":                 sub["consistency"],
                    "commitment":                  sub["commitment"],
                    "reciprocation":               sub["reciprocation"],
                }
            })
            success += 1

        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            print(f" ERROR ({elapsed}ms): {e}")
            annotations.append({"annotation_id": eid, "explicit": {}, "implicit": {}})

    total_time = time.time() - start_total
    return {
        "annotator_id":       annotator_name,
        "export_date":        datetime.utcnow().isoformat() + "Z",
        "guidelines_version": "refined_v2",
        "tool_version":       "llm_pipeline_1.0",
        "model":              lm.model if hasattr(lm, 'model') else str(lm),
        "n_emails":           n,
        "n_complete":         success,
        "total_time_seconds": int(total_time),
        "annotations":        annotations,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sub-label LLM annotation — WOAH 2026")
    parser.add_argument("--input",     required=True,  help="annotation_sample.json")
    parser.add_argument("--output",    required=True,  help="Output JSON file")
    parser.add_argument("--model",     default="claude", choices=list(MODELS.keys()))
    parser.add_argument("--annotator", default=None,   help="Annotator name in output (default: model short name)")
    parser.add_argument("--limit",     type=int, default=None, help="Limit to first N emails (for testing)")
    args = parser.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]

    model_cfg = MODELS[args.model]
    annotator = args.annotator or model_cfg["short"]

    print("=" * 60)
    print(f"  SUB-LABEL ANNOTATION — WOAH 2026")
    print(f"  Model: {model_cfg['display_name']}")
    print(f"  Annotator: {annotator}")
    print("=" * 60)

    with open(args.input, encoding="utf-8") as f:
        emails = json.load(f)

    if args.limit:
        emails = emails[:args.limit]
        print(f"  ⚠  Limited to {args.limit} emails")

    print(f"  Emails: {len(emails)}")
    print()

    lm = dspy.LM(
        model=model_cfg["openrouter_id"],
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=1000,
    )

    results = run_annotation(emails, lm, annotator)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"  ✓ {results['n_complete']}/{results['n_emails']} successful")
    print(f"  ✓ Saved → {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
