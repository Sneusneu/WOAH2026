#!/usr/bin/env python3
"""
hierarchical_prompts.py — Hierarchical few-shot DSPy annotation strategy (RQ2).

Implements the TWO-STEP strategy that mirrors the DIDECO taxonomy structure
(Popovic et al., LREC 2026), contrasting with the FLAT single-call baseline.

Theoretical grounding:
  Explicit intents: Speech Act Theory (Austin 1962, Searle 1969)
  Implicit intents: Ferreira & Teles (2019) framework, synthesising
    Cialdini (1993) — Authority, Social Proof, Scarcity, Reciprocity
    Gragg (2003) — social engineering principles
    Stajano & Wilson (2011) — scam victim psychology

Illustrative examples are synthetic summaries, not source-email excerpts.

Step 1: Classify 4 high-level explicit intent categories
Step 2: Classify 5 high-level implicit intent categories (strict criteria)
Step 3: Classify fine-grained sub-labels for active categories
"""

import dspy


# ── Definitions grounded in DIDECO paper (Section 3) ─────────────────────────

EXPLICIT_DEFINITIONS = """
EXPLICIT INTENTS — directly expressed communicative goals (Speech Act Theory, Searle 1969).
Annotate all that apply; most phishing emails combine multiple explicit intents.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.1 INFORMATION MANIPULATION
Central role of information exchange and control in digital interactions (Dabbish et al. 2005).

  1.1.1 Information Gathering — direct requests to obtain data, credentials, or confirmation.
    ✓ "Can you confirm that the price is the same direct from the factory?"
    ✓ "Could you please confirm your banking details for payment processing?"
    ✓ "Click here to verify your identity" (identity verification = information gathering)
    ✗ Mentioning information in passing without requesting it

  1.1.2 Information Control — instructions to manage information flow or system access.
    ✓ "Please modify your account preferences at [link]"
    ✓ "Please assign a user name to Randy Gay"
    ✓ Requests to update credentials, change permissions, or alter security configurations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.2 RESOURCE ACQUISITION
Resource requests constitute a distinct communicative objective (Dabbish & Kraut 2006).

  1.2.1 Direct Financial Gain — explicit money transfers, bank account details, payment credentials.
    ✓ "Wire the funds to this account today"
    ✓ "You will receive 40% of the total amount — please transfer now"
    ✓ Any request involving bank details, wire transfers, cryptocurrency

  1.2.2 Indirect Resource Acquisition — non-monetary resources with clear value.
    ✓ "I need you to purchase some gift cards for an employee appreciation event"
    ✓ "Please order this equipment and ship to this address"
    ✓ Requests for admin rights, software licenses, asset transfers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.3 TRUST ESTABLISHMENT
Establishing credibility is a distinct communicative goal preceding any request (Gragg 2003).

  1.3.1 Identity Establishment — sender verifies or asserts their identity, role, or position.
    ✓ "I am Prof Charles Soludo the Bill and Exchange Manager, Central Bank of Nigeria"
    ✓ "I am the personal attorney to Late Mr. Mark Michelle"
    ✓ Claims of being a colleague, executive, IT admin, or authority figure

  1.3.2 Legitimacy Creation — establishes credibility through institutional/organizational association.
    ✓ "We have detected unusual activity on your PayPal account" (uses PayPal's brand)
    ✓ "We are reaching out regarding a critical security update at SafeSecurity"
    ✓ References to known organizations, compliance requirements, official processes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.4 CALL TO ACTION
Distinct from long-term compliance strategies — about eliciting specific, identifiable behaviors
(Bennett & Carbonell 2005).

  1.4.1 Direct Action Requests — explicit instructions for immediate, specific actions.
    ✓ "Click here to verify your identity immediately to prevent permanent suspension"
    ✓ "Please confirm your information by clicking the link below within the next 24 hours"
    ✓ "Wire the funds to this account today" / "Fill in this form" / "Call this number now"

  1.4.2 Indirect Action Facilitation — preparatory steps enabling a future required action.
    ✓ "Keep these credentials for your next login"
    ✓ "Please reply urgently so that I will inform you of the next step to take"
    ✓ Sets up conditions for subsequent compliance without issuing the direct command yet

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYNTHETIC WORKED EXAMPLES (summaries, not email text):

- A fabricated access notice asks the reader to confirm credentials immediately.
  → 1.1.1 Information Gathering; 1.4.1 Direct Action Requests
- A fabricated manager request asks the reader to purchase stored-value cards.
  → 1.2.2 Indirect Resource Acquisition; 1.4.1 Direct Action Requests
- A synthetic billing scenario uses a department pretext to request payment details.
  → 1.1.1 Information Gathering; 1.3.2 Legitimacy Creation
"""


IMPLICIT_DEFINITIONS = """
IMPLICIT INTENTS — psychological persuasion mechanisms that indirectly influence behavior.
Framework: Ferreira & Teles (2019), synthesising Cialdini (1993), Gragg (2003),
Stajano & Wilson (2011).

STRICT CRITERIA: Surface-level professional language alone does NOT qualify.
Each label requires specific psychological mechanism evidence, not just general tone.
Human annotators achieved κ=0.111 on these labels — deliberate, conservative annotation
is required. When uncertain, do NOT annotate the category.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.1 AUTHORITY (Cialdini 1993)
Exploits conditioned tendency to submit to authority figures or authority signals.
Hierarchical language, organizational titles, power dynamics, institutional authority.

  ✓ "As the leader of our engineering team, your expertise and swift action are crucial"
     (invokes recipient's authority role to create obligation)
  ✓ "The director has approved this request and requires your immediate action"
  ✓ Consequences for non-compliance: "Failure to update will result in account suspension"
  ✗ Generic professional titles in a signature (no directive force)
  ✗ "Please complete this task" (polite request, no power dynamic)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.2 SOCIAL PROOF (Cialdini 1993)
Exploits conformity mechanisms and social validation. Three sub-types:

  2.2.1 Herd Mentality — "everyone/most have already done X"
    ✓ "Most of your colleagues have already completed this training"
    ✓ "Over 500 employees have already updated their credentials"
    ✗ Collective pronouns alone ("we", "our team") without behavioral norm claim

  2.2.2 Diffusion of Responsibility — collective framing dilutes individual agency
    ✓ "We kindly request ALL employees, including yourself, to review this"
    ✓ "This applies to the entire team — everyone must participate"
    ✗ "We ask for your collaboration" (individual request with "we" subject)

  2.2.3 Moral Duty — prosocial appeal; compliance benefits others
    ✓ "Your action is critical to protect your colleagues from this threat"
    ✓ "The safety of our organization and its members is of utmost importance"
    ✗ "This is important for the company" (importance claim without altruistic framing)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.3 LIKING, SIMILARITY AND DECEPTION
Interpersonal attraction mechanisms and deception (Ferreira & Teles 2019). Three sub-types:

  2.3.1 General Deception — false scenarios, fabricated emergencies, fictitious situations
    ✓ "There is USD$15 million in an abandoned account belonging to a deceased client"
    ✓ "We have detected a critical vulnerability requiring your immediate attention"
       (when the vulnerability is fabricated)
    ✗ Plausible but unverified claims (deception requires objective falsity or fabrication)

  2.3.2 Deceptive Relationships — falsified connections, fake prior interactions, fake referrals
    ✓ "Your colleague Isabelle recommended I contact you"
    ✓ "As per our previous conversation last Tuesday..."
    ✓ "I am the personal attorney of your late uncle Mr. X"
    ✗ Generic corporate relationship ("We at SafeSecurity have worked with your team")

  2.3.3 Liking and Similarity — exploits genuine/claimed shared characteristics for rapport
    ✓ "As a cybersecurity expert like yourself, I understand the urgency"
    ✓ "Fellow engineers know how critical these security protocols are"
    ✓ Flattery based on recipient's real role/expertise used to lower their guard
    ✗ Generic professional courtesy ("We appreciate your continued collaboration")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.4 DISTRACTION
Diverts attention from critical evaluation; exploits cognitive biases (Ferreira & Teles 2019).
Four mechanisms:

  2.4.1 Scarcity — time or resource constraints creating urgency (Cialdini 1993)
    Time: ✓ "Confirm your information within the next 24 hours"
          ✓ "This offer expires at midnight" / "Respond immediately" / "Last chance"
    Resource: ✓ "Only available to a select group of experts"
              ✓ "Limited spots remaining — register now"
    ✗ "Please respond at your earliest convenience" (polite, no artificial pressure)

  2.4.2 Overloading — cognitive overload via time pressure, info complexity, multiple demands
    ✓ "Review the 25-page protocol and complete tasks A, B, C, D before end of day"
    ✓ Dense, complex information requiring immediate decision deliberately exceeding capacity
    ✗ Detailed but functional information (length alone ≠ overloading)

  2.4.3 Strong Affect — deliberate emotional state manipulation bypassing rational judgment
    ✓ "Your account has been locked due to suspicious activity" (fear)
    ✓ "A critical security vulnerability has been identified" + urgent action (alarm)
    ✓ "You've been selected for an exclusive reward" (excitement/greed)
    ✗ "This is an important update" (importance claim without emotional trigger)

  2.4.4 Need and Greed — appeals to wealth, status, or exceptional personal reward
    ✓ "You will receive 30% of USD$15 million upon completion"
    ✓ "You have been pre-selected for this exclusive financial opportunity"
    ✗ "This will benefit your professional development" (career ≠ greed)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.5 COMMITMENT, INTEGRITY AND RECIPROCATION
Exploits psychological consistency and social exchange norms (Cialdini 1993; Stajano & Wilson 2011).

  2.5.1 Integrity — exploits presumption of honesty; sender signals their own transparency
    ✓ "I am revealing this to you in complete confidence and good faith"
    ✓ "I beg you to carefully go through this email — if it does not meet your ethics,
       please accept my apologies in advance"
    ✗ "We take your privacy seriously" (policy statement, not personal integrity claim)

  2.5.2 Consistency — appeals to recipient's past behavior, stated values, or established role
    ✓ "As someone who has always prioritized security, you'll understand the urgency"
    ✓ "We appreciate your continued dedication and commitment to excellence. Your
       contributions have been invaluable..." (invokes recipient's identity to create obligation)
    ✗ "As a cybersecurity expert, please review this" (role reference for relevance, not consistency pressure)

  2.5.3 Commitment — obligation framing via past agreements, role expectations, or coercion
    Coercion: ✓ "If you do not confirm your information, your account will remain limited"
              ✓ "Your prompt action is required — failure to respond will trigger escalation"
    Other:    ✓ "As department head, you are naturally responsible for this process"
              ✓ "Conformément à l'accord que vous avez signé..." (prior agreement invoked)
    ✗ "We ask for your prompt response" (polite request ≠ obligation)

  2.5.4 Reciprocation — prior favor/service creates felt obligation to return it (Cialdini 1993)
    ✓ "I am sharing this confidential information with you and trust you will reciprocate"
    ✓ "I am reaching out following the impressive information I found about you online [...]
       I will not fail to bring to your notice that this transaction is hitch-free"
       (effort of finding you = implied favor)
    ✗ "Thank you for your collaboration" (gratitude ≠ reciprocation unless obligation created)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYNTHETIC WORKED EXAMPLES (summaries, not email text):

- A fabricated access notice combines institutional authority, fear, and a deadline.
  → 2.1 Authority; 2.4.1 Scarcity; 2.4.3 Strong Affect
- A fabricated prior relationship is used to request stored-value cards under time pressure.
  → 2.1 Authority; 2.4.1 Scarcity; 2.5.2 Consistency
- A synthetic billing scenario uses a department pretext while requesting payment details.
  → 2.1 Authority; 2.5.1 Integrity; 2.5.4 Reciprocation
- A coercive notice threatens account limits if a recipient does not confirm information.
  → 2.5.3 Commitment
"""


FINE_GRAINED_RULES = """
Fine-grained sub-label assignment rules (only for active high-level categories):

EXPLICIT SUB-LABELS:
  information_manipulation:
    → information_gathering:   explicit question or request for specific data/credentials
    → information_control:     instructions about permissions, system access, credential management

  resource_acquisition:
    → direct_financial_gain:         money transfer, bank details, payment credentials
    → indirect_resource_acquisition: gift cards, equipment, admin rights, non-monetary value

  trust_establishment:
    → identity_establishment:  sender introduces name/role/credentials/organizational position
    → legitimacy_creation:     references known org, official process, brand name, compliance

  call_to_action:
    → direct_action_requests:       imperative + specific immediate task (click, fill, send, confirm)
    → indirect_action_facilitation: sets up future required action without issuing the direct command

IMPLICIT SUB-LABELS:
  authority:
    → authority: hierarchical power, credential-based authority, consequence of non-compliance

  social_proof:
    → herd_mentality:           "most/everyone already did X"
    → diffusion_of_responsibility: collective framing reducing individual agency
    → moral_duty:               prosocial/altruistic appeal benefiting others

  liking_similarity_deception:
    → general_deception:        objectively false or fabricated scenarios/claims
    → deceptive_relationships:  fake prior relationship, fabricated referral, false familiarity
    → liking_and_similarity:    genuine shared identity/expertise exploited for manipulative rapport

  distraction:
    → scarcity:                 deadline, urgency marker, limited quantity, or exclusive access claim
    → overloading:              multiple simultaneous demands, deliberate information overload
    → strong_affect:            fear, alarm, excitement — emotion triggering bypass of analysis
    → need_and_greed:           exceptional financial reward, status, or exclusive opportunity

  commitment_integrity_reciprocation:
    → integrity:        sender's explicit honesty/transparency claim to lower guard
    → consistency:      invokes recipient's established identity, role, or past commitments
    → commitment:       obligation via consequence of non-compliance or role-based expectation
    → reciprocation:    prior favor/service creates felt obligation to return it
"""


# ── DSPy Signatures ───────────────────────────────────────────────────────────

class ExplicitIntentSignature(dspy.Signature):
    """
    Classify explicit intents in this email using the DIDECO taxonomy
    (Popovic et al. 2026, grounded in Speech Act Theory, Searle 1969).
    Explicit intents are directly expressed communicative goals.
    Mark True only when clear textual evidence supports the category.
    """
    definitions: str = dspy.InputField(desc="Explicit intent definitions, criteria, and examples")
    email_text: str = dspy.InputField(desc="Full email text to classify")

    information_manipulation: bool = dspy.OutputField(
        desc="1.1 — Direct requests for data/credentials (Gathering) OR instructions to manage system access/information flow (Control)"
    )
    resource_acquisition: bool = dspy.OutputField(
        desc="1.2 — Requests for money/financial assets (Direct) OR non-monetary resources like gift cards, equipment, admin rights (Indirect)"
    )
    trust_establishment: bool = dspy.OutputField(
        desc="1.3 — Sender asserts identity/role/credentials (Identity Establishment) OR invokes known org/brand/official process for credibility (Legitimacy Creation)"
    )
    call_to_action: bool = dspy.OutputField(
        desc="1.4 — Explicit instruction for an immediate specific action (Direct) OR preparatory step enabling a future required action (Indirect Facilitation)"
    )


class ImplicitIntentSignature(dspy.Signature):
    """
    Classify implicit intents in this email using the DIDECO taxonomy
    (Ferreira & Teles 2019 framework, synthesising Cialdini 1993, Gragg 2003, Stajano & Wilson 2011).
    Implicit intents are psychological persuasion mechanisms — they influence behavior
    indirectly without stating the manipulation explicitly.

    STRICT ANNOTATION: Human IAA on these categories was κ=0.111, reflecting genuine
    difficulty. Apply conservative criteria — only annotate when specific mechanism
    evidence is present. Surface-level professional language does NOT qualify.
    """
    definitions: str = dspy.InputField(desc="Implicit intent definitions with strict criteria and worked examples")
    email_text: str = dspy.InputField(desc="Full email text to classify")

    authority: bool = dspy.OutputField(
        desc="2.1 — Exploits hierarchical authority, uses credentials/titles directively, or implies consequences for non-compliance (Cialdini 1993)"
    )
    social_proof: bool = dspy.OutputField(
        desc="2.2 — Herd mentality (group has done X), diffusion of responsibility (all employees), OR moral duty (compliance protects others)"
    )
    liking_similarity_deception: bool = dspy.OutputField(
        desc="2.3 — General deception (fabricated scenarios), deceptive relationships (fake prior bond), OR liking/similarity (genuine shared traits exploited for rapport)"
    )
    distraction: bool = dspy.OutputField(
        desc="2.4 — Scarcity (time/resource), overloading (cognitive overload), strong affect (fear/alarm/excitement), OR need and greed (exceptional financial reward)"
    )
    commitment_integrity_reciprocation: bool = dspy.OutputField(
        desc="2.5 — Integrity signal (sender claims honesty), consistency (invokes recipient's past behavior/role), commitment (obligation/coercion), OR reciprocation (prior favor creates debt)"
    )


class FineGrainedSignature(dspy.Signature):
    """
    Assign fine-grained sub-labels for the active high-level categories.
    Only classify sub-labels belonging to categories already confirmed True above.
    """
    rules: str = dspy.InputField(desc="Fine-grained sub-label assignment rules")
    email_text: str = dspy.InputField(desc="Full email text")
    active_categories: str = dspy.InputField(
        desc="Comma-separated list of high-level categories confirmed True"
    )

    # Explicit sub-labels
    information_gathering: bool = dspy.OutputField(desc="1.1.1 — Explicit request for specific data, credentials, or confirmation")
    information_control: bool = dspy.OutputField(desc="1.1.2 — Instructions to modify access, permissions, or system settings")
    direct_financial_gain: bool = dspy.OutputField(desc="1.2.1 — Money transfer, bank details, payment credentials")
    indirect_resource_acquisition: bool = dspy.OutputField(desc="1.2.2 — Gift cards, equipment, admin rights, non-monetary value")
    identity_establishment: bool = dspy.OutputField(desc="1.3.1 — Sender introduces name, role, credentials, or organizational position")
    legitimacy_creation: bool = dspy.OutputField(desc="1.3.2 — References known org, brand, official process, or compliance requirement")
    direct_action_requests: bool = dspy.OutputField(desc="1.4.1 — Imperative + specific immediate task (click, fill, send, confirm)")
    indirect_action_facilitation: bool = dspy.OutputField(desc="1.4.2 — Sets up a future required action without the direct command yet")

    # Implicit sub-labels
    authority_signal: bool = dspy.OutputField(desc="2.1 — Hierarchical power, credential-based authority, or consequence of non-compliance")
    herd_mentality: bool = dspy.OutputField(desc="2.2.1 — 'Most/everyone already did X'")
    diffusion_of_responsibility: bool = dspy.OutputField(desc="2.2.2 — Collective framing reducing individual agency")
    moral_duty: bool = dspy.OutputField(desc="2.2.3 — Prosocial/altruistic appeal — compliance benefits others")
    general_deception: bool = dspy.OutputField(desc="2.3.1 — Objectively false or fabricated scenarios/claims")
    deceptive_relationships: bool = dspy.OutputField(desc="2.3.2 — Fake prior relationship, fabricated referral, false familiarity")
    liking_and_similarity: bool = dspy.OutputField(desc="2.3.3 — Genuine shared identity/expertise exploited for manipulative rapport")
    scarcity: bool = dspy.OutputField(desc="2.4.1 — Artificial time pressure or resource limits that prompt action")
    overloading: bool = dspy.OutputField(desc="2.4.2 — Multiple simultaneous demands or deliberate information overload")
    strong_affect: bool = dspy.OutputField(desc="2.4.3 — Fear, alarm, or excitement triggering bypass of rational analysis")
    need_and_greed: bool = dspy.OutputField(desc="2.4.4 — Exceptional financial reward, status, or exclusive opportunity appeal")
    integrity: bool = dspy.OutputField(desc="2.5.1 — Sender's explicit honesty/transparency claim")
    consistency: bool = dspy.OutputField(desc="2.5.2 — Invokes recipient's established identity, role, or past commitments")
    commitment: bool = dspy.OutputField(desc="2.5.3 — Obligation via consequence of non-compliance or role-based expectation")
    reciprocation: bool = dspy.OutputField(desc="2.5.4 — Prior favor/service creates felt obligation to return it")


# ── Hierarchical Annotator ────────────────────────────────────────────────────

class HierarchicalAnnotator:
    """
    Two-step hierarchical annotator mirroring the DIDECO taxonomy structure.
    Contrasts with FlatAnnotator for RQ2 (strategy comparison).

    Step 1: Explicit intent classification (4 categories, Speech Act Theory)
    Step 2: Implicit intent classification (5 categories, Ferreira & Teles 2019)
    Step 3: Fine-grained sub-label classification (only for active categories)
    """

    def __init__(self, lm, fine_grained: bool = True):
        self.lm = lm
        self.fine_grained = fine_grained
        dspy.configure(lm=lm)
        self.explicit_predictor = dspy.Predict(ExplicitIntentSignature)
        self.implicit_predictor = dspy.Predict(ImplicitIntentSignature)
        if fine_grained:
            self.fine_predictor = dspy.Predict(FineGrainedSignature)

    def annotate_email(self, email: dict) -> dict:
        text = email.get("text") or f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}".strip()

        # Step 1: Explicit
        exp = self.explicit_predictor(definitions=EXPLICIT_DEFINITIONS, email_text=text)
        explicit = {
            "information_manipulation": bool(exp.information_manipulation),
            "resource_acquisition":     bool(exp.resource_acquisition),
            "trust_establishment":      bool(exp.trust_establishment),
            "call_to_action":           bool(exp.call_to_action),
        }

        # Step 2: Implicit
        imp = self.implicit_predictor(definitions=IMPLICIT_DEFINITIONS, email_text=text)
        implicit = {
            "authority":                          bool(imp.authority),
            "social_proof":                       bool(imp.social_proof),
            "liking_similarity_deception":        bool(imp.liking_similarity_deception),
            "distraction":                        bool(imp.distraction),
            "commitment_integrity_reciprocation": bool(imp.commitment_integrity_reciprocation),
        }

        annotations = {"explicit": explicit, "implicit": implicit}

        # Step 3: Fine-grained (only for active categories)
        if self.fine_grained:
            active = [k for k, v in {**explicit, **implicit}.items() if v]
            fine_result = self.fine_predictor(
                rules=FINE_GRAINED_RULES,
                email_text=text,
                active_categories=", ".join(active) if active else "none",
            )
            annotations["fine_grained"] = {
                # Explicit
                "information_gathering":         bool(fine_result.information_gathering),
                "information_control":           bool(fine_result.information_control),
                "direct_financial_gain":         bool(fine_result.direct_financial_gain),
                "indirect_resource_acquisition": bool(fine_result.indirect_resource_acquisition),
                "identity_establishment":        bool(fine_result.identity_establishment),
                "legitimacy_creation":           bool(fine_result.legitimacy_creation),
                "direct_action_requests":        bool(fine_result.direct_action_requests),
                "indirect_action_facilitation":  bool(fine_result.indirect_action_facilitation),
                # Implicit
                "authority":                     bool(fine_result.authority_signal),
                "herd_mentality":                bool(fine_result.herd_mentality),
                "diffusion_of_responsibility":   bool(fine_result.diffusion_of_responsibility),
                "moral_duty":                    bool(fine_result.moral_duty),
                "general_deception":             bool(fine_result.general_deception),
                "deceptive_relationships":       bool(fine_result.deceptive_relationships),
                "liking_and_similarity":         bool(fine_result.liking_and_similarity),
                "scarcity":                      bool(fine_result.scarcity),
                "overloading":                   bool(fine_result.overloading),
                "strong_affect":                 bool(fine_result.strong_affect),
                "need_and_greed":                bool(fine_result.need_and_greed),
                "integrity":                     bool(fine_result.integrity),
                "consistency":                   bool(fine_result.consistency),
                "commitment":                    bool(fine_result.commitment),
                "reciprocation":                 bool(fine_result.reciprocation),
            }

        return annotations

    def annotate_dataset(self, emails: list, verbose: bool = True) -> list:
        from datetime import datetime
        results = []
        n = len(emails)
        for i, email in enumerate(emails, 1):
            eid = email.get("id", f"email_{i}")
            if verbose:
                print(f"  [{i}/{n}] {eid}", end="", flush=True)
            try:
                ann = self.annotate_email(email)
                exp_pos = sum(1 for v in ann["explicit"].values() if v)
                imp_pos = sum(1 for v in ann["implicit"].values() if v)
                if verbose:
                    print(f" → explicit={exp_pos}/4  implicit={imp_pos}/5")
                result = {"id": eid, "annotations": ann}
            except Exception as e:
                if verbose:
                    print(f" ERROR: {e}")
                result = {"id": eid, "annotation_error": True}
            results.append(result)
        return results
