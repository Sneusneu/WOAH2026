# Refined Implicit Intent Annotation Guidelines
## Supplement to DIDECO v1 Guidelines (Popovic et al., LREC 2026)
### For WOAH 2026 Re-annotation Round

**Purpose**: Address the systematic ambiguity that produced κ=0.111 for implicit
intents in the DIDECO annotation campaign (Appendix D). These refined guidelines
add hard decision criteria, explicit boundary conditions, and negative examples
grounded in the paper's theoretical framework.

**Theoretical basis**: Ferreira & Teles (2019), synthesising Cialdini (1993)
on Authority/Social Proof/Scarcity/Reciprocation, Gragg (2003) on social
engineering, and Stajano & Wilson (2011) on scam victim psychology.

---

## Root cause analysis of κ=0.111

From inter-annotator disagreement patterns across rounds 1–5:

1. **Confounding professional tone with manipulation**: annotators disagreed
   on whether standard business language triggers implicit categories
2. **Overlap between sub-categories**: the same textual evidence was assigned
   to different sub-categories by different annotators (e.g., Consistency vs.
   Commitment; Integrity vs. Legitimacy Creation)
3. **Missing negative examples**: the v1 guidelines define each category but
   rarely show what definitively does NOT count

These guidelines fix all three by adding hard decision questions, exclusion
checklists, and worked negative examples for every category.

---

## Revised annotation protocol for implicit intents

**Old protocol**: Annotate any category whose definition could plausibly apply.

**New protocol**:
1. Read the full email
2. For each implicit category, answer the **decision question** (below)
3. If YES → annotate AND note the specific text span that triggered it
4. If MAYBE → apply the **exclusion checklist**; if text matches any
   exclusion pattern → do NOT annotate
5. If multiple sub-categories apply → annotate all (not mutually exclusive)
6. If still uncertain → **default to NOT annotating**

The new protocol is deliberately conservative. Raising precision at the cost
of some recall is the right trade-off here, since false positives were the
dominant IAA problem.

---

## 2.1 AUTHORITY (Cialdini 1993)

**Definition**: Exploits conditioned submission to authority figures or signals.
Hierarchical language, organizational titles, power dynamics, institutional authority.

**Decision question**: *Does the email invoke a power relationship — through
the sender's position over the recipient, the recipient's role-based obligations,
or consequences for non-compliance — to compel action?*

### Must have at least ONE:
- Position/title language combined with a directive ("As your manager...", "The CEO requires...")
- Explicit or implicit consequences for non-compliance ("account will be suspended", "escalation will follow")
- Credentials used to justify a request with implied non-optional compliance

### Worked positive examples:
- "As the leader of our engineering team, your expertise and swift action are crucial
  in resolving this issue." → Authority (recipient's role invoked as obligation)
- "The director has approved this request. Please comply immediately."
  → Authority (hierarchical cascade + consequence implied)
- "Your account has been locked due to suspicious activity." [from "PayPal Security"]
  → Authority (institutional authority of a major organization)

### Exclusion checklist — do NOT annotate if only:
| Pattern | Why NOT Authority |
|---|---|
| Signature includes a job title | Title in signature ≠ authority unless used directively |
| "We appreciate your expertise" | Flattery without a power-backed directive |
| "Please complete this form" | Polite request; no power dynamic |
| Professional tone throughout | Tone alone ≠ authority |

### Boundary: Authority vs. Distraction (2.4.3 Strong Affect)
- **Authority**: power *relationship* compels action → "As your manager, I need this now"
- **Distraction**: urgency/fear compels action without hierarchy → "Your account will be deleted"
- Co-occur when both elements are present simultaneously

---

## 2.2 SOCIAL PROOF (Cialdini 1993)

**Definition**: Exploits conformity mechanisms. Three sub-types with distinct criteria.

**Decision question**: *Does the email use group behavior, collective framing,
or altruistic obligation to lower the recipient's individual resistance?*

### 2.2.1 Herd Mentality
**STRICT**: Must claim or imply *a group has already performed the requested action*.

✓ "Most of your colleagues have already completed this security training"
✓ "Over 500 employees have already updated their credentials this week"
✗ "Our team works hard together" (collective language without behavioral claim)
✗ "All employees are required to do this" → this is Diffusion of Responsibility, not Herd

### 2.2.2 Diffusion of Responsibility
**STRICT**: Collective framing that *dilutes individual agency* by embedding
the request in a group process.

✓ "We kindly request ALL employees, including yourself, to complete this"
✓ "This security protocol applies to the entire organization — no exceptions"
✗ "We ask you to review this document" (individual ask with plural subject)

**Boundary with Herd Mentality**: Herd = "others have done it"; Diffusion = "you're one of many who must"

### 2.2.3 Moral Duty
**STRICT**: Frames compliance as *benefiting others* or fulfilling an *ethical obligation*.

✓ "Your participation is critical to protect your colleagues from this threat"
✓ "The safety of all 300 employees depends on everyone completing this verification"
✗ "This is important for the company" (organizational importance ≠ moral duty)
✗ "Your contribution is valued" (appreciation ≠ moral duty)

### Shared exclusion: collective pronouns alone
"We", "our team", "the organization" are insufficient on their own.
The framing must create a *behavioral norm*, *dilute responsibility*, or *invoke altruism*.

---

## 2.3 LIKING, SIMILARITY AND DECEPTION (Ferreira & Teles 2019)

High-confusion category. Use this priority order to avoid double-annotation:

**Priority 1 → General Deception** if objectively false/fabricated claims exist
**Priority 2 → Deceptive Relationships** if a fake relationship is constructed
**Priority 3 → Liking and Similarity** if genuine shared traits are exploited

### 2.3.1 General Deception
**STRICT**: Claims that are *objectively false* or *materially fabricated* and
presented as fact (not merely unverified).

✓ "There is USD$15 million in an abandoned account belonging to your deceased relative"
✓ Fabricated security alerts from organizations the sender has no connection to
✗ "We have strengthened our security protocols" (plausible claim, may be true)
✗ "We detected unusual activity" (could be genuine; not demonstrably false)

**Key test**: Would a reasonable observer, knowing the sender's actual identity,
recognize this as a false fabrication? If yes → General Deception.

### 2.3.2 Deceptive Relationships
**STRICT**: Sender *constructs a prior or existing relationship that does not exist*.

✓ "Your colleague Isabelle recommended I contact you regarding this opportunity"
✓ "As per our conversation at the conference last week..." (no such conversation occurred)
✓ "I am the personal attorney of your late uncle Mr. X" (fictional relationship)
✗ "We at SafeSecurity have been supporting your team" (organizational, not fake personal)

**Boundary with Liking**: Deceptive = *fabricated* bond; Liking = *genuine* shared
trait exploited.

### 2.3.3 Liking and Similarity
**STRICT**: Exploits *genuinely real or plausible* shared traits, professional identity,
or interests to build manipulative rapport.

✓ "As a cybersecurity expert like yourself, I understand the urgency of this alert"
   (recipient IS a security expert — real trait leveraged)
✓ "Fellow engineers know how critical these access protocols are to system integrity"
✗ "We appreciate your continued collaboration" (generic courtesy, not affinity exploitation)
✗ "Your dedication to excellence is commendable" (flattery without genuine shared trait)

**Key test**: Does the email reference something *verifiably true about the recipient*
as a basis for rapport and compliance?

---

## 2.4 DISTRACTION (Ferreira & Teles 2019)

**Decision question**: *Does the email use time pressure, resource scarcity,
cognitive overload, or emotional manipulation to bypass the recipient's
rational evaluation of the request?*

### 2.4.1 Scarcity (Time or Resource)

**Time Scarcity** — explicit deadline, counter, or urgency marker serving no other purpose:
✓ "Confirm your information within the next 24 hours or your account will be restricted"
✓ "Respond immediately" / "Last chance" / "Expires tonight"
✗ "Please respond at your earliest convenience" (polite, not artificial urgency)

**Decision rule**: Remove the time reference. If the factual request is unchanged,
the time pressure is artificial → Time Scarcity.

**Resource Scarcity** — limited quantity or restricted access motivating action:
✓ "Only available to a select group of industry experts — you qualify"
✓ "Only 3 spots remaining in this exclusive program"
✗ "This is a confidential communication" (confidentiality ≠ scarcity)

### 2.4.2 Overloading
**STRICT**: Deliberately excessive complexity or multiple simultaneous demands
*designed* to reduce critical thinking.

✓ Multiple deadlines + dense technical content + immediate required action
✓ "Review the 25-page security protocol and complete verification by end of day"
✗ A long but functional email with relevant information (length ≠ overloading)

**Key test**: Is the information volume functional (recipient needs it all) or
strategic (designed to overwhelm)? Only the latter qualifies.

### 2.4.3 Strong Affect
**STRICT**: Deliberate triggering of fear, alarm, or strong excitement that
*impairs rational judgment*.

✓ "Your account has been locked due to suspicious activity" (fear → bypasses analysis)
✓ "A critical security vulnerability has been identified — immediate action required" (alarm)
✗ "This is an important update regarding our security protocols" (importance ≠ affect)
✗ "We hope you find this information useful" (neutral tone)

**Boundary with Scarcity**: Strong Affect targets *emotional state*; Scarcity targets
*time/quantity window*. A fear appeal about a deadline triggers both simultaneously.

### 2.4.4 Need and Greed
**STRICT**: Explicit appeal to *exceptional* financial reward, status, or privilege.

✓ "Upon completion, you will be entitled to 30% of USD$15 million"
✓ "You have been pre-selected for this exclusive investment opportunity"
✗ "This will benefit your professional development" (career benefit ≠ greed)
✗ "Your expertise is valued" (flattery without exceptional material promise)

---

## 2.5 COMMITMENT, INTEGRITY AND RECIPROCATION (Cialdini 1993; Stajano & Wilson 2011)

**Highest annotator confusion**: Integrity and Consistency are frequently conflated.
Use the sub-type definitions strictly.

### 2.5.1 Integrity
**STRICT**: Sender makes an *explicit claim about their own honesty or good faith*
to lower the recipient's defenses.

✓ "I am revealing this to you in complete confidence and good faith"
✓ "I beg you to carefully review this — should it not meet your ethics, accept my apologies"
✗ "We take your privacy seriously" (organizational policy ≠ personal integrity claim)
✗ "This is a legitimate business opportunity" (legitimacy claim belongs under 1.3.2)

**Boundary with Trust Establishment (1.3.2)**: Integrity = sender claims *their own*
honesty; Legitimacy Creation = sender invokes *an organization's* credibility.

### 2.5.2 Consistency
**STRICT**: Invokes the *recipient's* established identity, role, or past behavior to
create cognitive consistency pressure — implying that non-compliance would be
inconsistent with who they are.

✓ "As someone who has always prioritized security, you'll understand why immediate
   action is essential" (invokes recipient's identity)
✓ "We appreciate your continued dedication to excellence. Your contributions have been
   invaluable..." (followed by a request — invokes identity as a springboard)
✗ "As a cybersecurity expert, please review this document" (role reference for relevance,
   not consistency pressure — no implication that refusing would be inconsistent)

**Key test**: Does the email imply the recipient *would be acting against their own
values/identity* if they don't comply?

### 2.5.3 Commitment
**STRICT**: Creates a *perceived obligation* through coercive framing, role-based
expectations, or references to prior agreements.

Coercion: ✓ "If you do not confirm your information, your account will remain limited"
           ✓ "Your prompt action is required — failure to respond will trigger escalation"
Role:     ✓ "As department head, you are naturally responsible for this compliance process"
Prior:    ✓ "As per the agreement signed last week, we need you to complete this step"
✗ "We ask for your prompt response" (polite request without obligation framing)

**Boundary with Authority (2.1)**: Authority = obligation from an *external* power figure;
Commitment = obligation from *role/past agreement/consequence* — can co-occur.

### 2.5.4 Reciprocation
**STRICT**: A *prior favor, service, or information* is referenced (explicitly or
implicitly) to create a felt obligation to return it.

✓ "I am sharing this confidential information with you and trust you will reciprocate
   by keeping our arrangement discreet"
✓ "Following the impressive research I conducted about your background, I am confident
   you are the right person..." (effort of research = implied investment creating debt)
✓ "Your company has benefited from our services — we now ask for your assistance"
✗ "Thank you for your collaboration" (gratitude ≠ reciprocation unless debt is created)
✗ "We look forward to our continued partnership" (future orientation, no prior favor)

---

## Quick boundary reference table

| If the email invokes... | Primary | Secondary? |
|---|---|---|
| Sender's hierarchical position + directive | Authority (2.1) | Commitment (2.5.3) |
| Consequence for non-compliance | Authority (2.1) / Commitment (2.5.3) | — |
| "Most colleagues already did X" | Social Proof — Herd (2.2.1) | — |
| "All employees must do this" | Social Proof — Diffusion (2.2.2) | — |
| "Protects your colleagues" | Social Proof — Moral Duty (2.2.3) | — |
| Objectively false fabrication | Liking/Decept — General Deception (2.3.1) | — |
| Fake prior relationship/referral | Liking/Decept — Deceptive Rel. (2.3.2) | — |
| Recipient's real traits for rapport | Liking/Decept — Liking/Similarity (2.3.3) | — |
| Artificial time pressure | Distraction — Time Scarcity (2.4.1) | Strong Affect? |
| Limited availability claim | Distraction — Resource Scarcity (2.4.1) | — |
| Fear/alarm/excitement | Distraction — Strong Affect (2.4.3) | Time Scarcity? |
| Exceptional financial reward | Distraction — Need & Greed (2.4.4) | Gen. Deception? |
| Sender claims own honesty | Commit./Integ. — Integrity (2.5.1) | — |
| Recipient's identity/past behavior | Commit./Integ. — Consistency (2.5.2) | — |
| Obligation via consequence/role | Commit./Integ. — Commitment (2.5.3) | Authority? |
| Prior favor creates debt | Commit./Integ. — Reciprocation (2.5.4) | — |
