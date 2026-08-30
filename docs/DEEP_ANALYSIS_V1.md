# V1 → V2: what the Round 1 submission got right, wrong, and left undefined

This document audits the Round 1 deck against what Round 2 required, and records what
changed. It is written to be useful rather than flattering.

**Summary judgement:** V1 identified the problem correctly and specified the solution
loosely. Almost every V1 claim survived into V2 — but most of them survived *with a
mechanism attached*, and three had to be substantially rethought.

---

## 1. What V1 got right, and kept

| V1 claim | Verdict |
|---|---|
| Triage is an event; deterioration is continuous — "the chasm" | **The whole thesis.** Kept verbatim and made the object of computation |
| Three wasted signals: nurse, camera, family | **Correct and underrated.** The attendant channel is the strongest idea in the submission |
| Zero per-patient hardware | Kept. Existing cameras, the family's own phone |
| Fusion, not voting — disagreement is the signal | Kept, and **specified** (see §3.5 below) |
| Only a human downgrades | Kept, and **enforced structurally** rather than stated |
| Deterministic red-flag rules fire if all models fail | Kept, and made un-suppressible by type |
| No prize for lying — escalation buys a re-check, not a queue position | Kept, and **enforced in the scoring layer** rather than on a slide |
| Cold start rules-only while collecting prospectively | Kept, and its performance is now **measured and published** |
| rPPG is skin-tone biased; Fitzpatrick-stratified recruitment mandatory | Kept, and promoted from a caveat to a monitored metric with intervals |
| The dual fix — under-triage safety and ED violence, one intervention | Kept, and promoted to a **co-primary outcome** |
| Consent at registration; faces never leave the device | Kept, and made structural — no retrieval path exists |

The Round 1 problem framing needed no correction. That is not nothing: most of the
distance in a project like this is spent discovering you were solving the wrong problem.

---

## 2. The three claims that had to be rethought

### 2.1 "0.80 AUC — visual appearance at the triage desk predicts admission better than most vital scores"

This was the headline evidence claim on the problem slide, and it is the wrong metric
presented the wrong way.

**What is wrong with it:**

- **AUC is discrimination, not calibration.** It summarises ranking across every possible
  threshold. We do not operate at every threshold; we operate at *one*. A model with 0.80
  AUC that is miscalibrated in the elderly is a safety hazard, not an achievement
- **It invites the reader to think the problem is solved by a better classifier.** It is not.
  The problem is that nobody looks again
- **"Predicts admission"** is a different target from *predicts deterioration during the
  wait*, which is a different target from *needs escalation now*. Conflating them is the most
  common quiet error in triage ML

**What V2 does instead.** AUC is reported once, for completeness, and explicitly set aside.
The headline metrics are **sensitivity at the operating point with confidence intervals**,
**under-triage rate**, and **calibration-in-the-large per age band**. The operating point is
fixed by a stated cost ratio, not by Youden's J or F1. See
[CLINICAL_AUDIT.md](CLINICAL_AUDIT.md).

### 2.2 "Sentence Embeddings" in the risk pipeline

V1's stack was drawn as:

```
Multilingual ASR → Sentence Embeddings → XGBoost → rPPG + pose → Temporal Fusion
```

Putting an embedding of the spoken complaint into the feature vector creates two problems
that are not obvious on a slide:

1. **It is a prompt-injection and adversarial-input surface.** The complaint is untrusted
   text. If it reaches the model as a dense vector, "IGNORE ALL PREVIOUS INSTRUCTIONS" is no
   longer inert — it is a perturbation in feature space with unknown effect, and no validator
   downstream can catch it because the damage happened before language was generated
2. **It is unauditable.** "Dimension 214 contributed +0.03" is not something a clinician can
   check, and it cannot be traced to a raw observation the way lineage requires

**What V2 does instead.** Free text is converted to symptoms by **exact-token matching
against a closed, clinician-authored vocabulary**. If a word is not in the vocabulary it
produces nothing. The only thing a complaint can do is name a symptom a clinician already
wrote a rule for. This is less powerful and enormously more defensible, and it is why
`test_injection_cannot_invent_a_symptom` can pass at all.

The cost is real and should be stated: SAATHI understands complaints less richly than an
embedding model would. That is the trade, made deliberately.

### 2.3 "The Ultimate Moat: a proprietary Indian skin-tone, language and clinical gestalt dataset"

This was V1's strategic case, and it does not survive contact with the governance frame it
sits inside.

**The tension.** A moat built on accumulating Indian patient data conflicts with DPDP's
purpose limitation and data minimisation, and it conflicts more directly with the premise of
**BODH** — a federated benchmarking platform whose entire point is that models can be
validated on Indian data *without anyone holding it*. Pitching data accumulation as the
competitive advantage, in a submission that also cites BODH approvingly, is incoherent.

It is also, practically, the claim most likely to be challenged by a judge who works in
health data governance.

**What V2 does instead.** The defensible asset is reframed as:

- The **Clinical Semantic Contract** — clinician-authored, versioned, portable, and the thing
  that actually makes the system safe to deploy at a new site
- The **safety architecture** — invariants, claim compiler, abstention, lineage
- **Validation through BODH**, which produces credibility without custody

Data still improves the model. But *"we will hold Indian patient data and that is our moat"*
became *"we can be validated on Indian data without holding it, and that is our credibility."*
The second is both more honest and, in a market where hospitals must trust you, more useful.

---

## 3. What V1 left undefined, and V2 had to build

V1 named these correctly. Naming is not designing.

### 3.1 Age stratification — the largest gap

**V1 contained no age handling at all.** No age bands, no age-dependent thresholds, no
mention of paediatric or geriatric physiology.

The problem statement calls this out by name, and it is the single most dangerous omission a
triage system can have: a one-size adult-calibrated model applied across all ages introduces
**silent** safety risk — silent because it produces plausible numbers for children and the
elderly that happen to be wrong.

V2 adds seven age bands, per-band thresholds for every vital in the contract, band-relative
z-scoring so age enters as *physiology* rather than as a number, and an explicit geriatric
blunting rule. This is the change that most affected the code: it touches the contract, the
feature builder, the rule engine, and the evaluation.

The evaluation shows it is necessary and insufficient — the elderly bands remain the worst
performers. See [CLINICAL_AUDIT.md](CLINICAL_AUDIT.md).

### 3.2 Signal quality — V1 had no gate

V1's diagram flowed signals straight into fusion. There was no representation of a signal
being **too poor to use**.

V2 adds quality gating with a **reject-not-down-weight** rule, per-method quality floors in
the contract, and the deliberate separation of the *display band* (what the human is told)
from the *accept floor* (what the engine will use).

### 3.3 Uncertainty and abstention — V1's output was a bare score

V1's output layer read `RISK / URGENCY → NURSE RE-CHECK → CLINICIAN`. There is no confidence
in that diagram and no path by which the system declines.

V2 adds a five-component confidence decomposition, five abstention gates, and the
`INSUFFICIENT_SIGNAL` epistemic state, enforced structurally — an abstaining Safety Contract
**cannot be constructed carrying a risk number**. Abstention raises the patient's rank rather
than lowering it.

### 3.4 "Temporal Fusion Model" — named, not specified

V1 listed this in the stack with no definition. V2 implements trajectory estimation as a
**Theil–Sen slope** over quality-passed evidence, computed per `(concept, acquisition_method)`
pair, with the escalation threshold relaxed by method reliability.

Two earlier attempts are worth recording because they failed in instructive ways:
median-of-endpoints lags badly on a ramp and misses exactly the gradual deterioration the
system exists to catch; naive endpoint differencing fires constantly on rPPG noise. Theil–Sen
is robust to the outliers rPPG actually produces.

### 3.5 "Fusion, not voting" — the right slogan, no mechanism

V1 stated the principle and gave the correct example (*a calm-looking patient whose family
reports confusion outranks a restless patient whose family says they are fine*). It did not
say how.

V2 makes **discordance a signed term that enters the score explicitly** and appears in the
decomposition the nurse sees. Escalating discordance raises rank. De-escalating discordance
(`baseline_veto`) can block a *borderline model-threshold* escalation, but only when there is
no material worsening trajectory, no red flag and no critical vital. A family can prevent a
marginal rise; they can never talk down a measured trend.

### 3.6 "No prize for lying" — stated on a slide, now enforced in the layer

V1 said answers are "calibrated per reporter". V2 had to solve the problem that slogan hides:

> Calibrate too aggressively and the channel dies on exactly the reports that matter most.
> New confusion has no sensor counterpart, so a naive calibration learns to ignore it.

Resolution: only a **nurse** adjudicates a report; only **change-claims** are scoreable;
weights are shrunk by `min(1, n/3)` and **floored at 0.30**. And critically — calibration
governs *score influence only*. **Nothing governs the re-check.** Asking always works. Asking
never moves you.

### 3.7 The LLM boundary — absent from V1 entirely

V1's architecture diagram has no language model in it, and no statement of where one would
sit. For a 2026 submission this is the question a judge asks first.

V2 draws the line explicitly, in the architecture and in the UI, and **tests it**: the same
patient assessed with the LLM enabled and disabled produces identical acuity, EWER, red flags
and abstention. Only the prose changes.

### 3.8 The claim compiler and Safety Contract — did not exist in V1

The central defensible innovation of V2 has no V1 antecedent. **What may be said is
determined by what the evidence supports, before any language is generated.** The Safety
Contract is generated by deterministic code; the LLM renders it; a validator rejects anything
outside it and logs the rejection.

### 3.9 Asymmetric cost — absent from V1

V1 never stated that under-triage and over-triage cost differently, which means it implicitly
accepted a 1:1 ratio. V2 makes the ratio explicit (**12:1**), configurable, versioned and
auditable, derives the operating threshold from it, and shows the symmetric comparison
side by side — the artefact where the accuracy-optimised threshold escalates nobody.

### 3.10 Lineage and personas

V1 had neither. V2 adds full lineage from displayed acuity back to the raw observation, and
four personas whose views differ in information depth, evidence selection and available
actions — rendered from the **backend projection**, so the attendant view cannot display an
acuity because the payload it receives does not contain one.

---

## 4. One V1 statement that got stronger, not weaker

> *"82% of family-raised deterioration alerts would not have tripped a standard early warning
> vital score."* (NHS Martha's Rule)

In V1 this was a supporting statistic on the problem slide. In V2 it became a **design
constraint** with consequences throughout the system:

- It is why the attendant channel exists at all
- It is why **materiality can escalate on its own**, without statistical support — a single
  new-onset confusion report has almost no statistical power and enormous clinical
  materiality, and the escalation engine must be able to act on that asymmetry
- It is why reporter calibration is restricted to nurse-adjudicated change-claims — a
  calibration that could be driven by sensor disagreement would learn to discard exactly the
  82%
- It is the reason `STATISTICAL EVIDENCE` and `CLINICAL MATERIALITY` are separate axes that
  are never equated

V1 cited it. V2 is shaped by it.

---

## 5. Honest note on the demo deck's numbers

The Round 1 deck presented several figures — 25–30% LMIC under-triage, 82% of family-raised
alerts, 4:1 attendant ratio, 0.80 AUC for triage-desk gestalt — as context. They are
literature-derived and remain in V2's framing.

**None of them are measurements of SAATHI.** Every performance number in this repository
comes from **simulated data with a stated generative process**, and is labelled as such at
every point of display: in the evaluation report, in the admin view, in the judge surface,
and in the README. The distinction between *the literature motivating the design* and
*evidence about this system* is one V1 did not need to make and V2 must.
