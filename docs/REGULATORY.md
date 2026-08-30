# SAATHI V2 — Regulatory and Data Governance

**Jurisdiction: India.** Every statement below is written for an Indian deployment. Where a
different classification would change the obligations, that is stated rather than glossed.

---

## 1. Positioning: clinical decision support, not a device

SAATHI is positioned as **clinical decision support (CDS)**. The positioning rests on four
properties that are enforced in code, not asserted in a policy document:

| Property | Enforcement |
|---|---|
| It **never diagnoses** | Diagnostic nouns are in the claim grammar's universally-forbidden lexicon. The validator rejects them in every epistemic state |
| It **only raises urgency** | `SafetyContract`'s Pydantic validator refuses to construct a downgrade without a human override |
| It **never acts** | No endpoint moves, discharges or de-prioritises a patient. `test_invariant_8_no_endpoint_discharges_or_removes_a_patient` |
| A **human decides everything** | Every output is advisory. Acuity changes are proposals until accepted; every acceptance and override is recorded |

### What would change if CDSCO classified it as a device

Stated honestly, because "we are CDS" is the claim every clinical AI product makes and the
one regulators examine hardest.

Under the Medical Devices Rules 2017, software intended for diagnosis, prevention,
monitoring or treatment can be a regulated medical device. A triage-support tool that
**recommended a disposition** or **assigned a clinical category autonomously** would be a
much stronger candidate for regulation than one that ranks a queue for human attention.

If SAATHI were classified as a device — likely Class B or C given that a miss carries risk
of serious deterioration — the following would change:

- **Licensing** via the State or Central Licensing Authority, with a notified-body audit
- **ISO 13485** quality management system for the development organisation
- **IEC 62304** software lifecycle compliance, with the safety classification driving the
  documentation depth. The architecture is already built to a 62304 mindset — hazard
  analysis, deterministic safety layer, no model on the safety path — but the *evidence*
  of that process would need to be maintained as controlled documents
- **ISO 14971** risk management file, with the invariant suite as verification evidence
- **Clinical investigation** under the MDR 2017 with an ethics-approved protocol
- **Post-market surveillance** and adverse-event reporting

What would *not* change is the architecture. The deterministic red-flag layer, monotonic
escalation, abstention, the Safety Contract, the claim validator and the audit trail were
designed to survive that classification. The gap is documentation and clinical evidence,
not design.

**We do not claim CDSCO exemption.** We claim a defensible CDS positioning and state the
path if a regulator disagrees.

---

## 2. DPDP Act 2023 — Digital Personal Data Protection

SAATHI processes personal data and health data. The Act applies in full.

### Role

The **hospital is the Data Fiduciary.** SAATHI is deployed on the hospital's
infrastructure (edge device or on-premises) and acts as a **Data Processor** under the
hospital's instruction. This matters: it keeps the fiduciary obligations with the entity
that already holds them for the medical record, and it is why the architecture is
edge-first rather than cloud-first.

### Section-by-section

**Consent (§6).** Captured at registration, in the attendant's own language, before any
camera or attendant channel activates for that patient. Free, specific, informed,
unconditional and unambiguous, with a clear affirmative action. Withdrawal is as easy as
giving — a single control on the attendant device.

**The opt-out must be real, so it costs nothing.** Declining disables the camera and
attendant channels for that patient and **shortens** the nurse re-check interval to
compensate for the lost observation. Queue position is unaffected. Enforced in code and
tested (`test_invariant_7_opting_out_shortens_the_interval_and_costs_no_queue_position`,
demonstrated by cohort case P-018). A consent mechanism whose refusal degrades care is not
consent.

**Purpose limitation (§7).** The stated purpose is continuous triage-safety monitoring
during the ED wait. Derived features are used for that and for the aggregate quality
metrics the hospital needs to govern the tool. They are not used for anything else, and
the contract's `access_restrictions` per concept are the machine-readable expression of
this.

**Data minimisation (§8).** The strongest control in the system:

- **Raw video frames are processed in memory and discarded within the window.** No frame is
  written to disk. No frame leaves the device. There is no function in the codebase that
  returns one, to any role, including operators. `test_there_is_no_endpoint_that_could_
  return_a_frame` asserts the absence of the path — *the absence is the control, not a
  filter on one*
- Only **derived features** persist: a number, its quality, its freshness, its lineage
- Face embeddings for waiting-room re-identification are **session-scoped, encrypted, and
  destroyed at disposition. No cross-visit biometric linkage** — the embedding cannot be
  used to recognise the same person on a later visit, because it no longer exists
- Attendant phone numbers are **hashed** for session linkage. The raw number is stored only
  where the ED already stores contact details, under the hospital's existing obligations
- The LLM receives a redacted, minimised, role-scoped payload with no name, phone number,
  ABHA number, face data or free identifier. `assert_no_pii()` raises rather than permitting
  egress. **The exact payload is displayed in the demo** so it can be checked rather than
  believed

**Accuracy (§8(3)).** Every value carries its acquisition method, signal quality, freshness
and reliability weight. Values below the contract's quality floor are rejected rather than
stored as fact. Stale values are marked and removed from scoring.

**Storage limitation (§8(7)).** Explicit retention periods per data class, with a
documented deletion path implemented in `AuditStore.purge_expired()`:

| Data class | Retention | Rationale |
|---|---|---|
| Raw video frames | **Zero — never persisted** | Not retained at all |
| Face embeddings | Session only, destroyed at disposition | Re-identification within one visit is the only purpose |
| Derived camera features | Short, per contract | Clinical utility expires with the encounter |
| Clinical evidence and lineage | Aligned to the hospital's medical-record retention | It is part of the clinical record of what was shown |
| Audit and access logs | Long — clinical accountability horizon | *"What did the clinician see?"* must be answerable years later |
| Override records | Long, with the full evidence snapshot | The accountability artefact |
| Aggregate quality metrics | Indefinite, no identifiers | Governance |

**Security safeguards (§8(5)).** Edge processing, encryption at rest for the audit store,
role-based access enforced in the backend at three points, and a complete access log with
`role, subject, resource, fields, decision, reason, timestamp` for every decision — including
ALLOWs and redactions, not only denials.

**Breach notification (§8(6)).** The audit log is designed to make the scope of any breach
determinable: which role accessed which fields of which subject, when. A notification that
cannot state scope is not a useful notification.

**Rights of the Data Principal (§11–14).** Access, correction and erasure requests are
served from the audit and lineage stores, which are keyed by patient. Grievance redressal
sits with the hospital as Fiduciary.

**Children's data (§9).** The cohort includes paediatric patients from age 0. Processing
children's data requires **verifiable parental consent** and prohibits tracking and targeted
advertising. SAATHI does neither, and the attendant giving consent for a paediatric patient
is in practice the parent or guardian. This is a point where a real deployment needs a
documented verification procedure rather than an assumption, and we flag it as such.

---

## 3. ABDM / ABHA — Ayushman Bharat Digital Mission

**Prior-record lookup, where it exists, goes through the ABDM consent flow.** Not around it.

- Linkage is via **ABHA**, with consent artefacts issued through the **HIE-CM** (Health
  Information Exchange and Consent Manager)
- Consent is **purpose-scoped and time-bound** — for this encounter, for triage support, for
  a bounded window
- The ABHA number is a `DIRECT_IDENTIFIER` in the contract's field classes and is
  **forbidden egress** to any model. `assert_no_pii()` catches ABHA-shaped tokens
- A patient may decline record linkage and still be triaged. **Roughly half of Indian ED
  arrivals have no prior record at all**, which is why the zero-history path
  (`OBSERVATION_ONLY_v1`) is a first-class model route rather than a fallback

Records retrieved through ABDM are marked with their record version and lookup time and are
subject to the same staleness treatment as any other signal. A prior record months old is
labelled as such, and conflicts between the ED system and the linked record are **surfaced as
contradictory evidence**, never silently reconciled.

---

## 4. SAHI — Strategy for AI in Healthcare for India

SAHI is the governance frame this design is written against. The mapping:

| SAHI principle | How SAATHI answers |
|---|---|
| Safety and reliability | 8 safety invariants as passing tests; deterministic layer beneath every model; fail-safe to FIFO-plus-red-flags |
| Human oversight | Every output advisory; only a human downgrades; override is first-class and acknowledged |
| Transparency | Full lineage from displayed acuity to raw observation; the LLM payload is displayed; EWER components always shown |
| Fairness and non-discrimination | Subgroup performance by age band, sex, Fitzpatrick band, language and attendant-presence is a **reported, monitored metric** with confidence intervals, not a footnote |
| Accountability | Override records carry the full evidence snapshot, contract id and model versions. *Who owns a miss* is answered explicitly in [ADOPTION.md](ADOPTION.md#7-who-owns-a-miss) |
| Privacy | Edge-first, frames discarded, session-scoped biometrics, minimised model payloads |
| Inclusivity | Voice-first multilingual, pictorial fallback for low literacy, SMS/IVR for feature phones, Tier C runs with no camera and no EHR |
| Validation | Prospective protocol defined; federated benchmarking via BODH named as the path |

---

## 5. BODH — the validation path

**BODH** (the IIT Kanpur / National Health Authority federated benchmarking platform) is the
intended third-party validation and audit route.

Why this matters more than any number in this repository:

> SAATHI can be **benchmarked on Indian data without ever holding it.**

Federated benchmarking means the model is evaluated against real Indian emergency-department
data inside the institutions that hold it. Nothing identifiable moves. This resolves the
central tension of the whole project — the model needs Indian data to be trustworthy, and
Indian patients should not have their data pooled to a vendor to get it.

The proposed sequence:

1. **Rules-only cold start.** The deterministic layer runs from day one at any site with no
   learned parameters. Its measured performance is reported honestly, because that is what
   the first month of a deployment actually delivers
2. **Prospective collection** in shadow mode, with the model scoring silently
3. **Federated evaluation through BODH**, per site and per subgroup, with the calibration
   gate applied before any model output is displayed anywhere
4. **Site-specific calibration**, human-reviewed, never auto-retrained on overrides
5. **Continuous subgroup monitoring** post-deployment, with the pre-committed stopping
   conditions in [ADOPTION.md](ADOPTION.md#9-what-would-make-us-stop)

Any pre-training on **MIMIC-IV-ED** or **NHAMCS** is **frozen and treated as a prior**. No
transfer claim is made. Both are US datasets; neither describes an Indian district ED, and
the distribution shift spans case mix, arrival mode, staffing ratios, documentation
practice, baseline physiology and presentation delay. See
[CLINICAL_AUDIT.md](CLINICAL_AUDIT.md) for the full statement of that problem.

---

## 6. Fairness as a regulatory obligation, not a virtue

DPDP does not mandate subgroup performance reporting. SAHI expects it. We treat it as
obligatory because of a specific, concrete risk in this design:

**rPPG signal quality degrades on darker skin tones under standard pipelines.** A system
whose observation quality depends on skin tone, deployed in India, would distribute its
benefit unevenly along exactly the axis a regulator should ask about.

Consequently:

- **Fitzpatrick-stratified recruitment is mandatory** in any prospective validation
- Subgroup SNR is a **monitored metric**, reported per band, not a caveat
- Sensitivity is reported per Fitzpatrick band **with confidence intervals**, and where the
  intervals are too wide to conclude anything, that is what is said
- The same treatment is applied to **attendant present vs absent** — if performance depends
  on whether a family member came along, that is an equity problem, and it belongs on the
  front page of the metrics view rather than in an appendix

The skin-tone gap visible in the current evaluation is **injected by the simulator** to make
the monitoring path visible. It demonstrates that the metric is watched. It is not a
measurement of real-world rPPG bias, and it is labelled as such everywhere it appears.

---

## 7. What is not addressed

- **Cross-border transfer.** Not applicable to the edge-first deployment as designed. A
  cloud-optional configuration would need to satisfy DPDP §16 and is out of scope here
- **Significant Data Fiduciary obligations (§10).** If the deploying hospital is designated
  one, it additionally requires a Data Protection Officer, independent audit and DPIA. The
  architecture supports a DPIA; the designation is the hospital's
- **Insurance and liability.** Out of scope for a prototype, and the accountability model in
  [ADOPTION.md](ADOPTION.md#7-who-owns-a-miss) is the input to that conversation, not a
  substitute for it
- **Formal IEC 62304 / ISO 14971 documentation.** The architecture is built to that mindset;
  the controlled documents do not exist
