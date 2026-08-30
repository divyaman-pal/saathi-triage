# SAATHI V2 — Architecture

Accenture Innovation Challenge 2026 · Round 2 · Track 2 — PatientTriage.ai
Contract v2.0.1 · grammar v1.2 · ruleset v1.3 · cost policy v1.4

---

## 0. The one-sentence version

SAATHI is a deterministic clinical-safety engine with three small models bolted onto it and
a language model bolted onto the far end of *that*, arranged so that removing any model
degrades the output's richness and never its safety.

---

## 1. The problem V2 is built around

Round 1 identified the gap correctly and then, on inspection, did not build for it.

```
0 MIN            2–5 MIN                                          END OF WAIT
ARRIVAL          TRIAGE                THE CHASM                  CLINICIAN
   |                |          nobody looks again                     |
   |                |     ┌──────────────────────────────┐            |
   ●────────────────●─────┤  25–30% LMIC under-triage    ├────────────■
                          │  82% of family-raised alerts  │
                          │  would not trip an EWS score  │
                          └──────────────────────────────┘
```

Triage assigns one fixed label from a two-minute snapshot. Deterioration then proceeds
continuously and unobserved. **The waiting interval is the object of computation in V2.**
Everything below follows from that single decision.

---

## 2. The layered pipeline

Signals do not go straight into a model. Nine layers stand between an observation and a
displayed number, and each one can stop the flow.

```
                         SIGNALS
                            ↓
   ┌────────────────────────────────────────────────────┐
   │ 1  SIGNAL QUALITY GATING     reject, not down-weight│
   │ 2  CLINICAL SEMANTICS        age-band evaluation    │
   │ 3  RISK ESTIMATION           XGBoost + calibration  │
   │ 4  MULTI-CHANNEL FUSION      reliability-weighted   │
   │ 5  DISCORDANCE ANALYSIS      disagreement is signal │
   │ 6  UNCERTAINTY / ABSTENTION  5 gates, any may trip  │
   │ 7  ASYMMETRIC-COST DECISION  threshold from cost    │
   │ 8  PERSONA PRESENTATION      4 projections          │
   │ 9  SAFETY CONTRACT           what may be said       │
   └────────────────────────────────────────────────────┘
                            ↓
                  CONSTRAINED NARRATIVE  ← the only LLM on this path
                            ↓
                     VALIDATOR (rejects and regenerates)
                            ↓
              HUMAN DECISION  (escalate / accept / override)
                            ↓
                    CONTINUOUS RE-CHECK
                            ↓
                        TELEMETRY
```

The measured runtime stage order, as instrumented in `core/pipeline.py`:

```
signal_acquisition → quality_gating → red_flags → feature_assembly → model_inference
→ fusion → confidence → cost_decision → ewer → contract_generation ─┬→ llm_render → validate
                                                                     │
   decision path ends here: p95 = 62.8 ms  ─────────────────────────────┘  narration, droppable
```

Note the position of `red_flags`: **before** `feature_assembly` and `model_inference`. The
deterministic layer runs before any model exists in the request, which is why no model
output can suppress it. `core/red_flags.py` imports no model, and a test asserts that by
inspecting the module's imports.

---

## 3. The Clinical Semantic Contract

Seven YAML files, loaded at runtime, consumed by the scoring engine. **They are not
documentation.** There is no vital-sign threshold anywhere in the Python source.

| File | What it governs |
|---|---|
| `clinical_contract.yaml` | 25 concepts · 7 age bands · per-band thresholds · acquisition methods · reliability weights · quality floors · staleness limits · materiality classes · abstention gates |
| `red_flags.yaml` | 15 clinician-authored rules + 4 time rules, un-suppressible |
| `cost_policy.yaml` | 12:1 asymmetric cost · SLAs per level · re-check modifiers · alert budget · surge posture · materiality matrix |
| `acuity_schemes.yaml` | ESI · MTS · CTAS · LOCAL3 · FREETEXT, with 5 versioned lossy mappings |
| `entitlements.yaml` | RBAC · field classes · retention policy |
| `claim_grammar.yaml` | Epistemic states · forbidden lexicon · validator checks |
| `deployment_profiles.yaml` | Tier A / B / C as data, not a code fork |

### A concept, in full

```yaml
concept_id: RESP_RATE
type: vital
unit: breaths/min
grain: patient x 30s_window
age_band_thresholds:
  age_1_5:      { normal: [20,30], concerning: [31,40], critical: [">40","<15"] }
  age_18_65:    { normal: [12,20], concerning: [21,24], critical: [">24","<8"] }
  age_65_plus:  { normal: [12,20], concerning: [21,24], critical: [">24","<8"],
                  note: "blunted response - absence of tachypnea does not lower risk" }
acquisition_method:
  nurse:     { method: manual_count,    reliability_weight: 1.0 }
  camera:    { method: rppg_rr,         reliability_weight: 0.6, quality_floor: { snr_db: 4.0 } }
  attendant: { method: guided_count_15s, reliability_weight: 0.5, calibration: per_reporter }
max_staleness_minutes: 15
materiality_class: HIGH
escalation_thresholds:
  delta_trigger: "+6 breaths/min over any 20-min window -> mandatory nurse re-check"
access_restrictions: { triage_nurse: full, attendant: none, admin: aggregate_only }
version: 1.2
```

### Red-flag rules are contract objects too

```yaml
rule_id: RF_AIRWAY_STRIDOR
type: red_flag
trigger: "stridor OR visible accessory muscle use OR unable to speak a full sentence"
source_channel: [nurse, camera, attendant]
action: IMMEDIATE_ESCALATE_L1
suppressible_by_model: false
authored_by: clinician
```

`suppressible_by_model: false` is not a comment. `RedFlagHit.suppressible_by_model` is
typed `Literal[False]` in Pydantic — a hit asserting otherwise cannot be constructed.

### Acuity is never a bare integer

```python
class Acuity(BaseModel):
    scheme_id: str        # ESI | MTS | CTAS | LOCAL3 | FREETEXT
    scheme_version: str
    level: int
```

Comparing two `Acuity` values from different schemes **raises** `SchemeMismatchError`
rather than silently coercing. Cross-scheme conversion must go through
`core/schemes.convert()`, which consults the versioned mapping table, declares its fidelity,
lists what is lost in translation, and **refuses** the edges registered UNSAFE. An ESI 2, an
MTS Orange and a local "Red" cannot be accidentally compared anywhere in this system,
because the type will not permit it.

---

## 4. Two layers, two questions, two labels

The single most important structural decision in V2, and the one that took the longest to
get right.

| | Arrival layer | Deterioration layer |
|---|---|---|
| Question | "How sick is this patient **now**?" | "Will this patient need escalation **during the wait**?" |
| Mechanism | Deterministic rules over age-band thresholds and a complaint resource floor | XGBoost over band-relative features |
| Label | `truth_arrival_acuity` | `truth_deteriorates` — nothing else |
| Output | An ESI level, 1–5 | A probability, binary decision at the cost threshold |
| Cold start | **Works on day one** | Needs data |

A composite label collapses these and hides which layer is working. They are never
recombined. The model's binary output is also never fanned into five ordered clinical
classes — one probability cannot honestly produce five, and `CostDecision.decision_asymmetric`
is deliberately typed as 1/0.

**Level 1 is reserved for red flags and arrival physiology.** `escalation_target()` floors
model-driven escalation at level 2. A probabilistic model never puts a patient in the
resuscitation bay on its own authority.

---

## 5. Age as physiology, not as a number

Feeding `age = 78` to a gradient-boosted tree and hoping it learns geriatric physiology is
the silent safety risk the problem statement names. V2 does not do this.

Vitals are converted to **band-relative z-scores** against the contract's normal range for
that patient's band before they reach the model:

```
HR 148  →  age 2:   z = +0.93   NORMAL      (normal band 90–150)
HR 148  →  age 75:  z = +3.40   CRITICAL    (normal band 60–100)
```

The interaction is supplied by clinicians through the contract, not learned from data that
would need to be enormous to contain it. Seven bands: `0–1`, `1–5`, `5–12`, `12–18`,
`18–65`, `65–80`, `80+`.

The geriatric bands carry an explicit blunting rule: **absence of fever or tachycardia does
not lower risk in the elderly.** This is encoded as a note on the threshold and as an
asymmetry in the scoring — a normal temperature in an 80-year-old with a concerning
respiratory rate does not offset it.

**The canonical demo:** P-001 (age 3) and P-002 (age 78) both present at 38.5 °C. Same
number, different bands, different arrival acuity, for reasons the UI shows.

---

## 6. Signal quality gating — reject, do not down-weight

A value below its contract quality floor is **excluded from the feature set entirely**. It
is not multiplied by a small weight, because a down-weighted value we do not believe can
still move a score.

`SignalQuality.passed_floor` is the load-bearing field. Note the deliberate distinction
between two different things that are usually conflated:

| | |
|---|---|
| **Display band** | GOOD / ACCEPTABLE / DEGRADED / FAILED — what the human is told about the signal |
| **Accept floor** | The per-method threshold below which the value is rejected outright |

These **overlap on purpose**. SNR 3.2 displays as DEGRADED and still clears the 3.0 floor
for rPPG heart rate, while failing the 4.0 floor for rPPG respiratory rate — because
respiratory rate from video in a crowded waiting room is a far weaker measurement than
heart rate, and the contract says so per method rather than per signal.

Failure modes modelled and exposed rather than hidden: motion, low light, occlusion,
skin-tone-related SNR loss, dialect and code-mixing, crowd noise, non-verbal patients,
absent or non-responsive attendants, unpaired devices, stale vitals, and conflicts between
the ED record and the linked health record.

**Degradation always moves the patient toward human attention, never away from it.**

---

## 7. Reconciling three cadences to one grain

| Channel | Native grain | Cadence |
|---|---|---|
| Nurse desk | encounter × timestamp | once at triage, then on re-check |
| Camera — arrival gestalt | encounter × arrival event | single pass, 5–10 s |
| Camera — waiting rPPG | patient × 30 s window | rolling, 30–60 s |
| Attendant | patient × prompt-response | event-driven + timed nudges |
| Prior record | patient × record version | on lookup, may be months stale |

All are reconciled to a common **patient × time** grain with explicit windowing. Out-of-order
arrivals are ordered by observation window end, not receipt time. Partial-channel operation
is a first-class mode: a channel that has produced nothing for N minutes is marked
`SILENT`, which is a different state from `ABSENT` (never present) and from `DEGRADED`
(present but below quality). These three are never collapsed into "no data".

### Trajectories

Rate of change matters more than level for the waiting interval. Trajectories use a
**Theil–Sen slope** over quality-passed evidence, computed per `(concept, acquisition_method)`
pair so a nurse count and an rPPG estimate are never regressed through the same line.

The escalation threshold is relaxed by the method's reliability:

```
effective_threshold = clinical_threshold / reliability ** 0.35
```

A less reliable channel must show a *larger* move to trigger the same action. Median-of-
endpoints was tried first and lags badly on a ramp; naive endpoint differencing fired
constantly on rPPG noise. Theil–Sen is robust to the outliers rPPG actually produces.

---

## 8. Fusion, not voting

Disagreement between channels is the signal, not noise to be averaged away.

Each channel produces a subscore. The **discordance term is signed and enters the score
explicitly**, so it appears in the decomposition the nurse sees:

- **Escalating discordance** — a calm-looking patient whose family reports new confusion
  *outranks* a restless patient whose family says they are at baseline. (P-005 vs P-006.)
- **De-escalating discordance** — `baseline_veto`. A family reporting the patient is at
  their baseline can block a *borderline model-threshold* escalation, but only when there
  is no material worsening trajectory, no red flag and no critical vital. **A family can
  prevent a marginal rise; they can never talk down a measured trend.**

### Reporter calibration, and why it is narrow

Attendant reliability is calibrated per reporter, but only under three constraints, each of
which exists to stop the calibration from defeating the channel:

1. **Only a nurse adjudicates.** The camera cannot see confusion; penalising a family for
   reporting something no sensor can observe would train the weight to zero on exactly the
   reports that matter most.
2. **Only change-claims count.** "He is more sleepy than before" is scoreable. "He is
   unwell" is not.
3. **Shrunk by `min(1, n/3)` and floored at 0.30.** No family is judged on one data point,
   and no family is ever silenced.

P-015 presses the escalation button seven times on a stable patient. Their weight decays to
the 0.30 floor, their patient's acuity does not move — and every press still buys a nurse
re-check. **There is no prize for lying, and no penalty for asking.**

### Accumulation

Three or more concurrent RECHECK-class findings escalate one level even when no single one
would. This is what catches P-014 at the moment the camera loses the respiratory-rate trend
to an SNR drop — the evidence gets thinner and the concern does not.

---

## 9. Confidence, decomposed

Never one number. Five named, inspectable axes, always displayed together:

| Axis | Measures | Floor |
|---|---|---|
| `signal_quality` | Per-channel SNR, ASR confidence, frame quality | 0.45 |
| `completeness` | Which features are actually present | 0.40 |
| `applicability` | Is this patient inside the training distribution | 0.50 |
| `channel_agreement` | Do the three channels concur | 0.30 |
| `calibration_status` | Is the model calibrated *for this subgroup*, and when | — |

```
signal quality ≠ model confidence ≠ clinical certainty ≠ escalation priority
```

`ConfidenceComponents.overall()` exists — a weakest-link `min()` — and is used **only** for
internal gating and ranking. It is never rendered to a clinical user without its parts.
There are no invented "87% confidence" numbers anywhere in this system.

---

## 10. Abstention

Five gates. Any one may trip, and tripping means the system declines to score.

| Gate | Trips when |
|---|---|
| `ABST_SIGNAL_FLOOR` | Signal quality below floor across the usable window |
| `ABST_COMPLETENESS` | Too few features present to run the route |
| `ABST_APPLICABILITY` | Patient outside the model's applicable distribution |
| `ABST_ASR_AND_NO_ATTENDANT` | Complaint unparsed **and** fewer than 2 live objective channels |
| `ABST_CAMERA_OCCLUSION_SUSTAINED` | Occlusion above threshold for consecutive windows |

The fourth gate carries a clause that took some thought: **abstention requires that we
cannot see the patient, not merely that we cannot hear their story.** A non-verbal patient
with a working camera and a nurse observation is assessable (P-016). A patient we can
neither hear nor see is not (P-009).

On abstention:

```
Patient held at current acuity — NOT downgraded
Queue position protected
Re-check timer forced to 0
Ranked HIGH — the EWER abstention term is weighted 2.20
A specific priority question is generated for the nurse
```

Enforced structurally: a `SafetyContract` in state `INSUFFICIENT_SIGNAL` **cannot be
constructed carrying a cost decision**. A Pydantic validator raises. There is no risk number
in the object for a renderer to leak.

---

## 11. Asymmetric cost

```
C(under-triage) = 12.0        C(over-triage) = 1.0        ratio 12:1
threshold = 1 / (1 + 12) = 0.0769
```

Configurable, versioned, auditable — `cost_policy.yaml` v1.4. Not a magic number in code.
The threshold is **derived from the cost ratio**, never tuned for accuracy, F1, or Youden's J.

The comparison, on held-out simulated data:

| | Symmetric (0.5) | Cost-derived (0.0769) |
|---|---|---|
| Sensitivity | **0.000** | 0.391 |
| Specificity | 0.999 | 0.850 |
| Under-triage n | 69 | 42 |
| Over-triage n | 1 | 230 |
| Policy cost per 100 | 51.81 | **45.88** |

The accuracy-optimised threshold escalates nobody. Under a rare-event distribution
(4.3% event rate) that is exactly what an accuracy objective *should* produce, and it is
clinically worthless. **24 named patients are caught by the cost threshold and missed by
the symmetric one.**

The two axes are kept separate and are never equated:

```
Statistical evidence WEAK   ·  Clinical materiality CRITICAL  →  ESCALATE
Statistical evidence STRONG ·  Clinical materiality LOW       →  LOG, DO NOT ALERT
```

A statistically detectable 2 bpm heart-rate drift is clinically irrelevant. A single
new-onset confusion report has almost no statistical power and enormous clinical
materiality. **The escalation engine can escalate on materiality alone**, and P-005 is the
case that exercises it.

---

## 12. EWER — Evidence-Weighted Escalation Rank

A transparent engineering ranking of who deserves human attention next.

It is **not** a probability, **not** a clinical certainty, **not** a diagnosis, and **not** a
severity level in any published scheme. The only renderer provided prints a bare index with
a disclaimer attached; formatting it as a percentage or a triage level is a category error
the module refuses to commit.

| Component | Weight |
|---|---|
| Model risk estimate | 1.00 |
| Confidence penalty (low confidence **raises** rank) | 0.45 |
| Channel discordance (net of de-escalating) | 0.55 |
| Time since human contact | 0.35 |
| Wait vs SLA for current level | 0.50 |
| Red flag fired | **6.00** — dominates by construction |
| Trajectory | 0.70 |
| Clinical materiality | 0.90 |
| Abstention | **2.20** — an unassessable patient ranks high, not low |

Acuity and EWER are different objects with different jobs. Acuity is a clinical
classification carrying a scheme and version. EWER is an ordering over a queue at an
instant. The queue sorts by acuity **first**, then by EWER within a level. They are never
collapsed.

---

## 13. The LLM boundary

A hard line, visible in the architecture and in the UI.

**Non-LLM — deterministic, testable, auditable:**
signal acquisition · quality gating · vital extraction · age-band evaluation · red-flag
rules · tabular risk model · temporal fusion · discordance computation · confidence
decomposition · cost thresholding · EWER ranking · access control · Safety Contract
generation · claim validation · audit logging · telemetry.

**LLM — bounded, replaceable, never load-bearing:**
multilingual ASR post-processing and code-mix normalisation · structured symptom extraction
into contract concepts · persona narrative rendering **from an already-decided Safety
Contract** · generating the clarification question · translating attendant prompts.

The LLM does not determine acuity, risk, confidence, the escalation decision, permissions,
evidence ranking, or whether a red flag fired.

**This is tested, not asserted.** The demo runs the same patient with the LLM enabled and
disabled and diffs the outputs: acuity, EWER, red flags and the abstention decision are
identical. Only the renderer changes, from `llm` to `deterministic_template`. Tier C runs
template-only by deployment profile, permanently.

---

## 14. The Safety Contract and the Claim Compiler

The central defensible innovation: **what may be said is determined by what the evidence
supports, before any language is generated.**

```
Evidence → gating → risk + confidence → cost decision
                            ↓
                    SAFETY CONTRACT          ← deterministic code writes this
                            ↓
                  narrative plan (per persona)
                            ↓
                      LLM rendering          ← the model's only job
                            ↓
        VALIDATOR — rejects any claim outside the contract
                            ↓
                         Output
```

The Safety Contract carries, among other fields:

```json
{
  "epistemic_status": "MODERATE_CONFIDENCE_ESCALATION",
  "allowed_claim_types":   ["observation", "association", "attribution", "recommendation_recheck"],
  "forbidden_claim_types": ["diagnosis", "causation", "prognosis", "treatment_recommendation", "reassurance"],
  "allowed_numbers":  [27, 7, 22, 34, 2, 3],
  "allowed_entities": ["P-014", "respiratory rate", "movement", "attendant report", "wait time"],
  "required_disclosures": ["prior_record_missing", "rppg_hr_degraded"],
  "max_words": 90
}
```

A number not in `allowed_numbers` cannot appear in the output. The validator rejects it.

### Claim grammar is status-aware

| State | Allowed | Forbidden |
|---|---|---|
| `OBSERVATION_ONLY` | "Respiratory rate increased from 20 to 27 over 22 minutes." | "The patient is deteriorating." |
| `ASSOCIATION` | "This pattern was associated with escalation in similar presentations." | "This pattern means the patient will deteriorate." |
| `ATTRIBUTION` | "Respiratory rate contributed most to the score change." | "Respiratory rate caused the risk increase." |
| `RED_FLAG_FIRED` | Direct imperative — "Stridor reported. Immediate clinician review." | — |
| `INSUFFICIENT_SIGNAL` | "Cannot assess reliably. Nurse re-check required." | Any risk statement, any number, any reassurance |

**Universally forbidden in every state:** any diagnosis, any treatment recommendation, any
prognosis, any reassurance to a patient or family, any statement that a patient is safe to
wait.

The distinction between **attribution**, **association** and **causation** is enforced at the
grammar level. A SHAP contribution licenses attribution language and never causal language.
The system writes *"14 minutes without movement contributed +0.18 to the risk score"* and
cannot write *"stillness caused the deterioration."*

The validator runs 8 checks at CRITICAL or STANDARD severity, rejects and regenerates on
violation, and logs every rejection. **It fires in the demo** — an injected draft produces 8
caught violations and a fall-back to the deterministic renderer.

---

## 15. Lineage

Every displayed acuity traces back to the raw observation, in one click:

```
DISPLAYED ACUITY
  ↓ decision rule / cost threshold applied
  ↓ fusion output + confidence components
  ↓ per-channel sub-scores
  ↓ feature values used, with staleness
  ↓ signal quality gate result per feature
  ↓ raw observation — timestamp, channel, acquisition method, device
  ↓ model versions + contract version
```

Stored in SQLite, retrievable by `lineage_ref`, exposed at
`GET /patient/{id}/lineage` and as a drill-down in the nurse view.

### The evidence object

Every insight traces to these. There is no way to construct one carrying a number without
carrying the epistemics of that number:

```json
{
  "evidence_id": "EV-000871", "patient_id": "P-014",
  "source_channel": "camera", "acquisition_method": "rppg_rr", "device_id": "WAIT-CAM-02",
  "concept_id": "RESP_RATE", "value": 27, "unit": "breaths/min",
  "observation_window": ["11:29:30", "11:30:00"], "grain": "patient x 30s_window",
  "signal_quality": { "snr_db": 4.2, "occlusion_pct": 12, "status": "GOOD", "passed_floor": true },
  "freshness": { "age_seconds": 45, "status": "CURRENT", "max_staleness_minutes": 15 },
  "reliability_weight": 0.6,
  "contribution": { "method": "shap", "value": 0.18, "direction": "escalating" },
  "supports_or_contradicts": "supports",
  "age_band_context": "age_18_65 threshold: concerning 21-24, critical >24",
  "model_version": "rppg_v2.1", "contract_version": "2.0.1", "lineage_ref": "LIN-000871"
}
```

### Contradictory evidence is retrieved, not suppressed

The nurse card has a **"what argues against this"** region, populated on every patient. The
system retrieves evidence that contradicts its own leading assessment and displays it beside
the evidence that supports it. A tool that only shows you why it is right is a tool that
teaches you to stop checking.

---

## 16. Four personas, four projections

Not the same payload with different labels. Genuinely different information depth, evidence
selection, available actions, and decision framing.

| | Triage nurse | ED physician | Attendant | Administrator |
|---|---|---|---|---|
| Patient clinical detail | Full, own queue | Full, own floor | **None** | **None** |
| Acuity score | Yes | Yes | **No** | Aggregate only |
| Raw complaint text | Yes | Yes | Own patient only | No |
| Attendant free text | Yes | Yes | Own only | No |
| **Raw video / frames** | **No — nobody** | **No — nobody** | **No** | **No** |
| Derived camera features | Yes | Yes | No | Aggregate only |
| Other patients' data | No | Own floor | No | Aggregate only |
| Override history | Own | All | No | Aggregate + identity |
| Subgroup performance | No | Own floor | No | Yes |

**The UI renders from the backend projection.** Every persona view in `saathi/ui/app.py` is
built from the dict returned by `core.rbac.project_assessment()` — the same function the
HTTP API uses. The attendant view cannot display an acuity because the projection it
receives does not contain one. Deleting the persona branch in the UI would change nothing
about what an attendant can see.

The attendant surface is a **separate allow-listed endpoint**, not a filtered clinical
payload, because building a restricted view by subtraction from a permissive one is how
fields leak.

### What each persona actually needs

**Nurse** — a decision in under five seconds of reading. Rank-ordered queue with deltas
highlighted; the two or three specific observations that drove the score; what argues
against; one-tap accept or one-tap override with a reason; and an explicit *"go and check
this"* instruction. If it needs scrolling to decide, it has failed.

**Physician / charge nurse** — the whole queue, not one patient. Who is past the safe wait
for their level. Where the queue breaks under current staffing, stated as a number. Cross-
patient patterns. Override rate and disagreement for the shift. Surge posture and what the
system changed about its own behaviour.

**Attendant** — one question at a time, in their language, voice-first, pictorial fallback.
Observable tasks only: *count his breaths for fifteen seconds*. Confirmation that the report
was received and by whom. **No score, no acuity, no queue position, ever.** A visible,
non-gameable escalation path.

**Administrator** — aggregate under- and over-triage, override rate trended as the headline
trust metric, subgroup performance by age band / sex / skin-tone band / language, cost and
infrastructure telemetry. Cells below `minimum_aggregation_cell_size` are suppressed to
prevent re-identification by small-cell inference.

---

## 17. Security and PHI minimisation

Three enforcement points, in this order:

1. **Before retrieval** — `scope_patients()` narrows the query itself. Rows the caller may
   not see are never read out of the store.
2. **Before analysis** — `project()` strips fields the role may not hold before anything
   computes on them.
3. **Before any LLM call** — `llm_payload()` builds the prompt from an already-minimised
   projection. We never hand a model data the caller may not see and trust it to withhold.

Every access decision writes `role, subject, resource, fields, decision, reason, timestamp`.
The 403 path is exercised by unauthorised HTTP requests in `tests/test_rbac.py`, which
asserts both the status code **and** the audit row.

### Non-negotiable data rules

- **Raw video frames are processed in memory and discarded.** No frame is written to disk or
  leaves the device. There is no function in this system that returns one, to any role,
  including operators — *the absence is the control, not a filter*.
- Face embeddings for waiting-room re-identification are session-scoped, encrypted, and
  destroyed at disposition. **No cross-visit biometric linkage.**
- Attendant phone numbers are hashed for session linkage.
- The LLM payload is displayed verbatim in the demo. `assert_no_pii()` raises rather than
  permitting egress, and the validator independently rejects identifier-shaped tokens in the
  output.
- **Consent at registration**, in the attendant's language, with a functional opt-out that
  does not degrade queue position. Opting out disables the camera and attendant channels and
  **shortens** the nurse re-check interval to compensate. (P-018.)

### Untrusted input

The spoken complaint and attendant free text are **data, never instruction**. Three layers:
sanitisation, a closed clinician-authored symptom vocabulary matched by exact token, and the
structural fact that free text has no wire into the feature vector, the cost threshold, or
the red-flag engine. The only thing a complaint can do is name a symptom a clinician already
wrote a rule for.

---

## 18. Safety invariants

Implemented as assertions and as a passing test suite — 33 invariant tests within 93 total.

1. **Monotonic escalation** — SAATHI raises urgency, never lowers it. Enforced in
   `SafetyContract`'s Pydantic validator: a contract moving a patient to a less urgent level
   without a human override **cannot be constructed**.
2. **Red-flag supremacy** — deterministic rules fire regardless of model output.
   `red_flags.py` imports no model; the stage runs before the model stage.
3. **Degraded-mode escalation** — model, signal or LLM failure moves the patient toward
   human attention, never away.
4. **Fail-safe default** — total stack failure falls back to FIFO-plus-red-flags and says so
   loudly on screen.
5. **No silent imputation** — a missing value is never replaced by a mean and treated as
   observed. Missing is `NaN` and is reported as missing.
6. **Attendant non-gaming** — escalation buys a nurse re-check, never a queue position.
7. **Wait-time SLA** — L1 immediate, L2 10 min, L3 30 min, L4 60 min, L5 120 min. Breach
   forces re-assessment.
8. **Human authority** — no patient is moved, discharged or de-prioritised without a human.

---

## 19. Surge

The system behaves **differently** at 3× volume, not merely slower. Detection is automatic
and the transition is announced on screen.

| | Normal | Surge (3×) |
|---|---|---|
| Re-check interval, L3 | 30 min | 15 min |
| Alert budget | 8/hour, per patient | 12/hour, batched top-N |
| Camera coverage | full | degraded — occlusion up, confidence down |
| Attendant prompt frequency | 20 min | 12 min |
| Abstention rate | baseline | **rises, and this is correct** |
| LLM narration | full | disabled for non-critical, to protect latency |

Under surge, degraded signal quality produces **more** abstentions and **more** nurse
re-checks. The system does not compensate for worse information by becoming more confident.

**Red flags, abstentions and L1/L2 SLA breaches are exempt from the alert budget** and always
surface immediately. Only the remainder competes for the ceiling; when it is exceeded, alerts
are re-ranked by EWER and the excess is deferred to the batched view — deferred and marked,
never discarded.

**The honest failure point is stated, not hidden.** At some volume the floor cannot absorb the
escalations. The physician view names that number for the current queue and says plainly that
the shortfall is a staffing fact this system cannot solve. Pretending otherwise is how
alerting tools get switched off in week two.

---

## 20. Deployment tiers — configuration as data

The same core assistant, the same contract, the same red-flag rules. What changes is which
channels exist and how confidence is computed. A **deployment profile**, not a code fork.

| | **Tier A** | **Tier B** | **Tier C** |
|---|---|---|---|
| Volume | 500+/day urban trauma centre | 200–300/day district | ~100/day rural |
| Cameras | Door + waiting area | Door only | **None** |
| Records | HMIS + ABDM | Partial | **No EHR** |
| Network | Reliable | Intermittent | Offline-first, sync on reconnect |
| Devices | Paired cuff/SpO₂, edge mini-PC | Shared tablet | One phone |
| Attendant | App | App | **SMS / IVR** |
| LLM | On | On | **Off — template only** |

**Guaranteed in every tier, however degraded:** deterministic red-flag layer · monotonic
escalation · wait-time SLA · override capture · audit trail · abstention · attendant channel
· consent and opt-out · lineage.

Tier C is the honest test of the design. With no camera, no EHR and no model confidence to
speak of, it still gets the clinician-authored rule layer and a family channel over SMS — and
that is already more than the waiting room has today.

---

## 21. Telemetry and cost

Every number **measured** at runtime with `perf_counter`, written per stage per patient.

```
Decision path p50 / p95 / p99      50.5 / 62.8 / 76.8 ms      budget 2000 ms - MET
End-to-end (with narration)        60.1 / 102.3 / 123.1 ms
Cost per 1,000 ED visits           $1.38   (stub backend, priced on measured token counts)
```

`decision_p95_ms` **excludes** the LLM render and validate stages and reports them
separately. A system whose triage decision waits on a network call to a language model has
put that model on the safety path, which is exactly what this architecture forbids.

---

## 22. Overrides — the trust loop

Overrides are the primary adoption signal and the primary safety signal. First-class
feature, not an exception path.

Captured per override: `override_id, patient_id, clinician_id, role, timestamp,
system_acuity, clinician_acuity, direction, reason_code, free_text,
evidence_shown_at_time` (the **full** snapshot as displayed), `safety_contract_id,
model_versions, contract_version, time_from_display_to_override_ms, outcome_link`.

Seven years from now, *"what did the clinician actually see?"* still has an answer.

The loop closes: override rate is the headline trust metric on the admin view, trended.
Reasons are clustered to find systematic model failures. Disagreement feeds recalibration
**only through a documented human review gate** — the system never auto-retrains on
overrides. And the nurse is shown that their override was recorded and what happened to it,
because staff work around tools that ignore them.

---

## 23. What is designed but not built

Stated plainly, because a demo that implies something works when it does not is worse than
a smaller demo.

| Designed, specified, not built | Why |
|---|---|
| Live video capture and a trained rPPG model | Out of scope; values are replayed from a scripted signal-quality trace |
| Real ASR | Complaints are pre-transcribed with realistic confidence scores |
| Hospital HMIS / ABDM integration | Requires a hospital |
| Prospective clinical validation | Requires patients, ethics approval, and BODH |
| Face-track re-identification | The failure path is designed and tested; the CV is not built |

Everything else in this document is running code, exercised by the test suite, and visible
in the demo.
