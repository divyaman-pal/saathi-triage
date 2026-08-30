# SAATHI V2 — Adoption and Change Management

> The question a judge should ask: *why does a fatigued nurse at 2 a.m. use this instead of
> working around it?*

Most clinical AI does not fail because the model was weak. It fails because it added a step,
alarmed too often, ignored the people using it, or left nobody able to answer who owns a
miss. This document answers each of those concretely.

---

## 1. Zero added workload — the hard constraint

**The nurse's workflow gains no new mandatory step.** Not one field, not one tap, not one
screen to dismiss.

Everything SAATHI consumes is already happening:

| Input | Where it already comes from |
|---|---|
| Spoken chief complaint | The nurse is already asking. A microphone at the desk listens |
| Vitals | Already taken, on devices the ED already owns |
| AVPU, age, sex | Already recorded at registration |
| Arrival gestalt | The camera at the door already sees it. Nobody records it today |
| Waiting-room vitals | The waiting-area camera already sees the patient |
| Attendant observations | The family is already watching, for hours, and is currently ignored |

If SAATHI ever requires a nurse to fill in a field to work, it dies. That is a design
constraint, not an aspiration — and it is why the attendant channel exists. The only way to
get continuous observation into an ED at 4:1 attendant-to-patient ratios without hiring
anyone is to give the people already in the room a defined job.

---

## 2. The first 30 days: shadow mode

**SAATHI scores and shows nothing.**

| Week | What happens |
|---|---|
| 1–4 | Silent operation. Every assessment is computed, contracted and logged. No screen, no alert, no influence on any decision |
| End of week 4 | The comparison is published **to the staff who were being compared against**, before go-live |
| Week 5 | Go-live only if the floor agrees to it |

What gets published, honestly, including the parts that make the system look bad:

- Where SAATHI agreed with the nurse — and how often the nurse was already right
- Where SAATHI escalated and the patient turned out fine (the over-triage cost, in beds)
- Where SAATHI escalated and the patient deteriorated (the catch)
- **Where SAATHI missed something a nurse caught** — published first and most prominently
- Alert volume per shift, so staff can judge the noise before living with it

Shadow mode also produces the only honest measurement of **automation bias** available: nurse
decisions in weeks 1–4 are uncontaminated by a displayed recommendation, and become the
baseline against which post-go-live decisions are compared. If nurses start agreeing with
SAATHI *more* over time without outcomes improving, that is anchoring, and it is a reason to
pull the display — not to celebrate adoption.

---

## 3. The alert budget

An escalation rate the floor cannot absorb gets ignored within a week. So it is capped.

```
Normal   8 alerts/hour, presented per patient
Surge   12 alerts/hour, presented as a batched top-N
```

**Exempt from the budget and always surfaced immediately:** red flags, abstentions, and SLA
breaches at levels 1–2.

When the budget is exhausted, remaining alerts are **re-ranked by EWER and deferred to the
batched queue view**. A deferred alert is still visible to the charge nurse with a "deferred
by budget" marker. Nothing is discarded.

The measured escalation rate at the current operating point is **0.161**, i.e. roughly 16 of
every 100 arrivals. At Tier A's 21 arrivals/hour that is ~3.4 escalations per hour - inside
the budget of 8. **At a higher volume or a looser threshold it would not be**, and
the physician view states the shortfall as a number rather than absorbing it silently.

---

## 4. Override: one tap, never punished, always acknowledged

Overrides are the primary adoption signal and the primary safety signal. They are a
first-class feature, not an exception path.

**One tap.** Pick a level, pick a reason from a structured list, done. Free text optional.

**Never punished.** No dashboard ranks nurses by override rate. No individual is shown a
"disagreement score". Override reasons are clustered to find *systematic model failures* —
the unit of analysis is the model, not the nurse.

**Always acknowledged.** The nurse is told, on screen and in writing:

> Your override has been recorded with the full evidence snapshot, the Safety Contract id and
> every model version. It will appear in this shift's override report and is fed to
> recalibration only through a documented human review gate — the system never auto-retrains
> on your disagreement.

**Never auto-retrained on.** Overrides feed recalibration through a human review gate.
Auto-retraining on overrides creates a loop where the model learns to predict what nurses
already believe, which destroys the only independent signal in the system.

**Visibly acted on.** Override rate is trended on the admin view and published to staff per
shift. Staff work around tools that ignore them; the fastest way to make SAATHI ignorable is
to collect disagreement and never show anyone what came of it.

---

## 5. Trust metrics, published to staff

Per shift, on a board the floor can see, not in a management report:

| Metric | Why it is there |
|---|---|
| Override rate, trended | Falling = growing trust, or growing complacency. Both need watching |
| Catch rate | Escalations that preceded a real deterioration |
| False-alert rate | Escalations where nothing happened — the cost staff actually feel |
| Abstention rate | How often the system admitted it could not see |
| Alerts per hour vs budget | Whether the tool is inside its noise ceiling |

Publishing the false-alert rate to the people who bear it is the point. A system that reports
only its catches is asking to be distrusted by anyone who has worked a floor.

---

## 6. Attendant onboarding

At registration. In their language. Under 60 seconds.

1. *"You are with the patient. We will occasionally ask you simple things you can see — count
   his breaths, does he answer his name. Your answers go to the nurse."*
2. *"If you are worried, press this. A nurse will come and check."*
3. *"This does not change your place in the queue. The nurses decide that."*
4. *"You can say no. Nothing changes for your patient if you do."*

**Opting out costs nothing.** Consent is captured with a functional opt-out that does not
degrade queue position. Opting out disables the camera and attendant channels for that
patient and **shortens** the nurse re-check interval to compensate for the lost observation.
The patient loses SAATHI's continuous monitoring and gains more frequent human contact.
That is the correct trade and it is enforced in code — `test_invariant_7_opting_out_shortens_
the_interval_and_costs_no_queue_position`.

Voice-first with pictorial fallback, for low-literacy attendants. SMS/IVR for feature phones,
which is the Tier C default.

---

## 7. Who owns a miss

**Ambiguity here is what actually kills clinical AI deployments.** So it is stated flatly.

> **The clinician decides. SAATHI records what it showed, and when.**

| | Responsibility |
|---|---|
| **Clinician** | Every triage, escalation, downgrade and disposition decision. SAATHI has no authority to move, discharge or de-prioritise any patient |
| **SAATHI** | Producing an auditable record of what was displayed, on what evidence, at what time, with which model and contract versions |
| **Hospital** | Staffing to the escalation volume it accepts, and acting on the published trust metrics |
| **Vendor / developer** | The contract, the rules, the models, the calibration gate, and disclosing subgroup performance honestly |

SAATHI is positioned as **clinical decision support**, not a diagnostic device. It raises
urgency and can never lower it. Every output is advisory and every action requires a human.

The audit trail is built for exactly this question. Seven years from now, *"what did the
clinician actually see when they made that call?"* has a complete answer: the full evidence
snapshot as displayed, the Safety Contract id, the model versions, the contract version, and
the milliseconds between display and decision.

---

## 8. The violence angle — a co-primary outcome

**4:1 attendant-to-patient ratio. Families feeling ignored is the top trigger of ED
violence.**

This is not a footnote to the clinical case; it is a co-primary outcome and should be
presented as one.

The mechanism is simple and it is the same mechanism that produces the clinical benefit:

```
family is given a defined, real clinical job
        ↓
their reports are visibly received and acknowledged, by name
        ↓
they can see that something is being done
        ↓
the single largest trigger of ED violence is addressed
```

The attendant channel is not a data-collection trick with a safety story attached. It is a
communication intervention that happens to produce clinically valuable observations. For a
hospital administrator deciding whether to fund this, staff-safety improvement is a harder
number than triage sensitivity, and it arrives sooner.

**The dual fix:** giving families a continuous clinical job addresses under-triage safety and
staff violence with one intervention.

---

## 9. What would make us stop

Pre-committed stopping conditions, so the decision is not made in the moment by whoever has
the most invested:

| Condition | Action |
|---|---|
| Under-triage rate in any age band exceeds the nurse-alone baseline | **Halt.** The system is worse than nothing for that group |
| Calibration bias in any subgroup turns negative and persists | **Halt for that subgroup.** Under-prediction in the elderly is the failure this exists to prevent |
| Override rate rises above 40% and stays there | **Halt the display.** The floor does not believe it, and an ignored recommendation is an anchoring risk with no upside |
| Alert volume exceeds the budget for two consecutive weeks | **Raise the threshold or halt.** An ignored alarm is worse than no alarm |
| A red flag is found not to have fired when it should have | **Halt everything** pending root cause. The deterministic layer is the floor of the whole design |
| Automation bias detected — nurse agreement rises without outcome improvement | **Revert to shadow mode** and re-evaluate the display |

---

## 10. Honest limits of this plan

Everything above is a **protocol, not a result**. No part of it has been executed, because
executing it requires a hospital, an ethics approval, and patients.

What can be said today is narrower and should be stated as such: the system has been built so
that these things are *measurable* — shadow mode is a deployment profile, the alert budget is
a contract parameter, override capture is a first-class store, trust metrics are computed
from the audit log rather than reconstructed. The instrumentation for this plan exists. The
evidence does not, and will not until [prospective validation](CLINICAL_AUDIT.md#the-transfer-problem)
through BODH/ABDM.
