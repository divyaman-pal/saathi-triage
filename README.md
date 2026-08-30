# SAATHI

**Continuous, multi-channel triage assistance for the waiting interval.**
Accenture Innovation Challenge 2026 · Round 2 · Track 2 — PatientTriage.ai · Team Mavericks

> Clinical decision support. It raises urgency and can never lower it. It does not diagnose.

---

## The claim

Triage is treated everywhere as an **event**. Deterioration is a **process**.

A patient is assessed once, in two minutes, at a desk — and then nobody looks again until
a clinician is free. The waiting interval is where under-triage turns into harm, and it is
the one part of the pathway no system computes over. SAATHI makes the waiting interval the
object of computation rather than the arrival snapshot.

It does this using three signals that are already present in every Indian emergency
department and are currently thrown away:

| Channel | What it contributes | Cadence |
|---|---|---|
| **Nurse** (at the desk) | Spoken complaint, vitals, AVPU — the anchor | Once at triage, then on re-check |
| **Camera** (door + waiting area) | Arrival gestalt, rPPG heart/respiratory rate, stillness | ~5–10 s at the door, then rolling 30–60 s windows |
| **Attendant** (family, on their own phone) | New confusion, sleepiness, pain trajectory — the only channel that catches these | Event-driven plus timed nudges |

Zero per-patient hardware. Existing cameras, the family's own phone.

---

## What is in this repository

A **working prototype**, not a slide deck with a mock behind it.

```
93/93 tests pass          33 safety invariants · 35 adversarial · 25 RBAC
62-patient cohort         20 hand-designed cases + 42 procedural surge arrivals
62.8 ms                   decision-path p95 latency, against a 2000 ms budget
```

Run it:

```bash
pip install -r requirements.txt
python -m saathi.data.generate      # train the models  (~40 s)
streamlit run saathi/ui/app.py      # the four persona surfaces
```

Also available:

```bash
uvicorn saathi.api.main:app --reload   # the HTTP API the UI is built on
python -m saathi.eval.evaluate         # the full evaluation report
python -m pytest saathi/tests/ -q      # the safety invariant suite  (~2 min)
```

---

## Documents

| | |
|---|---|
| [Architecture](docs/SAATHI_V2_ARCHITECTURE.md) | The V2 design in full — layers, contract, fusion, claim compiler, tiers |
| [Clinical & statistical audit](docs/CLINICAL_AUDIT.md) | Where this is weak, stated precisely. Read this one first if you are sceptical |
| [Safety red team](docs/SAFETY_RED_TEAM.md) | Attacks attempted against the system, and what survived |
| [V1 → V2 analysis](docs/DEEP_ANALYSIS_V1.md) | What Round 1 got wrong and what changed |
| [Adoption](docs/ADOPTION.md) | Why a fatigued nurse at 2 a.m. uses this instead of working around it |
| [Regulatory](docs/REGULATORY.md) | DPDP 2023, ABDM/ABHA, SAHI, BODH, CDSCO positioning |
| [Build state](HANDOFF.md) | Current status, architectural decisions, open items |

---

## The five things worth looking at

**1. It can say "I don't know."**
Patient P-009 — waiting room at 3× occupancy, camera 62% occluded, rPPG SNR below floor,
ASR confidence 0.34 on a dialect outside coverage, no attendant, no prior record, nurse
vitals 41 minutes old. The system declines to score, holds the patient at their current
acuity, protects their queue position, and demands a nurse re-check with a specific
question. It ranks **high** in the queue, not low. A patient the system cannot see is a
patient a human must see.

**2. The same 38.5 °C means two different things.**
P-001 is 3 years old; P-002 is 78. Every threshold is a function of age band and lives in
`saathi/contract/clinical_contract.yaml`. There is no vital-sign threshold anywhere in the
Python source — the scoring engine *consumes* the contract, it does not document it. The
geriatric bands additionally encode blunted response: the absence of fever or tachycardia
does not lower risk in the elderly.

**3. Turning the language model off changes nothing that matters.**
Judge surface → *LLM boundary* → **Run both ways and compare**. Acuity, EWER rank, red
flags and the abstention decision are identical with the LLM disabled. Only the prose
changes. The LLM renders an already-decided Safety Contract; it never authors one.

**4. The validator fires.**
Judge surface → *Validator* → **Inject a forbidden claim**. Eight violations caught on a
single injected draft, logged, and the system falls back to the deterministic renderer
rather than shipping the text. A validator that has never fired is a validator nobody
believes.

**5. The symmetric threshold is the persuasive artefact.**
At a cost ratio of 12:1, the decision threshold is 0.0769, not 0.5. On held-out data the
accuracy-optimised 0.5 threshold escalates **nobody** — sensitivity 0.000 — which is
"accurate" under a rare-event distribution and clinically worthless. The cost-derived
threshold catches 24 patients it misses.

---

## What is real and what is simulated

Judges forgive scope limits. They do not forgive a demo that implies something works when
it does not.

**Real, and load-bearing:**
Clinical Semantic Contract as loaded YAML driving actual scoring · age-band threshold
engine · deterministic red-flag rule engine (15 rules + 4 time rules) · XGBoost tabular
model with per-age-group isotonic calibration · temporal fusion with Theil–Sen trajectory
estimation · five-component confidence decomposition · cost-threshold decisioning ·
Safety Contract generation · claim validator · backend-enforced RBAC returning real 403s ·
full audit and lineage store · measured latency, token and cost telemetry.

**Simulated, and labelled as such everywhere it appears:**
rPPG values and their signal-quality traces (a replayed trace, not live computer vision) ·
ASR output and confidence scores (pre-transcribed) · camera occlusion events · prior
records · the entire patient cohort.

**Deliberately out of scope:**
Live video capture · real patient data of any kind · hospital system integration · a
trained rPPG model · prospective clinical validation.

**Every number in this repository comes from simulated data with a stated generative
process. None of it is evidence of clinical performance.** The transfer problem from
MIMIC-IV-ED and NHAMCS to an Indian district ED is real and unsolved. The sanctioned path
to Indian validation is federated benchmarking through BODH/ABDM — SAATHI can be
benchmarked on Indian data without ever holding it.

---

## Team

**Mavericks** — IIT Kanpur
Peeyush Agarwal (Aerospace, 2027) · Divyaman Pal (Civil, 2027) · Adiba Khan (Chemistry, 2027)
