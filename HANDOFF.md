# SAATHI V2 — BUILD STATE / CONTEXT HANDOFF

**Working dir:** `c:\Users\divya\Downloads\accenture_divyaman`
**Python:** 3.11.7 — see `requirements.txt` for pinned versions (all installed).

---

## 1. WHAT THIS IS

Accenture Innovation Challenge 2026, Round 2, Track 2 (PatientTriage.ai).
Project **SAATHI** — continuous multi-channel triage assistance for the **waiting interval**.

Central claim: *triage is an event; deterioration is a process.* V2 makes the waiting interval
the object of computation, not the arrival snapshot.

Three channels: **nurse** (desk), **camera** (door gestalt + waiting-area rPPG), **attendant**
(family's own phone).

---

## 2. STATUS — COMPLETE AND WORKING

```
93/93 tests pass       33 invariants + 35 adversarial + 25 RBAC
62-patient cohort      20 designed cases + 42 surge fill, assessed end-to-end
4 persona UIs          nurse / physician / attendant / admin + judge surface
7 documents            architecture, clinical audit, red team, V1→V2, adoption, regulatory, README
Reproducible           bit-identical across processes (was not, until fixed — see §5)
```

### Run commands

```bash
pip install -r requirements.txt
python -m saathi.data.generate            # rebuild models (~40s)
streamlit run saathi/ui/app.py            # THE DEMO — 4 personas + judge surface
uvicorn saathi.api.main:app --reload      # HTTP API
python -m saathi.eval.evaluate            # full evaluation report (~3 min)
python -m pytest saathi/tests/ -q         # 93 tests (~2.5 min)
```

---

## 3. FILE MAP

```
README.md                        Entry point — the five things worth looking at
requirements.txt                 Pinned deps
docs/
  SAATHI_V2_ARCHITECTURE.md      Main deliverable — 23 sections
  CLINICAL_AUDIT.md              The honest one. Weaknesses table in §12
  SAFETY_RED_TEAM.md             ~50 attacks mapped to tests; §10 = what did NOT hold
  DEEP_ANALYSIS_V1.md            V1→V2: 3 claims rethought, 10 gaps filled
  ADOPTION.md                    Shadow mode, alert budget, who owns a miss, stopping conditions
  REGULATORY.md                  DPDP 2023, ABDM/ABHA, SAHI, BODH, CDSCO

saathi/contract/                 LOADED AT RUNTIME — drives real behaviour, not documentation
  clinical_contract.yaml         25 concepts, 7 age bands, thresholds, quality floors, gates
  red_flags.yaml                 15 rules + 4 time rules, un-suppressible
  cost_policy.yaml               12:1 cost, SLAs, alert budget, surge posture, materiality matrix
  acuity_schemes.yaml            ESI/MTS/CTAS/LOCAL3/FREETEXT + 5 lossy mappings, UNSAFE edges
  entitlements.yaml              RBAC, field classes, retention
  claim_grammar.yaml             Epistemic states, forbidden lexicon, 8 validator checks
  deployment_profiles.yaml       TIER_A / TIER_B / TIER_C as DATA

saathi/core/                     20 modules — see architecture doc §2 for the stage order
saathi/data/                     cohort.py (20 designed cases), simulate.py (THE BOUNDARY), generate.py
saathi/api/main.py               FastAPI, backend-enforced RBAC, real 403s
saathi/ui/app.py                 Streamlit, 4 personas + 11-tab judge surface
saathi/eval/evaluate.py          9-section evaluation
saathi/tests/                    test_invariants.py, test_adversarial.py, test_rbac.py
artifacts/                       *.ubj models, evaluation.json, evaluation_report.txt,
                                 test_report.txt, saathi_audit.db
```

---

## 4. KEY ARCHITECTURAL DECISIONS (hard-won — do not undo)

1. **Two layers, two questions, two labels.** Arrival acuity = deterministic rules
   (`triage_rules.py`). Deterioration = XGBoost on `truth_deteriorates` ONLY. A composite
   label hides which layer works. Never recombine.
2. **Age enters as physiology, not a number.** Band-relative z-scores against contract normal
   ranges. HR 148 → z=+0.93 at age 2, +3.40 at age 75.
3. **Cost decision is BINARY**, not a 5-level fan-out. One probability cannot honestly
   produce five ordered clinical classes.
4. **Level 1 reserved for red flags + arrival physiology.** `escalation_target()` floors at 2.
5. **Quality DISPLAY band ≠ accept/reject FLOOR.** Deliberately overlapping. SNR 3.2 displays
   DEGRADED, clears the 3.0 rPPG-HR floor, fails the 4.0 rPPG-RR floor.
6. **Theil–Sen trajectories** per (concept, acquisition_method), threshold relaxed by
   `reliability**0.35`. Median-of-endpoints lags on a ramp; endpoint diffs fire on rPPG noise.
7. **Point-value materiality only for concepts declaring `materiality_escalation`** — else
   every concerning vital double-counts.
8. **Reporter calibration: only a NURSE adjudicates, only CHANGE-claims count, shrunk by
   min(1,n/3), floored 0.30.** Otherwise the channel dies on exactly the reports that matter.
9. **`baseline_veto`** blocks a *model-threshold* escalation only when there is no material
   worsening trajectory, no red flag, no critical vital.
10. **`accumulation_escalates()`** — 3+ concurrent RECHECK findings escalate one level.
    Catches P-014 when the camera loses the RR trend to an SNR drop.
11. **Abstention requires we cannot SEE the patient, not merely can't hear their story.**
    `ABST_ASR_AND_NO_ATTENDANT` has a `live_objective_channels < 2` clause. P-016 vs P-009.
12. **Deployment profile gates the LLM.** Tier C is template-only.
13. **Deterioration has a PRODROME** (15 min at 0.45 gain). Without it there is no observable
    pre-onset signal and no honest system could predict anything.
14. **The UI renders from `rbac.project_assessment()`,** never from the Assessment object.
    The attendant view cannot show an acuity because the projection has none.

---

## 5. THREE BUGS FOUND AND FIXED THIS SESSION

**5.1 — Red-flag latch cleared globally on replay.** `Runtime.replay()` rebuilt the whole
`RedFlagLatch`, which is keyed by patient — so replaying one patient un-latched every other
patient's fired flags. Would have surfaced the first time a judge clicked "replay" live.
Fixed to pop only that patient's entries. Verified: P-011 keeps its stridor flag through a
P-010 replay.

**5.2 — Evaluation not reproducible across processes.** `simulate.py` seeded rPPG jitter with
builtin `hash()` on a string, which Python randomises per process. Sensitivity moved ±0.03
and AUC ±0.02 between runs, and the *deterministic* arrival layer moved too. Replaced with
`blake2b`. Now bit-identical across processes. Documented in `CLINICAL_AUDIT.md` §0.

**5.3 — Alert budget treated as a scalar.** It is a dict, and the contract **exempts** red
flags, abstentions and L1/L2 SLA breaches from it. The physician view was counting exempt
alerts against the ceiling and overstating the shortfall. Fixed to compute budget-eligible
alerts per the contract's own exemption rule.

---

## 6. CANONICAL NUMBERS (reproducible; `artifacts/evaluation_report.txt`)

```
ARRIVAL RULES (cold start, no learned parameters)
  exact 0.456 [0.431,0.480] | under L1/L2→L3+ 0.062 [0.051,0.075] | over 0.469
  under-triage by band: age_5_12 0.000 · age_18_65 0.025 · age_65_80 0.181 ← WORST

MODEL (deterioration within 30 min, event rate 0.043)
  AUC 0.709 (rules-only 0.542) — reported once, then set aside
  at threshold 0.0769: sensitivity 0.391 [0.285,0.509], escalation rate 0.161
  calibration bias: paediatric +0.0027 · adult −0.0087 (unsafe dir) · geriatric +0.0141

SYMMETRIC vs ASYMMETRIC
  0.5 → sensitivity 0.000 (correct for a rare event, and inert)
  cost-derived → 0.391, cost/100 45.88 vs 51.81, 24 named patients rescued

BASELINES
  model 0.391 | rules-only at matched load 0.232 (+0.159) | nurse-alone 0.493 | escalate-all 1.000
  NOTE: nurse-alone 0.493 > model 0.391 and they are NOT comparable — see CLINICAL_AUDIT §6

WHOLE SYSTEM (real pipeline, 260 held-out, 9 deteriorated)
  any action 1.000 [0.701,1.000] · urgency raised 0.444 · false raise 0.187
  paths that caught them: ARRIVAL 3, TRAJECTORY_MATERIAL_WORSENING 1 — the MODEL DOES NOT APPEAR

TELEMETRY (62 patients)
  decision p50/p95/p99 = 50.5 / 62.8 / 76.8 ms (budget 2000 ms, MET)
  end-to-end 60.1 / 102.3 / 123.1 ms · cost/1000 visits $1.38 (stub backend)
```

---

## 7. THE 20 DESIGNED CASES — all verified after the simulator fix

18/20 arrival acuity exact; 20/20 escalate-iff-deteriorates. The two divergences are
**intentional one-level over-triages**: P-006 (L3 vs truth L4, defensible under 12:1) and
P-019 (L2 vs truth L3, *designed* that way for the override demo).

P-009 abstains · P-011 fires stridor + respiratory distress · P-012 escalates L3→L1 from the
attendant channel · P-015 presses 7× and moves nothing · P-016 does NOT abstain.

---

## 8. KNOWN WEAKNESSES (stated in the docs, not hidden)

1. `age_65_80` under-triage 0.181 — 7× working-age, non-overlapping CIs. **The largest
   clinical weakness.** Improved by age bands, not solved.
2. Adult calibration bias −0.0087 — the unsafe direction, in the largest group.
3. Geriatric calibration sign flips across retrains → needs a subgroup calibration gate.
4. Unexplained 0.25 sensitivity gap by sex (M 0.533 vs F 0.282), no mechanism identified.
5. Arrival over-triage 0.469, dominated by truth-L4 → assigned-L3 via the complaint floor.
6. Subgroup CIs too wide to conclude anything; the skin-tone ordering is currently the
   *opposite* of the injected penalty, which is the point.
7. Automation bias and feedback-loop contamination: unmeasured, unmeasurable here.

---

## 9. IF PICKING THIS UP AGAIN

Everything in the Round 2 brief is built. Optional polish, in order of value:

- Record a demo walkthrough (nurse → P-010 replay → P-009 abstention → LLM kill switch →
  validator → 403 → symmetric comparison → override). That sequence is the pitch.
- Slide deck drawing on `README.md` §"five things worth looking at".
- The `artifacts/saathi_audit.db` grows across runs; `AuditStore.purge_expired()` exists but
  is not scheduled.
- `docs/` has no diagram images — the architecture doc uses ASCII, which survives everywhere
  but is less striking than rendered SVG.
