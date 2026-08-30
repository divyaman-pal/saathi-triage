# PROMPT — copy everything below this line into ChatGPT

---

You are a senior product designer and front-end engineer who has shipped software used inside
hospital emergency departments. You understand that a clinical interface is not a dashboard,
that density is a safety property rather than an aesthetic one, and that anything requiring a
scroll to reach a decision has failed.

I need you to design and build the front end for a working system called **SAATHI**.

Deliver a **single self-contained `index.html`** — inline CSS, inline vanilla JS, no build step,
no external requests except Google Fonts. It must open by double-clicking. Use the mock data I
give you at the bottom, shaped exactly as given, so the markup can later be wired to the real
backend without reshaping anything.

---

## 1. What the system is

SAATHI is clinical decision support for a hospital emergency department **waiting room** in
India. It is not a diagnostic device and it never diagnoses.

The insight it is built on: **triage happens once, but deterioration is continuous.** A patient
is assessed at the desk in two minutes, given a severity level, and then nobody looks at them
again until a clinician is free. SAATHI watches the waiting interval using three channels that
already exist and cost nothing per patient:

1. **Nurse** — spoken complaint plus vitals at the desk, once, then on re-check.
2. **Camera** — existing waiting-room camera. Arrival appearance (walked / carried / posture /
   visible breathing effort) and, while waiting, heart and respiratory rate estimated from video.
   Frames are processed on-device and discarded; only derived numbers persist.
3. **Family attendant** — prompts on the family's own phone, in their language. Observable tasks
   only: "count his breaths for 15 seconds", "can he tell you his own name", "is he more sleepy
   than before". This is the only channel that reliably catches new confusion.

The system fuses these, and **can only ever raise urgency. It can never lower it.** Only a human
can lower an acuity level, and that is recorded as an override.

## 2. Non-negotiable behaviours the UI must express

- **It can say "I don't know."** When signals are too degraded, it refuses to score, holds the
  patient at their current level, protects their queue position and demands a nurse re-check.
  This state must look *more* urgent, never less.
- **Deterministic red-flag rules** (stridor, unresponsive, shock) fire regardless of any model
  and cannot be suppressed. These outrank everything.
- **Contradictory evidence is always shown.** Every patient card has a populated "what argues
  against this" region. Surfacing only confirming evidence turns decision support into an
  anchoring device.
- **Confidence is five named axes**, never one blended number: signal quality, completeness,
  model applicability, channel agreement, calibration status. On the primary surface show a
  single qualitative chip that *names the weakest axis* — never an average, never a percentage.
- **An attendant never sees an acuity level, a score, or a queue position.** Not hidden — absent.

## 3. Four personas, four genuinely different screens

Do not restyle one screen four times. Change information depth, evidence selection, available
actions, and decision framing.

### PERSONA 1 — Triage nurse (the screen that matters most)

Standing at a desk, three other patients waiting, reading between tasks. **Under five seconds of
reading to a decision. If it needs scrolling to decide, it has failed.**

This is a **worklist of actions, not a browser of patients.** Most patients need nothing; the
screen shows the few that need something and what that something is.

Three tiers, in this order:

- **NEEDS YOU NOW** — red flags and cannot-assess cases. Always shown, never batched, never
  capped. Largest type on the page.
- **GOT WORSE WHILE WAITING** — patients whose level rose during the wait. This is SAATHI's
  entire reason to exist, so it deserves its own band. Capped at 5 visible; the rest behind
  "deferred by the alert budget, not discarded".
- **PAST THEIR SAFE WAIT** — collapsed to **one sentence**, not a list.

That last rule matters enormously. In the surge data below, 45 of 62 patients are past their
safe wait, because in a surge everyone is. Listing them individually is exactly the alarm
fatigue that gets clinical software switched off in week two. It is a floor-level staffing fact
addressed to the charge nurse, not 45 alerts aimed at one nurse.

Tapping a row opens **one patient, full screen** (not a side panel — one thing at a time):

- Patient ID, age, sex, minutes waited, whether past the safe limit
- Current acuity, and the arrival acuity if it changed (`ESI 2 ← was 3`)
- **The action, in the largest type on the screen** — e.g. "RE-CHECK" with the literal
  instruction: *"Count the respiratory rate yourself for a full 30 seconds and look for accessory
  muscle use."*
- **Why** — max 4 plain-English lines
- **What argues against** — always populated
- One confidence chip plus a "based on" line naming the sources and what is missing
- Two actions: **Agree** (one tap) and **Change the level** (level + structured reason)
- One disclosure — "Show the working" — containing evidence with timestamps and quality,
  the five confidence axes, the ranking breakdown, and full lineage

**Banned from the primary surface**, allowed only inside "Show the working": SNR figures,
reliability weights, SHAP values, confidence vectors, raw concept identifiers like `RESP_RATE`,
scheme version strings, probabilities. Nurses get English. A row must never read
`RESP_RATE delta=+8.0 SNR 4.2 w=0.6` — it reads **"Breathing rate up 8 in 28 min"**.

### PERSONA 2 — ED physician / charge nurse

Sitting, running the floor. Denser is fine and correct here. Whole-queue view, not per-patient:
who is past the safe wait and by how much; where the queue breaks against current staffing;
cross-patient patterns ("three respiratory presentations in 20 minutes"); override rate for the
shift; the surge posture and what the system changed about its own behaviour.

Include an honest capacity statement: when budget-eligible alerts exceed the ceiling, say so
plainly — some patients will wait longer than the design intends, and that is a staffing fact
the software cannot solve.

### PERSONA 3 — Family attendant (phone)

A frightened relative in a noisy room, possibly low-literacy, on a cheap Android. Portrait,
single column, max ~460px. **One question at a time**, very large type, thumb-sized buttons,
voice-first with pictorial fallback.

Only observable tasks — never clinical judgement. A visible "I am worried, ask a nurse to check"
button. And it must state, plainly: *asking brings a nurse to look; it does not change your place
in the queue.* There is nothing to gain by exaggerating and nothing to lose by asking.

Absolutely no score, no acuity, no queue position, no clinical interpretation, ever.

### PERSONA 4 — Hospital administrator / quality lead

Aggregate only, no patient identifiers. Override rate trended as the headline trust metric;
under- and over-triage rates; performance broken down by age band, sex, skin-tone band, language,
and attendant-present-or-not; cost and latency telemetry. Suppress any subgroup cell with fewer
than 5 patients and say why.

## 4. Visual direction

- **Calm, clinical, high-contrast. Not a fintech dashboard, not a neon "AI" aesthetic.**
  No gradients, no glow, no purple-blue tech clichés, no glassmorphism.
- Light theme primary; support dark via `prefers-color-scheme`.
- One clear accent per acuity level, consistent everywhere. Suggested: L1 deep red, L2 orange,
  L3 amber, L4 blue, L5 grey. Never rely on colour alone — always pair with a number and a word.
- Type: a clean humanist sans (Inter or IBM Plex Sans). Base 16px minimum; the nurse action line
  around 24–28px. Generous line height. Numerals tabular where they align in columns.
- Touch targets ≥ 44px — this runs on a shared, possibly grubby tablet.
- Must work at 1280×800 (desk) and 390×844 (phone, attendant view).
- Accessibility is not optional: WCAG AA contrast, real focus states, semantic landmarks,
  `aria-live="polite"` on the worklist, full keyboard operability.

## 5. What to produce

1. A single `index.html` with a persona switcher (top bar or left rail).
2. All four persona screens, fully rendered from the mock data.
3. The nurse worklist → patient card navigation working, with a back action.
4. Working "Agree" and "Change the level" interactions (log to console; show a confirmation).
5. The attendant flow answering one question and receiving a confirmation.
6. Realistic empty, degraded and cannot-assess states.
7. Clean CSS custom properties for the palette and spacing so it can be re-themed.

Then, separately, give me:
- A short rationale for each significant layout decision, tied to the constraint it serves.
- Anything you think I got wrong in the brief above, and what you'd do instead.

## 6. Mock data — use exactly these shapes

```js
// Worklist rows. Real system: 62 patients, 45 past SLA during a 3x surge.
const WORKLIST = [
 {patient_id:"P-011", acuity_now:1, acuity_arrival:1, waited_min:4,  max_wait_min:0,  sla_breached:true,  red_flag:true,  abstained:false, ewer:6.60, age:5,  sex:"M",
  action:"GO NOW", why:"Stridor, or unable to speak a full sentence."},
 {patient_id:"P-012", acuity_now:1, acuity_arrival:3, waited_min:22, max_wait_min:0,  sla_breached:true,  red_flag:true,  abstained:false, ewer:7.45, age:66, sex:"M",
  action:"GO NOW", why:"Family answers NO to 'Does he answer when you call his name?'"},
 {patient_id:"P-009", acuity_now:3, acuity_arrival:3, waited_min:45, max_wait_min:30, sla_breached:true,  red_flag:false, abstained:true,  ewer:3.80, age:40, sex:"M",
  action:"SEE THEM — CANNOT ASSESS", why:"Can the patient speak a full sentence without pausing for breath?"},
 {patient_id:"P-014", acuity_now:2, acuity_arrival:3, waited_min:34, max_wait_min:10, sla_breached:true,  red_flag:false, abstained:false, ewer:2.35, age:58, sex:"M",
  action:"RE-CHECK", why:"Breathing rate up 8 in 28 min"},
 {patient_id:"P-010", acuity_now:2, acuity_arrival:3, waited_min:36, max_wait_min:10, sla_breached:true,  red_flag:false, abstained:false, ewer:2.37, age:52, sex:"M",
  action:"RE-CHECK", why:"Breathing rate up 10 in 30 min"},
 {patient_id:"P-005", acuity_now:2, acuity_arrival:3, waited_min:30, max_wait_min:10, sla_breached:true,  red_flag:false, abstained:false, ewer:1.99, age:54, sex:"F",
  action:"RE-CHECK", why:"Family reports: new confusion"},
 {patient_id:"P-013", acuity_now:3, acuity_arrival:3, waited_min:38, max_wait_min:30, sla_breached:true,  red_flag:false, abstained:false, ewer:1.13, age:37, sex:"F",
  action:"DUE FOR RE-CHECK", why:"waiting 38 min, limit 30"},
 {patient_id:"P-020", acuity_now:4, acuity_arrival:4, waited_min:34, max_wait_min:60, sla_breached:false, red_flag:false, abstained:false, ewer:0.74, age:29, sex:"M",
  action:"", why:""}
];

const FLOOR = { total:62, needs_now:10, changed:9, past_sla:26, stable:17,
                surge_active:true, surge_ratio:3.0, alert_budget_per_hour:12, batch_size:5 };

// One patient card.
const CARD = {
 patient_id:"P-014", age:58, sex:"M",
 acuity_now:2, acuity_arrival:3, waited_min:34, max_wait_min:10,
 action:"RE-CHECK",
 instruction:"Count the respiratory rate yourself for a full 30 seconds and look for accessory muscle use.",
 why:["Breathing rate up 8 in 28 min",
      "Family reports: more sleepy than before",
      "Time without moving up 13 in 30 min",
      "Heart rate up 19 in 30 min"],
 against:["Temperature 37.4 °C",
          "Responds to name — yes",
          "New confusion — no",
          "How they arrived — walked unaided"],
 confidence:{ word:"PARTIAL", reason:"key information is missing",
   components:{signal_quality:0.90, completeness:0.69, applicability:0.90, channel_agreement:0.18},
   calibration:"CALIBRATED_adult_2026-08-29" },
 based_on:"nurse vitals + family · camera degraded · no previous records · sources disagree",
 missing:["Blood pressure","Oxygen","Previous records"],
 ewer:2.35,
 epistemic_state:"MATERIALITY_ESCALATION",
 evidence:[
  {signal:"Breathing rate", value:"26 /min", source:"family, guided count", observed:"11:28", age:"3 min", quality:"GOOD",     used:true},
  {signal:"Heart rate",     value:"108 bpm", source:"camera, rPPG",         observed:"11:31", age:"0 min", quality:"DEGRADED", used:true},
  {signal:"More sleepy",    value:"yes",     source:"family",               observed:"11:27", age:"4 min", quality:"—",        used:true},
  {signal:"Blood pressure", value:"—",       source:"nurse, cuff",          observed:"10:57", age:"34 min",quality:"STALE",    used:false},
  {signal:"Previous records",value:"—",      source:"health record",        observed:"—",     age:"—",     quality:"ABSENT",   used:false}
 ]
};

// The abstention case. Nothing here may be rendered as a risk number.
const CANNOT_ASSESS = {
 patient_id:"P-009", age:40, sex:"M", acuity_now:3, waited_min:45,
 action:"SEE THEM — CANNOT ASSESS",
 instruction:"Go and take a fresh set of vitals and a direct look. The system cannot see this patient.",
 priority_question:"Can the patient speak a full sentence without pausing for breath?",
 failing:["camera occlusion 62%","video pulse signal below usable floor","speech recognition confidence 0.34"],
 missing:["family attendant","previous records"],
 stale:["nurse vitals — 41 min old, limit 15"],
 held_at_level:3, queue_position_protected:true, recheck_timer_min:0
};

const ATTENDANT = {
 language:"Hindi",
 question:"Please count his breaths for 15 seconds and tell us the number.",
 type:"timed_observation_task", seconds:15,
 waited_min:34,
 escalation_note:"Asking brings a nurse to look at your patient. It does not change your place in the queue."
};

const ADMIN = {
 override_rate:0.08, overrides:5, contracts:62, validator_rejections:8, abstention_rate:0.05,
 by_age_band:[
  {band:"1–5 years",  n:6,  escalated:1, abstained:0, observation_only:4},
  {band:"18–65 years",n:34, escalated:6, abstained:2, observation_only:20},
  {band:"65–80 years",n:14, escalated:3, abstained:1, observation_only:9}],
 sensitivity_by_skin_tone:[
  {band:"I–II",  sens:0.42, lo:0.21, hi:0.66},
  {band:"III–IV",sens:0.39, lo:0.24, hi:0.56},
  {band:"V–VI",  sens:0.28, lo:0.11, hi:0.52}],
 decision_latency_p95_ms:63, budget_ms:2000, cost_per_1000_visits_usd:1.35
};
```

## 7. Two things I will judge your output on

1. **Can a nurse decide in under five seconds without scrolling?** If the patient card needs a
   scroll to reach the action and the two buttons, it has failed, however handsome it is.
2. **Does the "cannot assess" state read as more urgent than a normal one?** If refusing to score
   looks like a quiet grey empty state, it is dangerous and wrong. A patient the system cannot
   see is a patient a human must see.

Start with the nurse worklist and the patient card. Get those right before touching the others.
