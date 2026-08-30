"""
SAATHI simulated cohort.

=============================================================================
STATED GENERATIVE PROCESS - read this before interpreting any number produced
by this system.
=============================================================================

There is no public Indian emergency-department dataset with the multi-channel
structure SAATHI consumes. Rather than imply we have one, this module generates
a cohort from an explicitly declared process, so that every downstream result is
interpretable as "what the machinery does", never as "what the clinic will see".

  1. LATENT SEVERITY.  Each encounter has a latent severity s in [0, 1].
     For the 20 mandatory cases, s is hand-set to produce the specific clinical
     situation each case is designed to demonstrate. For the surge fill
     (P-021 onward) s ~ Beta(2, 5), giving a right-skewed population in which
     most arrivals are low severity - which matches the shape, though not the
     specifics, of a real ED case mix.

  2. AGE.  Drawn from a mixture chosen to resemble an Indian ED profile:
     18% under 12, 8% 12-18, 55% 18-65, 14% 65-80, 5% 80+. Paediatric load is
     deliberately higher than a US academic centre.

  3. BASELINE PHYSIOLOGY.  Vitals are drawn around the midpoint of the
     CONTRACT's normal range for the patient's age band, then shifted by
     severity. The shift is age-band aware:
        - children mount tachycardia and tachypnoea early, and hold blood
          pressure until late (compensated shock)
        - the 65+ bands have their tachycardic and febrile response
          deliberately BLUNTED by a factor of ~0.45, so that a severely unwell
          older adult presents with unimpressive numbers
     This is the mechanism by which an adult-calibrated single model becomes a
     silent safety hazard, and it is reproduced here on purpose so the
     age-stratified engine has something real to catch.

  4. DETERIORATION.  P(deteriorate during wait) = logistic(-3.9 + 5.4 s),
     giving a population rate of roughly 8-12% - the right order of magnitude
     for "gets meaningfully worse while waiting", and an order of magnitude
     below "is unwell on arrival". Onset time ~ Exponential with mean
     50 / (0.35 + s) minutes, censored at the end of the wait. After onset,
     vitals trend at a severity-scaled rate.

     Getting this rate right matters more than it looks: the cost-derived
     threshold is a function of the base rate, and calibrating a 12:1 policy
     against a 30% event rate would escalate almost everyone and quietly
     exceed any real department's alert budget.

  5. OBSERVATION MODEL.  The three channels see the same latent patient
     differently:
        - nurse:      observes the true value (measurement error only)
        - camera:     observes with noise inversely proportional to signal
                      quality; rPPG respiratory rate is noisier than rPPG heart
                      rate by design
        - attendant:  observes a coarse, discretised version with a per-reporter
                      bias term (over-reporters and under-reporters both exist)

  6. SIGNAL QUALITY.  Camera quality degrades with (a) waiting-room occupancy,
     (b) subject motion, and (c) Fitzpatrick skin-tone band. The occupancy term
     is the important one: it makes missingness INFORMATIVE, and correlated with
     exactly the conditions under which triage matters most.

  7. LABELS - TWO OF THEM, KEPT APART.
        truth_arrival_acuity : the correct ESI level at the desk. Evaluated
                               against the deterministic rules layer.
        truth_deteriorates   : whether the patient got materially worse during
                               the wait. Evaluated against the learned model.
     `needs_escalation` is the union of the two and is reported for reference
     only. It is NOT the model's training target - see data/generate.py for why
     a composite label would have hidden which layer was doing the work.

=============================================================================
WHAT THIS COHORT CAN AND CANNOT SUPPORT
=============================================================================
CAN:    demonstrate that the pipeline, the gating, the age-band engine, the
        fusion, the abstention path, the cost asymmetry, the claim validator
        and the invariant suite all behave as specified.
CANNOT: support any claim about clinical accuracy, sensitivity, calibration or
        safety in a real emergency department. A model trained on data drawn
        from a process we wrote will recover the process we wrote. Every
        performance number in eval/ is a property of this file, not of medicine.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

SkinToneBand = Literal["I-II", "III-IV", "V-VI"]


@dataclass
class VitalPlan:
    """Piecewise-linear vital trajectory over the wait."""

    hr: float
    rr: float
    spo2: float
    sbp: float | None            # None = no NIBP cuff available at this site
    temp: float
    avpu: str = "A"
    cap_refill: float = 1.8
    pain: int = 3

    onset_min: float | None = None       # deterioration onset; None = no deterioration
    prodrome_minutes: float = 15.0       # compensation phase before overt decline
    prodrome_gain: float = 0.45          # fraction of one decade-step reached by onset
    hr_per_10min: float = 0.0
    rr_per_10min: float = 0.0
    spo2_per_10min: float = 0.0
    sbp_per_10min: float = 0.0
    avpu_at_min: dict[float, str] = field(default_factory=dict)

    def _progress(self, t_min: float) -> float:
        """
        How far along the decline is, in units of one 'per 10 minutes' step.

        THE PRODROME IS THE WHOLE POINT. Real deterioration does not switch on:
        a patient compensates first - a respiratory rate creeping up, a heart
        rate drifting - and only then decompensates. If this simulator gave
        deterioration a hard onset with a flat run-up, there would be nothing
        observable before the event and no honest system could predict it.

        We model a `prodrome_minutes` window before onset over which the vitals
        move `prodrome_gain` of one decade-step. It is deliberately SUBTLE: a
        respiratory rate rising 3.2 per ten minutes drifts about 1.4 breaths
        across the whole prodrome. That is far too small for any single reading
        to reveal and is precisely what a TREND over the waiting interval is
        for - which is the claim the entire product rests on, so the simulator
        had better not contradict it.
        """
        if self.onset_min is None:
            return 0.0
        if t_min >= self.onset_min:
            return self.prodrome_gain + (t_min - self.onset_min) / 10.0
        start = self.onset_min - self.prodrome_minutes
        if t_min <= start:
            return 0.0
        return self.prodrome_gain * (t_min - start) / max(1e-6, self.prodrome_minutes)

    def at(self, t_min: float) -> dict[str, float | str | None]:
        k = self._progress(t_min)
        avpu = self.avpu
        for tm in sorted(self.avpu_at_min):
            if t_min >= tm:
                avpu = self.avpu_at_min[tm]
        return {
            "HEART_RATE": self.hr + self.hr_per_10min * k,
            "RESP_RATE": self.rr + self.rr_per_10min * k,
            "SPO2": self.spo2 + self.spo2_per_10min * k,
            "SBP": None if self.sbp is None else self.sbp + self.sbp_per_10min * k,
            "TEMP": self.temp,
            "AVPU": avpu,
            "CAP_REFILL": self.cap_refill,
            "PAIN_SELF_REPORT": self.pain,
        }


@dataclass
class AttendantScript:
    """
    What the family answers, and when.

    reporter_bias:
       0.0  = accurate reporter
      +1.0  = systematic over-reporter (answers yes to everything)
      -1.0  = systematic under-reporter (minimises, common where the family
              fears cost or fears being sent away)
    Calibration is learned per reporter over the visit in core/fusion.py.
    """

    present: bool = True
    reporter_bias: float = 0.0
    responds_to_name: dict[float, bool] = field(default_factory=dict)
    confusion_new: dict[float, bool] = field(default_factory=dict)
    sleepiness_increase: dict[float, bool] = field(default_factory=dict)
    guided_rr: dict[float, float] = field(default_factory=dict)
    concern_presses: list[float] = field(default_factory=list)
    free_text: dict[float, str] = field(default_factory=dict)
    transport: Literal["smartphone_app", "sms", "ivr", "none"] = "smartphone_app"


@dataclass
class CameraScript:
    """
    Camera availability and the quality trace over the wait.

    occlusion_override / snr_override are piecewise-constant traces keyed by the
    minute at which they take effect. Where absent, quality is generated from
    occupancy, motion and skin-tone band by the stated process above.
    """

    enabled: bool = True
    arrival_mode: str = "walked_unaided"
    work_of_breathing: int = 0
    posture: dict[float, str] = field(default_factory=dict)
    stillness_from_min: float | None = None
    skin_color_change: dict[float, str] = field(default_factory=dict)
    motion_index: float = 0.25
    occlusion_override: dict[float, float] = field(default_factory=dict)
    snr_override: dict[float, float] = field(default_factory=dict)
    reid_failure_at: float | None = None      # face track lost / re-identification failure


@dataclass
class PriorRecord:
    available: bool = False
    comorbidity_count: int = 0
    prior_ed_visits_90d: int = 0
    prior_icu_admission: bool = False
    record_age_days: int = 0
    conflicts_with_ed_system: bool = False
    note: str = ""


@dataclass
class PatientProfile:
    patient_id: str
    age_years: float
    sex: Literal["M", "F", "O"]
    skin_tone_band: SkinToneBand
    language: str
    asr_confidence: float
    complaint_text: str
    complaint_symptoms: list[str]

    vitals: VitalPlan
    camera: CameraScript
    attendant: AttendantScript
    prior: PriorRecord

    arrival_minutes_ago: float
    latent_severity: float
    consent_given: bool = True

    truth_arrival_acuity: int = 3
    truth_deteriorates: bool = False
    truth_deterioration_onset_min: float | None = None
    truth_outcome: str = "discharged"
    needs_escalation: bool = False

    nurse_recheck_minutes: list[float] = field(default_factory=lambda: [0.0])
    nurse_vitals_offset_min: float = 0.0   # how stale the last nurse reading is

    demonstrates: str = ""
    justification: str = ""
    is_mandatory_case: bool = False

    @property
    def has_prior_record(self) -> bool:
        return self.prior.available


# ---------------------------------------------------------------------------
# The 20 mandatory cases
# ---------------------------------------------------------------------------


def _p(pid: str, **kw) -> PatientProfile:
    kw.setdefault("skin_tone_band", "III-IV")
    kw.setdefault("language", "Hindi")
    kw.setdefault("asr_confidence", 0.86)
    kw.setdefault("camera", CameraScript())
    kw.setdefault("attendant", AttendantScript())
    kw.setdefault("prior", PriorRecord())
    kw.setdefault("is_mandatory_case", True)
    return PatientProfile(patient_id=pid, **kw)


def mandatory_cases() -> list[PatientProfile]:
    """
    The 20 hand-specified cases. These are DESIGNED, not sampled: each exists to
    force a specific code path to be visible in the demo. They are labelled as
    hand-specified everywhere they are reported, and they are excluded from the
    model's training set (see data/generate.py) so the model is never scored on
    cases written to make it look good.
    """
    cases: list[PatientProfile] = []

    # -- P-001 -- paediatric, age-band thresholds -----------------------------
    cases.append(_p(
        "P-001",
        age_years=3, sex="M",
        complaint_text="bukhar hai do din se, khana nahi kha raha",
        complaint_symptoms=["fever", "reduced_intake"],
        vitals=VitalPlan(hr=162, rr=34, spo2=96, sbp=92, temp=38.5, cap_refill=2.0, pain=3),
        camera=CameraScript(arrival_mode="walked_assisted", work_of_breathing=0, motion_index=0.45),
        attendant=AttendantScript(
            responds_to_name={5: True, 25: True}, confusion_new={5: False, 25: False},
            sleepiness_increase={5: False, 25: False}, guided_rr={12: 36}),
        arrival_minutes_ago=28, latent_severity=0.35,
        truth_arrival_acuity=3, truth_outcome="admitted", needs_escalation=False,
        nurse_recheck_minutes=[0.0],
        demonstrates="Age-band thresholds: HR 162 and RR 34 are CONCERNING for age 1-5, "
                     "and would be CRITICAL in an adult. TEMP 38.5 is CONCERNING here.",
        justification="Febrile toddler, tachycardic for age, well perfused, normal cap refill. "
                      "ESI 3. Age-appropriate reading prevents both over- and under-triage.",
    ))

    # -- P-002 -- geriatric, SAME 38.5 degC, different urgency ----------------
    cases.append(_p(
        "P-002",
        age_years=78, sex="F", skin_tone_band="III-IV",
        complaint_text="weakness since morning, not eating",
        complaint_symptoms=["weakness", "reduced_intake"],
        vitals=VitalPlan(hr=88, rr=20, spo2=94, sbp=118, temp=38.5, cap_refill=2.6, pain=2),
        camera=CameraScript(arrival_mode="wheelchair", work_of_breathing=0, motion_index=0.15,
                            posture={0: "slumped"}, stillness_from_min=6),
        attendant=AttendantScript(
            responds_to_name={8: True, 28: True}, confusion_new={8: False, 28: False},
            sleepiness_increase={8: False, 28: True}),
        prior=PriorRecord(available=True, comorbidity_count=3, prior_ed_visits_90d=1,
                          prior_icu_admission=False, record_age_days=112),
        arrival_minutes_ago=32, latent_severity=0.62,
        truth_arrival_acuity=2, truth_deteriorates=False,
        truth_outcome="admitted_ward", needs_escalation=True,
        demonstrates="THE CANONICAL AGE DEMO. Identical 38.5 degC to P-001, but CRITICAL in "
                     "age_65_80 because the febrile response is blunted. HR 88 and RR 20 are "
                     "'normal' numbers that carry no reassuring value in this band.",
        justification="Fever at 38.5 in a 78-year-old with no tachycardic response, slumped "
                      "posture and SpO2 94. ESI 2. An adult-calibrated model scores this as "
                      "unremarkable, which is precisely the silent safety risk.",
    ))

    # -- P-003 -- zero history, observation-only route ------------------------
    cases.append(_p(
        "P-003",
        age_years=34, sex="M", language="Bhojpuri", asr_confidence=0.71,
        complaint_text="pet me dard, ulti hui teen baar",
        complaint_symptoms=["abdominal_pain", "vomiting"],
        vitals=VitalPlan(hr=104, rr=21, spo2=97, sbp=122, temp=37.4, cap_refill=1.9, pain=6),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.35,
                            posture={0: "leaning"}),
        attendant=AttendantScript(present=False),
        prior=PriorRecord(available=False, note="No ABHA linkage. First presentation at this facility."),
        arrival_minutes_ago=20, latent_severity=0.30,
        truth_arrival_acuity=3, truth_outcome="discharged", needs_escalation=False,
        demonstrates="Zero-history path: OBSERVATION_ONLY_v1 model, reduced feature set, "
                     "MODERATE confidence, re-check interval halved to compensate.",
        justification="Roughly half of Indian ED arrivals have no retrievable record. This is "
                      "the norm, not an edge case, and the system must not fabricate risk from "
                      "absence nor impute a comorbidity count.",
    ))

    # -- P-004 -- returning patient, rich record ------------------------------
    cases.append(_p(
        "P-004",
        age_years=61, sex="M",
        complaint_text="saans phool rahi hai, do din se khaansi",
        complaint_symptoms=["dyspnoea", "cough"],
        vitals=VitalPlan(hr=106, rr=23, spo2=93, sbp=128, temp=37.8, cap_refill=2.2, pain=2),
        camera=CameraScript(arrival_mode="walked_assisted", work_of_breathing=1, motion_index=0.3),
        attendant=AttendantScript(responds_to_name={10: True}, confusion_new={10: False},
                                  sleepiness_increase={10: False}, guided_rr={15: 24}),
        prior=PriorRecord(available=True, comorbidity_count=4, prior_ed_visits_90d=3,
                          prior_icu_admission=True, record_age_days=34),
        arrival_minutes_ago=25, latent_severity=0.58,
        truth_arrival_acuity=2, truth_outcome="admitted", needs_escalation=True,
        demonstrates="Value of history, quantified. Same observations as a first-timer, but "
                     "4 comorbidities, 3 ED visits in 90 days and a prior ICU admission move "
                     "the FULL model materially above the OBSERVATION_ONLY model. The delta is "
                     "displayed, not asserted.",
        justification="ESI 2. History is the difference between 'breathless adult' and "
                      "'breathless adult who was ventilated five weeks ago'.",
    ))

    # -- P-005 -- calm but confused: materiality escalation --------------------
    cases.append(_p(
        "P-005",
        age_years=54, sex="F",
        complaint_text="chakkar aa raha tha ghar pe",
        complaint_symptoms=["dizziness"],
        vitals=VitalPlan(hr=84, rr=17, spo2=97, sbp=124, temp=37.1, cap_refill=1.7, pain=1),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.2,
                            posture={0: "upright"}),
        attendant=AttendantScript(
            responds_to_name={6: True, 24: True},
            confusion_new={6: False, 24: True},
            sleepiness_increase={6: False, 24: False},
            free_text={24: "wo mujhe pehchan nahi rahi, ghar pe aisi nahi thi"}),
        prior=PriorRecord(available=True, comorbidity_count=1, prior_ed_visits_90d=0,
                          record_age_days=380),
        arrival_minutes_ago=30, latent_severity=0.66,
        truth_arrival_acuity=3, truth_deteriorates=True, truth_deterioration_onset_min=24,
        truth_outcome="admitted", needs_escalation=True,
        demonstrates="CALM-BUT-CONFUSED. Every vital sign is inside its age band. The tabular "
                     "model does not move. The family reports new confusion. SAATHI escalates "
                     "on CLINICAL MATERIALITY with WEAK statistical support, and says so "
                     "explicitly. Age 54, so the geriatric red flag does NOT fire - this is the "
                     "materiality path, not the rule path.",
        justification="ESI 2 on re-check. This is the NHS Martha's Rule finding made operational: "
                      "82% of family-raised deterioration alerts would not trip a vital-sign score.",
    ))

    # -- P-006 -- anxious but well: discordance the other way -----------------
    cases.append(_p(
        "P-006",
        age_years=24, sex="F",
        complaint_text="dil bahut tez chal raha hai, ghabrahat ho rahi hai",
        complaint_symptoms=["palpitations", "anxiety"],
        vitals=VitalPlan(hr=118, rr=22, spo2=99, sbp=126, temp=36.9, cap_refill=1.5, pain=2),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.78,
                            posture={0: "upright", 10: "leaning", 20: "upright"}),
        attendant=AttendantScript(
            responds_to_name={5: True, 20: True}, confusion_new={5: False, 20: False},
            sleepiness_increase={5: False, 20: False}, guided_rr={14: 20},
            free_text={20: "jab bhi exam hota hai aisa hi hota hai, roz jaisa hi hai"}),
        arrival_minutes_ago=26, latent_severity=0.14,
        truth_arrival_acuity=4, truth_outcome="discharged", needs_escalation=False,
        demonstrates="ANXIOUS-BUT-WELL. Tachycardia and high motion index argue up; the family, "
                     "who know the baseline, argue down. Discordance in the DE-ESCALATING "
                     "direction prevents an escalation - it does NOT produce a downgrade, "
                     "because monotonicity forbids that. The patient holds at arrival acuity.",
        justification="ESI 4. Should not consume a resuscitation bay. Note what SAATHI is NOT "
                      "permitted to do here: it cannot lower her acuity and it cannot tell "
                      "anyone she is fine.",
    ))

    # -- P-007 -- stoic under-reporter ----------------------------------------
    cases.append(_p(
        "P-007",
        age_years=47, sex="M",
        complaint_text="thoda sa pet me kuch lag raha hai, kuch khaas nahi",
        complaint_symptoms=["abdominal_discomfort"],
        vitals=VitalPlan(hr=98, rr=18, spo2=96, sbp=132, temp=37.2, cap_refill=2.0, pain=1,
                         onset_min=6, rr_per_10min=2.8, hr_per_10min=3.5),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=1, motion_index=0.18,
                            posture={0: "leaning", 12: "slumped"}, stillness_from_min=14),
        attendant=AttendantScript(
            reporter_bias=-0.4,
            responds_to_name={8: True, 26: True}, confusion_new={8: False, 26: False},
            sleepiness_increase={8: False, 26: False}, guided_rr={16: 24},
            free_text={26: "kuch bolta nahi hai, dard hone pe bhi chup rehta hai"}),
        arrival_minutes_ago=34, latent_severity=0.60,
        truth_arrival_acuity=3, truth_deteriorates=True, truth_deterioration_onset_min=6,
        truth_outcome="admitted", needs_escalation=True,
        demonstrates="STOIC UNDER-REPORTER. Self-reported pain 1/10 against guarded posture, "
                     "increasing stillness and a respiratory rate climbing 18 -> 26. Self-report "
                     "carries reliability weight 0.40 by design; the denial is treated as "
                     "DISCORDANCE, never as reassurance.",
        justification="ESI 2 on re-check. Denied pain is the least reliable input in the room "
                      "and the one most often taken at face value.",
    ))

    # -- P-008 -- paediatric compensated shock --------------------------------
    cases.append(_p(
        "P-008",
        age_years=4, sex="F",
        complaint_text="loose motion do din se, sust ho gayi hai",
        complaint_symptoms=["diarrhoea", "lethargy"],
        vitals=VitalPlan(hr=165, rr=38, spo2=95, sbp=88, temp=37.9, cap_refill=4.0, pain=4),
        camera=CameraScript(arrival_mode="carried", work_of_breathing=1, motion_index=0.12,
                            posture={0: "supine"}, stillness_from_min=3),
        attendant=AttendantScript(responds_to_name={4: True, 18: True},
                                  confusion_new={4: False, 18: False},
                                  sleepiness_increase={4: True, 18: True}),
        arrival_minutes_ago=12, latent_severity=0.80,
        truth_arrival_acuity=2, truth_outcome="admitted_icu", needs_escalation=True,
        demonstrates="THRESHOLD MASKING. SBP 88 is NORMAL for age 1-5. A blood-pressure-driven "
                     "rule would clear this child. RF_PAEDS_COMPENSATED_SHOCK fires on the "
                     "combination of tachycardia + tachypnoea + capillary refill 4.0 s WHILE THE "
                     "BLOOD PRESSURE IS STILL NORMAL, which is the only time the finding is "
                     "useful.",
        justification="ESI 2. Children compensate until they do not. Waiting for paediatric "
                      "hypotension is waiting for the arrest.",
    ))

    # -- P-009 -- degraded signals: ABSTENTION --------------------------------
    cases.append(_p(
        "P-009",
        age_years=40, sex="M", skin_tone_band="V-VI",
        language="Santali", asr_confidence=0.34,
        complaint_text="[partially unparsed] ... dard ... [unintelligible]",
        complaint_symptoms=[],
        vitals=VitalPlan(hr=96, rr=20, spo2=96, sbp=118, temp=37.3, cap_refill=2.1, pain=4),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.55,
                            occlusion_override={0: 62.0}, snr_override={0: 1.8},
                            reid_failure_at=22),
        attendant=AttendantScript(present=False),
        prior=PriorRecord(available=False),
        arrival_minutes_ago=45, latent_severity=0.45,
        nurse_recheck_minutes=[0.0], nurse_vitals_offset_min=41.0,
        truth_arrival_acuity=3, truth_outcome="admitted", needs_escalation=True,
        demonstrates="MANDATORY ABSTENTION. Camera occluded 62%, rPPG SNR 1.8 (below the 3.0 "
                     "floor), ASR confidence 0.34, face covering, no attendant, no prior record, "
                     "nurse vitals 41 minutes old against a 15-minute threshold. The system "
                     "declines to score, holds acuity, protects the queue position, zeroes the "
                     "re-check timer and emits a priority question for the nurse.",
        justification="A patient the system cannot see is a patient a human must see. The "
                      "correct output here is an admission of blindness, not a confident guess "
                      "assembled from degraded inputs.",
    ))

    # -- P-010 -- silent deteriorator -----------------------------------------
    cases.append(_p(
        "P-010",
        age_years=52, sex="M",
        complaint_text="seene me halka dard tha subah se",
        complaint_symptoms=["chest_discomfort"],
        vitals=VitalPlan(hr=92, rr=18, spo2=97, sbp=134, temp=37.0, cap_refill=1.8, pain=3,
                         onset_min=16, rr_per_10min=3.4, hr_per_10min=5.0, spo2_per_10min=-0.8),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.22,
                            posture={0: "upright", 26: "slumped"}, stillness_from_min=22,
                            skin_color_change={30: "paler"}),
        attendant=AttendantScript(
            responds_to_name={8: True, 28: True}, confusion_new={8: False, 28: False},
            sleepiness_increase={8: False, 28: True}, guided_rr={30: 26}),
        prior=PriorRecord(available=True, comorbidity_count=2, prior_ed_visits_90d=0,
                          record_age_days=210),
        arrival_minutes_ago=36, latent_severity=0.68,
        truth_arrival_acuity=3, truth_deteriorates=True, truth_deterioration_onset_min=16,
        truth_outcome="admitted_icu", needs_escalation=True,
        demonstrates="THE CENTRAL THESIS. Arrival acuity of 3 was CORRECT. Nothing about the "
                     "triage snapshot was wrong. The patient changed during the wait, and no "
                     "triage system that scores once can see that. Escalates at T+34.",
        justification="Triage is an event; deterioration is a process. This case is the entire "
                      "argument for making the waiting interval the object of computation.",
    ))

    # -- P-011 -- red flag at arrival: stridor --------------------------------
    cases.append(_p(
        "P-011",
        age_years=5, sex="M",
        complaint_text="raat se awaaz aa rahi hai saans lene me, gala baith gaya",
        complaint_symptoms=["stridor", "cough", "hoarse_voice"],
        vitals=VitalPlan(hr=140, rr=32, spo2=94, sbp=96, temp=38.1, cap_refill=2.2, pain=2),
        camera=CameraScript(arrival_mode="carried", work_of_breathing=2, motion_index=0.4,
                            posture={0: "tripod"}),
        attendant=AttendantScript(responds_to_name={2: True}, confusion_new={2: False},
                                  sleepiness_increase={2: False}),
        arrival_minutes_ago=4, latent_severity=0.85,
        truth_arrival_acuity=1, truth_outcome="admitted_icu", needs_escalation=True,
        demonstrates="RULE SUPREMACY. RF_AIRWAY_STRIDOR fires from the complaint AND "
                     "independently from WORK_OF_BREATHING grade 2 on the camera. It escalates "
                     "to L1 before any model runs, and no model output can lower it. Tested with "
                     "every model disabled.",
        justification="ESI 1. Airway precedes everything.",
    ))

    # -- P-012 -- red flag from the attendant channel -------------------------
    cases.append(_p(
        "P-012",
        age_years=66, sex="M",
        complaint_text="sar me bahut dard tha, ab thik lag raha hai",
        complaint_symptoms=["headache"],
        vitals=VitalPlan(hr=76, rr=16, spo2=97, sbp=158, temp=36.9, cap_refill=2.0, pain=5,
                         avpu_at_min={18: "V"}),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.2,
                            posture={0: "upright", 16: "slumped"}, stillness_from_min=15),
        attendant=AttendantScript(
            responds_to_name={6: True, 18: False},
            confusion_new={6: False, 18: True},
            sleepiness_increase={6: False, 18: True},
            concern_presses=[18.5],
            free_text={18: "uth nahi raha, naam lene pe bhi nahi bol raha"}),
        prior=PriorRecord(available=True, comorbidity_count=2, prior_ed_visits_90d=0,
                          record_age_days=90),
        arrival_minutes_ago=22, latent_severity=0.88,
        truth_arrival_acuity=3, truth_deteriorates=True, truth_deterioration_onset_min=16,
        truth_outcome="admitted_icu", needs_escalation=True,
        demonstrates="ATTENDANT CHANNEL VALUE. Vitals at arrival were unremarkable. The family, "
                     "sitting beside him, is the first and only channel to detect that he has "
                     "stopped responding. RF_UNRESPONSIVE_PROXY fires at T+18 from a single "
                     "keypad press on a feature phone.",
        justification="ESI 1. The 4:1 attendant-to-patient ratio is the largest untapped "
                      "monitoring resource in an Indian ED waiting room.",
    ))

    # -- P-013 -- SLA breach with no clinical change --------------------------
    cases.append(_p(
        "P-013",
        age_years=37, sex="F",
        complaint_text="kal se kamar me dard hai",
        complaint_symptoms=["back_pain"],
        vitals=VitalPlan(hr=86, rr=16, spo2=98, sbp=120, temp=36.8, cap_refill=1.6, pain=5),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.3),
        attendant=AttendantScript(responds_to_name={10: True, 30: True},
                                  confusion_new={10: False, 30: False},
                                  sleepiness_increase={10: False, 30: False}),
        arrival_minutes_ago=38, latent_severity=0.18,
        truth_arrival_acuity=3, truth_outcome="discharged", needs_escalation=False,
        demonstrates="SLA-DRIVEN RE-CHECK WITHOUT ESCALATION. Waited 38 minutes against a "
                     "30-minute L3 ceiling. TR_SLA_BREACH forces a human re-assessment. The "
                     "ACUITY DOES NOT CHANGE - nothing clinical has changed, only the clock. "
                     "Conflating 'waited too long' with 'sicker' would corrupt the acuity signal.",
        justification="ESI 3, unchanged, with a mandatory re-check. The distinction between a "
                      "time obligation and a clinical finding is one most systems blur.",
    ))

    # -- P-014 -- multi-factor escalation, 5+ contributing signals ------------
    cases.append(_p(
        "P-014",
        age_years=58, sex="M",
        complaint_text="kamzori lag rahi hai, saans thodi bhaari hai",
        complaint_symptoms=["weakness", "mild_dyspnoea"],
        vitals=VitalPlan(hr=88, rr=20, spo2=96, sbp=126, temp=37.4, cap_refill=2.0, pain=2,
                         onset_min=12, rr_per_10min=3.2, hr_per_10min=7.0),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.2,
                            posture={0: "upright", 20: "slumped"}, stillness_from_min=20,
                            skin_color_change={0: "unchanged"},
                            snr_override={30: 3.2}),
        attendant=AttendantScript(
            responds_to_name={10: True, 30: True}, confusion_new={10: False, 30: False},
            sleepiness_increase={10: False, 30: True}, guided_rr={31: 26},
            reporter_bias=0.0),
        prior=PriorRecord(available=False),
        arrival_minutes_ago=34, latent_severity=0.64,
        truth_arrival_acuity=3, truth_deteriorates=True, truth_deterioration_onset_min=12,
        truth_outcome="admitted", needs_escalation=True,
        demonstrates="MULTI-FACTOR ACCUMULATION. No single finding here is enough to escalate. "
                     "Several arrive together: increased sleepiness reported by the family; "
                     "14 unbroken minutes of stillness (camera pose, GOOD); a rising heart rate "
                     "(camera rPPG, DEGRADED - SNR falls to 3.2 at T+30, so the respiratory-rate "
                     "trend is LOST at the moment it would have been most useful); wait 34 min "
                     "against a 30 min L3 SLA; and no prior record, which lowers confidence. One "
                     "signal ARGUES AGAINST: skin colour is unchanged, and it is displayed "
                     "alongside the rest. Three concurrent RECHECK-class findings trip the "
                     "accumulation rule and the patient moves to L2 at T+34.",
        justification="ESI 2 at T+34. This case exists to show what happens when the strongest "
                      "single signal is destroyed by degraded signal quality - which, because "
                      "occlusion tracks crowding, is exactly when it will happen in a real "
                      "waiting room. Each contribution is labelled as attribution or "
                      "association. None is labelled as a mechanism.",
    ))

    # -- P-015 -- attendant over-reports, patient stable ----------------------
    cases.append(_p(
        "P-015",
        age_years=45, sex="M",
        complaint_text="haath me chot lagi hai gir gaye the",
        complaint_symptoms=["limb_injury"],
        vitals=VitalPlan(hr=82, rr=16, spo2=98, sbp=124, temp=36.9, cap_refill=1.6, pain=4),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.35),
        attendant=AttendantScript(
            reporter_bias=1.0,
            responds_to_name={5: True, 15: True, 25: True, 35: True},
            confusion_new={5: True, 15: True, 25: True, 35: True},
            sleepiness_increase={5: True, 15: True, 25: True, 35: True},
            concern_presses=[6, 9, 14, 19, 24, 29, 33],
            free_text={9: "jaldi dekh lijiye please, bahut serious hai"}),
        arrival_minutes_ago=38, latent_severity=0.12,
        truth_arrival_acuity=4, truth_outcome="discharged", needs_escalation=False,
        nurse_recheck_minutes=[0.0, 12.0, 27.0],
        demonstrates="NON-GAMING AND REPORTER CALIBRATION. Seven concern presses and four "
                     "positive confusion reports, each contradicted by a nurse re-check and by "
                     "the camera. Each press bought a nurse re-check - which is the promised "
                     "behaviour - and ZERO queue positions. The reporter's calibrated weight "
                     "falls from 0.70 toward 0.30 as contradictions accumulate.",
        justification="ESI 4 throughout. The escalation path must be non-gameable in the SCORING "
                      "LAYER, not merely disclaimed on a slide.",
    ))

    # -- P-016 -- non-verbal, no attendant, no record -------------------------
    cases.append(_p(
        "P-016",
        age_years=71, sex="F", asr_confidence=0.0,
        complaint_text="",
        complaint_symptoms=[],
        vitals=VitalPlan(hr=104, rr=22, spo2=93, sbp=104, temp=37.6, cap_refill=3.0, pain=0),
        camera=CameraScript(arrival_mode="wheelchair", work_of_breathing=1, motion_index=0.1,
                            posture={0: "slumped"}, stillness_from_min=4),
        attendant=AttendantScript(present=False),
        prior=PriorRecord(available=False),
        arrival_minutes_ago=18, latent_severity=0.70,
        truth_arrival_acuity=2, truth_outcome="admitted", needs_escalation=True,
        demonstrates="WORST-CASE INFORMATION STATE that is NOT an abstention. No speech, no "
                     "family, no record - but the nurse vitals are fresh and the camera is "
                     "clean. Completeness is low, applicability is reduced, the re-check "
                     "interval is shortened. Contrast directly with P-009, where the SENSORS "
                     "failed rather than the HISTORY being absent.",
        justification="ESI 2. Missing history is not the same epistemic problem as failed "
                      "acquisition, and the system must not conflate them.",
    ))

    # -- P-017 -- language outside ASR coverage -------------------------------
    cases.append(_p(
        "P-017",
        age_years=29, sex="F", language="Gondi", asr_confidence=0.41,
        complaint_text="[dialect outside training coverage] ... pet ... [unparsed]",
        complaint_symptoms=["abdominal_pain"],
        vitals=VitalPlan(hr=100, rr=19, spo2=97, sbp=112, temp=37.5, cap_refill=1.9, pain=6),
        camera=CameraScript(arrival_mode="walked_assisted", work_of_breathing=0, motion_index=0.3,
                            posture={0: "leaning"}),
        attendant=AttendantScript(transport="ivr", responds_to_name={8: True, 24: True},
                                  confusion_new={8: False, 24: False},
                                  sleepiness_increase={8: False, 24: False}, guided_rr={20: 20}),
        prior=PriorRecord(available=False),
        arrival_minutes_ago=27, latent_severity=0.34,
        truth_arrival_acuity=3, truth_outcome="admitted_ward", needs_escalation=False,
        demonstrates="GRACEFUL DEGRADATION, NOT ABSTENTION. ASR confidence 0.41 is below the "
                     "0.55 floor, so the complaint is marked PARTIALLY UNPARSED and excluded "
                     "from symptom extraction. The attendant channel over IVR and the nurse "
                     "vitals still carry the assessment. One channel failing is not the same as "
                     "the system being blind.",
        justification="ESI 3. India has 22 scheduled languages and several hundred spoken ones. "
                      "ASR coverage failure must be a routine, well-handled event.",
    ))

    # -- P-018 -- consent declined --------------------------------------------
    cases.append(_p(
        "P-018",
        age_years=44, sex="M", consent_given=False,
        complaint_text="bukhar aur badan dard",
        complaint_symptoms=["fever", "myalgia"],
        vitals=VitalPlan(hr=102, rr=20, spo2=97, sbp=118, temp=38.6, cap_refill=2.0, pain=4),
        camera=CameraScript(enabled=False),
        attendant=AttendantScript(present=False),
        prior=PriorRecord(available=False),
        arrival_minutes_ago=24, latent_severity=0.38,
        truth_arrival_acuity=3, truth_outcome="admitted_ward", needs_escalation=False,
        demonstrates="OPT-OUT PATH. Consent declined at registration. Camera and attendant "
                     "channels are switched off for this patient. Queue position is UNAFFECTED. "
                     "The nurse re-check interval is HALVED to compensate for the loss of "
                     "observation. Opting out costs the patient nothing and costs the department "
                     "a little.",
        justification="ESI 3. DPDP Act 2023 requires a functional opt-out. A functional opt-out "
                      "is one that does not quietly degrade the patient's care.",
    ))

    # -- P-019 -- over-triage, clinician downgrades ---------------------------
    cases.append(_p(
        "P-019",
        age_years=33, sex="F",
        complaint_text="seene me dard hai aur ghabrahat",
        complaint_symptoms=["chest_pain_central", "anxiety"],
        vitals=VitalPlan(hr=122, rr=24, spo2=98, sbp=132, temp=37.0, cap_refill=1.6, pain=6),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=1, motion_index=0.7,
                            posture={0: "upright"}),
        attendant=AttendantScript(responds_to_name={7: True}, confusion_new={7: False},
                                  sleepiness_increase={7: False}),
        prior=PriorRecord(available=True, comorbidity_count=0, prior_ed_visits_90d=2,
                          record_age_days=45),
        arrival_minutes_ago=16, latent_severity=0.22,
        truth_arrival_acuity=3, truth_outcome="discharged", needs_escalation=False,
        demonstrates="OVER-TRIAGE AND OVERRIDE CAPTURE. The asymmetric cost threshold escalates "
                     "this patient to L2. The clinician examines her and downgrades to L3. The "
                     "override is captured with the full evidence snapshot, the Safety Contract "
                     "id, every model version, and the time from display to decision. This is "
                     "the cost asymmetry working as designed, and the trust loop closing.",
        justification="ESI 3 on examination. At a 12:1 cost ratio we EXPECT to over-triage cases "
                      "like this. An override rate near zero would mean the nurses are not "
                      "reading the screen.",
    ))

    # -- P-020 -- true negative ------------------------------------------------
    cases.append(_p(
        "P-020",
        age_years=29, sex="M",
        complaint_text="haath par kat gaya hai, khoon ruk gaya",
        complaint_symptoms=["laceration"],
        vitals=VitalPlan(hr=78, rr=15, spo2=99, sbp=122, temp=36.7, cap_refill=1.4, pain=3),
        camera=CameraScript(arrival_mode="walked_unaided", work_of_breathing=0, motion_index=0.4),
        attendant=AttendantScript(responds_to_name={10: True, 30: True},
                                  confusion_new={10: False, 30: False},
                                  sleepiness_increase={10: False, 30: False}, guided_rr={22: 16}),
        arrival_minutes_ago=34, latent_severity=0.05,
        truth_arrival_acuity=4, truth_outcome="discharged", needs_escalation=False,
        demonstrates="NOT EVERYTHING IS AN EMERGENCY. Stable in every channel for 34 minutes. "
                     "SAATHI reports 'no change observed in any channel' and is FORBIDDEN by the "
                     "claim grammar from saying he is stable, is fine, or can safely wait - even "
                     "though he is, in fact, fine. The system may describe its sensors; it may "
                     "not vouch for the patient.",
        justification="ESI 4 - a laceration needing closure consumes one resource, and ESI 5 "
                      "means none at all. A triage assistant that escalates everyone is a "
                      "triage assistant that will be switched off in a week.",
    ))

    return cases


# ---------------------------------------------------------------------------
# Surge fill - procedurally generated from the stated process
# ---------------------------------------------------------------------------

_AGE_MIX = [
    ((0.5, 12.0), 0.18),
    ((12.0, 18.0), 0.08),
    ((18.0, 65.0), 0.55),
    ((65.0, 80.0), 0.14),
    ((80.0, 92.0), 0.05),
]

_COMPLAINTS = [
    ("bukhar hai teen din se", ["fever"], 0.25),
    ("pet me dard", ["abdominal_pain"], 0.25),
    ("saans phool rahi hai", ["dyspnoea"], 0.55),
    ("seene me dard", ["chest_pain_central"], 0.60),
    ("chot lag gayi hai", ["limb_injury"], 0.10),
    ("ulti aur dast", ["vomiting", "diarrhoea"], 0.30),
    ("sar me dard", ["headache"], 0.20),
    ("kamzori lag rahi hai", ["weakness"], 0.35),
    ("chakkar aa raha hai", ["dizziness"], 0.30),
    ("khaansi aur bukhar", ["cough", "fever"], 0.30),
]

_LANGUAGES = ["Hindi", "Bhojpuri", "Marathi", "Bengali", "Tamil", "Telugu", "Awadhi", "Gondi"]
_TONES: list[SkinToneBand] = ["I-II", "III-IV", "V-VI"]


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _draw_age(rng: random.Random) -> float:
    r = rng.random()
    acc = 0.0
    for (lo, hi), w in _AGE_MIX:
        acc += w
        if r <= acc:
            return round(rng.uniform(lo, hi), 1)
    return round(rng.uniform(18.0, 65.0), 1)


def _baseline_vitals(age: float, s: float, rng: random.Random) -> VitalPlan:
    """
    Baseline physiology around the contract's normal midpoint, shifted by
    severity, with an explicit blunting factor for the 65+ bands.
    """
    if age < 1:
        hr0, rr0, sbp0 = 130, 44, 84
    elif age < 5:
        hr0, rr0, sbp0 = 120, 25, 92
    elif age < 12:
        hr0, rr0, sbp0 = 95, 21, 102
    elif age < 18:
        hr0, rr0, sbp0 = 82, 16, 110
    elif age < 65:
        hr0, rr0, sbp0 = 78, 16, 120
    else:
        hr0, rr0, sbp0 = 76, 16, 130

    paediatric = age < 12
    geriatric = age >= 65
    blunt = 0.45 if geriatric else 1.0

    hr = hr0 + (46 if paediatric else 40) * s * blunt + rng.gauss(0, 9)
    rr = rr0 + (14 if paediatric else 10) * s * blunt + rng.gauss(0, 3.1)
    spo2 = 98 - 9 * s + rng.gauss(0, 1.9)
    # Children hold blood pressure until decompensation: the severity term is
    # suppressed until s is high, then falls off a cliff.
    if paediatric:
        sbp = sbp0 - (0 if s < 0.82 else 34 * (s - 0.82) / 0.18) + rng.gauss(0, 4)
    else:
        sbp = sbp0 - 26 * s + rng.gauss(0, 7)
    temp = 36.9 + (1.9 * s if not geriatric else 1.0 * s) + rng.gauss(0, 0.62)
    cap = 1.5 + 2.0 * s + rng.gauss(0, 0.58)
    pain = max(0, min(10, int(round(3 + 6 * s + rng.gauss(0, 2)))))

    deteriorates = rng.random() < _logistic(-3.9 + 5.4 * s)
    onset = None
    if deteriorates:
        onset = min(55.0, rng.expovariate(1.0 / (50.0 / (0.35 + s))))

    return VitalPlan(
        hr=round(hr, 1), rr=round(rr, 1), spo2=round(min(100.0, spo2), 1),
        sbp=round(sbp, 1), temp=round(temp, 1), cap_refill=round(max(0.8, cap), 1), pain=pain,
        onset_min=onset,
        hr_per_10min=round(3.0 + 6.0 * s, 2) if deteriorates else 0.0,
        rr_per_10min=round(1.6 + 3.4 * s, 2) if deteriorates else 0.0,
        spo2_per_10min=round(-0.4 - 1.2 * s, 2) if deteriorates else 0.0,
        sbp_per_10min=round(-1.5 - 5.0 * s, 2) if deteriorates else 0.0,
    )


def _true_acuity(age: float, s: float, v: VitalPlan) -> int:
    """
    Ground-truth arrival acuity. Deliberately a function of LATENT severity and
    age, not of the observed vitals - so a model that reads only the vitals can
    be genuinely wrong, particularly in the blunted 65+ bands.
    """
    adj = s + (0.18 if age >= 65 else 0.0) + (0.10 if age < 5 else 0.0)
    if adj >= 0.88:
        return 1
    if adj >= 0.62:
        return 2
    if adj >= 0.32:
        return 3
    if adj >= 0.14:
        return 4
    return 5


def surge_fill(n: int, seed: int = 20260412, start_index: int = 21, occupancy: float = 3.0) -> list[PatientProfile]:
    """
    Procedurally generate the surge cohort at `occupancy` x normal volume.

    Camera quality degrades with occupancy, and degrades further for higher
    Fitzpatrick bands - the documented rPPG skin-tone bias, reproduced here so
    the subgroup monitoring in eval/ has something to detect.
    """
    rng = random.Random(seed)
    out: list[PatientProfile] = []
    for i in range(n):
        pid = f"P-{start_index + i:03d}"
        age = _draw_age(rng)
        s = rng.betavariate(2, 5)
        v = _baseline_vitals(age, s, rng)
        text, symptoms, sym_sev = rng.choice(_COMPLAINTS)
        tone = rng.choices(_TONES, weights=[0.15, 0.55, 0.30])[0]
        lang = rng.choice(_LANGUAGES)

        base_occl = 12.0 + 14.0 * (occupancy - 1.0) + rng.gauss(0, 6)
        base_occl = float(max(2.0, min(88.0, base_occl)))
        tone_snr_penalty = {"I-II": 0.0, "III-IV": 0.55, "V-VI": 1.25}[tone]
        base_snr = float(max(0.4, 5.6 - 0.7 * (occupancy - 1.0) - tone_snr_penalty + rng.gauss(0, 0.8)))

        attendant_present = rng.random() < 0.72
        has_record = rng.random() < 0.48
        asr_conf = float(max(0.05, min(0.98, rng.gauss(0.78, 0.18) - (0.22 if lang in ("Gondi", "Santali") else 0.0))))

        wait = round(rng.uniform(6, 55), 1)
        prompts = [t for t in (7.0, 19.0, 31.0, 43.0) if t < wait]

        att = AttendantScript(
            present=attendant_present,
            reporter_bias=rng.gauss(0.0, 0.45),
            transport=rng.choice(["smartphone_app", "sms", "ivr"]),
            # A family reporting that their relative will not answer to his own
            # name is a rare and grave thing to say. The earlier linear form made
            # it a coin flip at moderate severity, which fired the
            # RF_UNRESPONSIVE_PROXY rule on 9% of all arrivals and would have
            # buried a real department in false resuscitation calls. Logistic
            # forms keep these where they belong: rare in the well, common in
            # the genuinely obtunded.
            responds_to_name={t: (rng.random() > _logistic(-6.0 + 6.0 * s)) for t in prompts},
            confusion_new={t: (rng.random() < _logistic(-4.5 + 5.0 * s)) for t in prompts},
            sleepiness_increase={t: (rng.random() < _logistic(-3.6 + 4.6 * s)) for t in prompts},
            guided_rr={t: round(v.at(t)["RESP_RATE"] + rng.gauss(0, 3.2)) for t in prompts[:2]},
            concern_presses=[t + 0.5 for t in prompts if rng.random() < 0.10 + 0.30 * s],
        )

        cam = CameraScript(
            enabled=True,
            arrival_mode=rng.choices(
                ["walked_unaided", "walked_assisted", "wheelchair", "carried"],
                weights=[max(0.05, 0.72 - 0.6 * s), 0.16, 0.08 + 0.2 * s, 0.02 + 0.35 * s])[0],
            # Probabilistic, NOT a deterministic readout of latent severity.
            # A deterministic mapping here would leak s directly into the feature
            # set and hand the model an oracle it will never have in a clinic.
            work_of_breathing=(2 if rng.random() < _logistic(-5.0 + 5.6 * s)
                               else (1 if rng.random() < _logistic(-2.4 + 4.6 * s) else 0)),
            motion_index=float(max(0.05, min(0.95, rng.gauss(0.32, 0.18)))),
            posture={0.0: rng.choices(["upright", "leaning", "slumped"],
                                      weights=[max(0.1, 0.7 - 0.6 * s), 0.22, 0.08 + 0.5 * s])[0]},
            stillness_from_min=(wait * 0.55) if s > 0.55 and rng.random() < 0.5 else None,
            occlusion_override={0.0: base_occl},
            snr_override={0.0: base_snr},
        )

        prior = PriorRecord(
            available=has_record,
            comorbidity_count=int(max(0, round(rng.gauss(1.4 + 2.2 * s, 1.1)))) if has_record else 0,
            prior_ed_visits_90d=int(max(0, round(rng.gauss(0.6 + 1.8 * s, 0.9)))) if has_record else 0,
            prior_icu_admission=(rng.random() < 0.10 + 0.30 * s) if has_record else False,
            record_age_days=int(rng.uniform(5, 700)) if has_record else 0,
        )

        eff_s = min(1.0, s + 0.35 * sym_sev)
        acuity = _true_acuity(age, eff_s, v)
        deteriorates = v.onset_min is not None and v.onset_min < wait

        out.append(PatientProfile(
            patient_id=pid, age_years=age, sex=rng.choice(["M", "F"]),
            skin_tone_band=tone, language=lang, asr_confidence=asr_conf,
            complaint_text=text, complaint_symptoms=symptoms,
            vitals=v, camera=cam, attendant=att, prior=prior,
            arrival_minutes_ago=wait, latent_severity=eff_s,
            truth_arrival_acuity=acuity,
            truth_deteriorates=deteriorates,
            truth_deterioration_onset_min=v.onset_min if deteriorates else None,
            truth_outcome=("admitted_icu" if acuity <= 2 and deteriorates else
                           "admitted" if acuity <= 2 else
                           "admitted_ward" if acuity == 3 and rng.random() < 0.3 else "discharged"),
            needs_escalation=bool(deteriorates or acuity <= 2),
            nurse_recheck_minutes=[0.0],
            demonstrates="Surge fill - procedurally generated from the stated process.",
            is_mandatory_case=False,
        ))
    return out


def full_cohort(surge_n: int = 42, occupancy: float = 3.0, seed: int = 20260412) -> list[PatientProfile]:
    """20 designed cases + a 3x-volume surge fill. 62 encounters by default."""
    return mandatory_cases() + surge_fill(surge_n, seed=seed, occupancy=occupancy)


def training_population(n: int = 6000, seed: int = 7) -> list[PatientProfile]:
    """
    A larger draw from the SAME generative process, used to fit the tabular
    models. The 20 designed cases are NOT in here - the model is never trained
    on the cases used to demonstrate it.
    """
    return surge_fill(n, seed=seed, start_index=100000, occupancy=1.0)
