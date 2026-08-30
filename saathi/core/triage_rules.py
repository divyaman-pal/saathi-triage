"""
Deterministic arrival triage.

This module assigns the ARRIVAL acuity, and it does so with no model at all -
only the contract's age-band severities, the clinician-authored red flags, and
the arrival gestalt. Three reasons it works this way:

  1. IT IS THE COLD-START SYSTEM. On day one at any site, before a single
     prospective case has been collected, this is what runs. Reporting its
     performance honestly (eval/evaluate.py) is reporting what the first month
     of deployment actually delivers.

  2. IT IS THE BASELINE THE MODEL MUST BEAT. A learned model that does not
     improve on age-band thresholds plus red flags is not worth the deployment
     risk, and we would rather find that out here than in a hospital.

  3. IT SEPARATES THE TWO QUESTIONS. Arrival acuity answers "how sick is this
     person now" - a classification. The model answers "is this person about to
     need escalation during the wait" - a different question with a different
     target. Conflating them is what makes a triage model look like it works
     while quietly re-deriving the nurse's own snapshot.

The model's job begins AFTER this. It can raise what this assigned. It can
never lower it.
"""

from __future__ import annotations

import math

from .features import FeatureVector
from .models import RedFlagHit

# ESI levels 3, 4 and 5 are RESOURCE PREDICTIONS, not severity statements. A
# patient with entirely normal vital signs and central chest discomfort is an
# ESI 3 because they are going to need an ECG and a troponin - two resources -
# not because anything about them looks abnormal at the desk.
#
# A triage rule that reads only vital signs assigns that patient a 5. This table
# is the floor the presenting complaint puts under the acuity, and it is the
# difference between "nothing is abnormal" and "nothing is abnormal YET".
COMPLAINT_RESOURCE_FLOOR = {
    # Two or more resources expected -> ESI 3 or better
    "chest_pain_central": 3, "chest_discomfort": 3, "dyspnoea": 3, "mild_dyspnoea": 3,
    "abdominal_pain": 3, "abdominal_discomfort": 3, "vomiting": 3, "diarrhoea": 3,
    "headache": 3, "dizziness": 3, "weakness": 3, "seizure_active": 3,
    "bleeding_uncontrolled": 3, "bleeding_pv": 3, "pregnant": 3, "confusion": 3,
    "lethargy": 3, "reduced_intake": 3, "fever": 3, "cough": 3, "back_pain": 3,
    "palpitations": 3, "hoarse_voice": 3, "stridor": 3,
    # One resource expected -> ESI 4 or better
    "limb_injury": 4, "laceration": 4, "myalgia": 4, "anxiety": 4,
}


def complaint_floor(symptoms: list[str]) -> tuple[int, str | None]:
    """The most urgent floor any presenting complaint puts under the acuity."""
    best, which = 5, None
    for s in symptoms:
        lvl = COMPLAINT_RESOURCE_FLOOR.get(s)
        if lvl is not None and lvl < best:
            best, which = lvl, s
    return best, which


def _g(fv: FeatureVector, key: str, default: float = float("nan")) -> float:
    v = fv.values.get(key, float("nan"))
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


def arrival_acuity(
    fv: FeatureVector,
    red_flag_hits: list[RedFlagHit],
    symptoms: list[str] | None = None,
) -> tuple[int, list[str]]:
    """
    Returns (ESI level, human-readable reasons).

    Band severities come from the contract: 0 normal, 1 unknown, 2 concerning,
    3 critical - already age-adjusted, so a heart rate of 148 arrives here as
    'normal' for a toddler and 'critical' for a seventy-five-year-old without
    this function knowing anything about heart rates.
    """
    reasons: list[str] = []

    if red_flag_hits:
        target = min(h.target_acuity.level for h in red_flag_hits)
        names = ", ".join(h.rule_id for h in red_flag_hits if h.target_acuity.level == target)
        return target, [f"Red flag fired: {names}"]

    avpu = _g(fv, "avpu_ord", 0.0)
    wob = _g(fv, "work_of_breathing", 0.0)
    arrival_mode = _g(fv, "arrival_mode_ord", 0.0)
    posture = _g(fv, "posture_ord", 0.0)
    n_crit = _g(fv, "n_critical", 0.0)
    n_conc = _g(fv, "n_concerning", 0.0)
    pain = _g(fv, "pain", 0.0)

    # -- Level 1: immediate life-saving intervention expected ---------------
    if avpu >= 2:                                   # responds to Pain, or Unresponsive
        return 1, ["AVPU is P or U"]
    if _g(fv, "band_spo2", 0.0) >= 3 and _g(fv, "band_sbp", 0.0) >= 3:
        return 1, ["Critical hypoxaemia together with critical hypotension"]
    if n_crit >= 3:
        return 1, [f"{int(n_crit)} vital signs in the age-band critical range simultaneously"]

    # -- Level 2: high risk, expected to need emergent care -----------------
    if n_crit >= 1:
        crit_names = [k.replace("band_", "") for k in
                      ("band_hr", "band_rr", "band_spo2", "band_sbp", "band_temp", "band_cap")
                      if _g(fv, k, 0.0) >= 3]
        reasons.append(f"age-band CRITICAL: {', '.join(crit_names)}")
        return 2, reasons
    if wob >= 2:
        return 2, ["Visible work of breathing at grade 2 or above"]
    if arrival_mode >= 3:                           # carried or stretchered
        return 2, ["Arrived carried or on a stretcher"]
    if avpu >= 1:                                   # responds to Voice only
        return 2, ["AVPU is V - not fully alert"]
    # Not all "concerning" values weigh the same. An abnormal oxygen saturation
    # or blood pressure is a statement about oxygenation and perfusion; an
    # abnormal temperature and heart rate in a febrile child is largely a
    # statement about having a fever. Counting them equally is how a well
    # toddler and a hypoxic adult end up at the same triage level.
    burden = n_conc
    if _g(fv, "band_spo2", 0.0) >= 2:
        burden += 1
    if _g(fv, "band_sbp", 0.0) >= 2:
        burden += 1

    if burden >= 5:
        return 2, [f"weighted abnormality burden {int(burden)} "
                   f"({int(n_conc)} concerning for age, oxygenation and/or perfusion among them)"]
    if n_conc >= 2 and arrival_mode >= 2:
        # Wheelchair or worse. 'Walked in assisted' is deliberately NOT enough:
        # most febrile toddlers arrive in a parent's arms, and treating that as
        # emergent would flood level 2 with well children.
        return 2, [f"{int(n_conc)} concerning vital signs for age, and arrived by wheelchair or carried"]

    # -- Level 3: expected to need two or more resources --------------------
    if n_conc >= 1:
        conc_names = [k.replace("band_", "") for k in
                      ("band_hr", "band_rr", "band_spo2", "band_sbp", "band_temp", "band_cap")
                      if _g(fv, k, 0.0) == 2]
        return 3, [f"age-band CONCERNING: {', '.join(conc_names) or 'vital sign'}"]
    if pain >= 7:
        return 3, [f"Self-reported pain {int(pain)}/10"]
    if arrival_mode >= 1 or posture >= 2 or wob >= 1:
        return 3, ["Did not walk in unaided, or abnormal posture / breathing effort observed"]

    # -- Level 4/5 ----------------------------------------------------------
    if pain >= 4:
        return 4, [f"Self-reported pain {int(pain)}/10, no abnormal vital signs for age"]
    return 5, ["No abnormal vital sign for age, walked in unaided, low reported pain"]


def arrival_acuity_with_complaint(
    fv: FeatureVector, red_flag_hits: list[RedFlagHit], symptoms: list[str]
) -> tuple[int, list[str]]:
    """
    Physiology first, then the resource floor implied by the complaint.

    The two are combined by taking the MORE urgent of the pair. A normal-looking
    patient with chest discomfort lands at 3 on the complaint; an abnormal-
    looking patient with a trivial complaint lands at 2 or better on the
    physiology. Neither can be talked down by the other.
    """
    physio_level, reasons = arrival_acuity(fv, red_flag_hits)
    if red_flag_hits:
        return physio_level, reasons

    floor, which = complaint_floor(symptoms or [])
    if floor < physio_level:
        reasons = reasons + [
            f"presenting complaint '{which}' is expected to consume "
            f"{'two or more' if floor <= 3 else 'one'} resources (ESI {floor} floor)"]
        return floor, reasons
    return physio_level, reasons


def escalation_target(current_level: int, *, floor: int = 2) -> int:
    """
    One level at a time, and never into level 1.

    Level 1 means "needs a life-saving intervention right now" and is reserved
    for the deterministic red-flag layer and for arrival physiology. A
    probabilistic model is not permitted to put a patient in the resuscitation
    room on its own - it can take them to level 2, where a human looks, and the
    human decides.
    """
    return max(floor, current_level - 1)
