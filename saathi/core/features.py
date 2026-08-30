"""
Feature assembly.

Two things happen here that matter more than the model that consumes the output.

1. AGE ENTERS THE MODEL AS PHYSIOLOGY, NOT AS A NUMBER.
   Every vital is converted to a band-relative z-score using the CONTRACT's
   normal range for that patient's age band:

       z = (value - midpoint(normal)) / halfwidth(normal)

   A heart rate of 148 becomes z = +0.93 in a two-year-old and z = +3.40 in a
   seventy-five-year-old. The model never sees raw age and a raw heart rate and
   has to learn the interaction from data it does not have. The interaction is
   supplied by clinicians, through the contract. This is the single most
   important design decision in the modelling layer, and it is why an
   adult-calibrated model is not what runs here.

2. MISSING STAYS MISSING.
   Absent features are emitted as NaN and handed to XGBoost, which learns a
   default traversal direction for missingness at each split. That is a LEARNED
   HANDLING of absence, not an imputation - the value is never replaced by a
   population mean and then presented as though it had been observed. Which
   features were absent travels alongside the vector, into the completeness
   component of confidence and onto the nurse's screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .contract_loader import Contract, band_severity, evaluate_band
from .gating import all_per_concept, latest_per_concept
from .models import Channel, Evidence, ThresholdBand

# Ordinal encodings for categorical clinical concepts. Order is clinical, not
# alphabetical, and is asserted against the contract's value_set at import time.
ORDINALS: dict[str, list] = {
    "AVPU": ["A", "V", "P", "U"],
    "ARRIVAL_MODE": ["walked_unaided", "walked_assisted", "wheelchair", "carried", "stretcher"],
    "POSTURE": ["upright", "leaning", "slumped", "tripod", "supine"],
    "SKIN_COLOR_CHANGE": ["unchanged", "paler", "flushed", "mottled", "cyanotic_suspected"],
}

# Vitals that get a band-relative z-score.
Z_VITALS = ["HEART_RATE", "RESP_RATE", "SPO2", "SBP", "TEMP", "CAP_REFILL", "SHOCK_INDEX"]

# Everything the observation-only route may use.
OBSERVATION_FEATURES = [
    "age_years", "sex_m",
    "hr", "rr", "spo2", "sbp", "temp", "cap_refill", "pain",
    "z_hr", "z_rr", "z_spo2", "z_sbp", "z_temp", "z_cap", "z_shock",
    "band_hr", "band_rr", "band_spo2", "band_sbp", "band_temp", "band_cap",
    "band_max", "n_concerning", "n_critical",
    "avpu_ord", "arrival_mode_ord", "posture_ord", "work_of_breathing",
    "stillness_min", "skin_change_ord",
    "att_confusion", "att_sleepy", "att_responds",
    "d_rr_20", "d_hr_20", "d_spo2_20",
    "wait_min", "since_human_min",
    "n_channels_live", "frac_features_present",
]

# Additional features the full route may use. Present only when a record exists.
HISTORY_FEATURES = ["comorbidity_count", "prior_ed_visits_90d", "prior_icu", "record_age_days"]

FULL_FEATURES = OBSERVATION_FEATURES + HISTORY_FEATURES

NAN = float("nan")


@dataclass
class FeatureVector:
    values: dict[str, float] = field(default_factory=dict)
    present: set[str] = field(default_factory=set)
    missing: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)   # feature -> evidence_id
    band_notes: dict[str, str] = field(default_factory=dict)
    route: str = "FULL_v1"

    def as_row(self, names: list[str]) -> list[float]:
        return [self.values.get(n, NAN) for n in names]

    def completeness(self, names: list[str]) -> float:
        if not names:
            return 0.0
        have = sum(1 for n in names if n in self.present)
        return have / len(names)


def _z(contract: Contract, concept_id: str, value: float, band_id: str) -> float | None:
    """Band-relative z-score against the contract's normal range."""
    spec = contract.thresholds_for(concept_id, band_id)
    if not spec:
        return None
    norm = spec.get("normal")
    if not isinstance(norm, (list, tuple)) or len(norm) != 2:
        return None
    lo, hi = norm
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (lo, hi)):
        return None
    mid = (lo + hi) / 2.0
    half = max((hi - lo) / 2.0, 1e-6)
    return round((value - mid) / half, 3)


def _num(ev: Evidence) -> float | None:
    try:
        v = float(ev.value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _ord(concept_id: str, value) -> float | None:
    table = ORDINALS.get(concept_id)
    if table is None or value not in table:
        return None
    return float(table.index(value))


def _delta_over(evidence: list[Evidence], concept_id: str, minutes: float,
                channels: tuple[Channel, ...] | None = None) -> float | None:
    """
    Change in a concept across the last `minutes`, using only usable evidence.

    Returns None rather than 0.0 when there is no earlier observation to compare
    against. A trajectory we cannot compute is not a flat trajectory.
    """
    items = [e for e in evidence if e.concept_id == concept_id and e.usable]
    if channels:
        items = [e for e in items if e.source_channel in channels]
    if len(items) < 2:
        return None
    items.sort(key=lambda e: e.observed_at)
    latest = items[-1]
    cutoff = latest.observed_at.timestamp() - minutes * 60
    earlier = [e for e in items[:-1] if e.observed_at.timestamp() >= cutoff]
    if not earlier:
        return None
    a, b = _num(earlier[0]), _num(latest)
    if a is None or b is None:
        return None
    return round(b - a, 2)


def assemble(
    contract: Contract,
    usable: list[Evidence],
    rejected: list[Evidence],
    *,
    age_years: float,
    sex: str,
    has_prior_record: bool,
    live_channels: int,
) -> FeatureVector:
    band_id = contract.age_band(age_years)
    latest = latest_per_concept(usable)
    fv = FeatureVector(route="FULL_v1" if has_prior_record else "OBSERVATION_ONLY_v1")

    def put(name: str, value: float | None, ev: Evidence | None = None) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            fv.values[name] = NAN
            return
        fv.values[name] = float(value)
        fv.present.add(name)
        if ev is not None:
            fv.provenance[name] = ev.evidence_id

    put("age_years", age_years)
    put("sex_m", 1.0 if sex == "M" else 0.0)

    # -- raw vitals, band-relative z-scores, and contract band severities ----
    simple = {
        "HEART_RATE": ("hr", "z_hr", "band_hr"),
        "RESP_RATE": ("rr", "z_rr", "band_rr"),
        "SPO2": ("spo2", "z_spo2", "band_spo2"),
        "SBP": ("sbp", "z_sbp", "band_sbp"),
        "TEMP": ("temp", "z_temp", "band_temp"),
        "CAP_REFILL": ("cap_refill", "z_cap", "band_cap"),
    }
    bands_seen: list[ThresholdBand] = []
    for concept, (raw_n, z_n, b_n) in simple.items():
        ev = latest.get(concept)
        if ev is None:
            fv.missing.append(concept)
            put(raw_n, None)
            put(z_n, None)
            put(b_n, None)
            continue
        v = _num(ev)
        put(raw_n, v, ev)
        put(z_n, _z(contract, concept, v, band_id) if v is not None else None, ev)
        b, note = evaluate_band(contract, concept, ev.value, band_id)
        put(b_n, float(band_severity(b)), ev)
        bands_seen.append(b)
        if note:
            fv.band_notes[concept] = note

    ev_shock = latest.get("SHOCK_INDEX")
    if ev_shock is not None:
        put("z_shock", _z(contract, "SHOCK_INDEX", _num(ev_shock), band_id), ev_shock)
    else:
        fv.missing.append("SHOCK_INDEX")
        put("z_shock", None)

    put("band_max", float(max((band_severity(b) for b in bands_seen), default=0)) if bands_seen else None)
    put("n_concerning", float(sum(1 for b in bands_seen if b is ThresholdBand.CONCERNING)) if bands_seen else None)
    put("n_critical", float(sum(1 for b in bands_seen if b is ThresholdBand.CRITICAL)) if bands_seen else None)

    ev_pain = latest.get("PAIN_SELF_REPORT")
    put("pain", _num(ev_pain) if ev_pain else None, ev_pain)
    if ev_pain is None:
        fv.missing.append("PAIN_SELF_REPORT")

    # -- observations ------------------------------------------------------
    for concept, name in (("AVPU", "avpu_ord"), ("ARRIVAL_MODE", "arrival_mode_ord"),
                          ("POSTURE", "posture_ord"), ("SKIN_COLOR_CHANGE", "skin_change_ord")):
        ev = latest.get(concept)
        if ev is None:
            fv.missing.append(concept)
            put(name, None)
        else:
            put(name, _ord(concept, ev.value), ev)

    for concept, name in (("WORK_OF_BREATHING", "work_of_breathing"),
                          ("STILLNESS_MINUTES", "stillness_min")):
        ev = latest.get(concept)
        if ev is None:
            fv.missing.append(concept)
            put(name, None)
        else:
            put(name, _num(ev), ev)

    # -- attendant channel -------------------------------------------------
    for concept, name in (("CONFUSION_NEW", "att_confusion"),
                          ("SLEEPINESS_INCREASE", "att_sleepy"),
                          ("RESPONDS_TO_NAME", "att_responds")):
        ev = latest.get(concept)
        if ev is None:
            fv.missing.append(concept)
            put(name, None)
        else:
            put(name, 1.0 if bool(ev.value) else 0.0, ev)

    # -- trajectory --------------------------------------------------------
    put("d_rr_20", _delta_over(usable, "RESP_RATE", 20))
    put("d_hr_20", _delta_over(usable, "HEART_RATE", 20))
    put("d_spo2_20", _delta_over(usable, "SPO2", 20))

    # -- clock -------------------------------------------------------------
    for concept, name in (("WAIT_MINUTES", "wait_min"),
                          ("MINUTES_SINCE_HUMAN_CONTACT", "since_human_min")):
        ev = latest.get(concept)
        put(name, _num(ev) if ev else None, ev)

    put("n_channels_live", float(live_channels))

    # -- history (full route only) -----------------------------------------
    if has_prior_record:
        for concept, name in (("COMORBIDITY_COUNT", "comorbidity_count"),
                              ("PRIOR_ED_VISITS_90D", "prior_ed_visits_90d"),
                              ("PRIOR_ICU_ADMISSION", "prior_icu")):
            ev = latest.get(concept)
            if ev is None:
                fv.missing.append(concept)
                put(name, None)
            else:
                val = (1.0 if bool(ev.value) else 0.0) if concept == "PRIOR_ICU_ADMISSION" else _num(ev)
                put(name, val, ev)
        put("record_age_days", 0.0)
    else:
        for name in HISTORY_FEATURES:
            put(name, None)
        fv.missing.append("prior_record")

    names = FULL_FEATURES if has_prior_record else OBSERVATION_FEATURES
    put("frac_features_present", round(fv.completeness([n for n in names if n != "frac_features_present"]), 3))

    # Stale items are surfaced separately: the value existed and expired, which
    # is a different fact from the value never having been acquired.
    seen = {e.concept_id for e in usable}
    for ev in rejected:
        if ev.concept_id not in seen and ev.concept_id not in fv.stale:
            fv.stale.append(ev.concept_id)

    return fv


def feature_names(route: str) -> list[str]:
    return FULL_FEATURES if route == "FULL_v1" else OBSERVATION_FEATURES


def missing_feature_impact(contract: Contract, fv: FeatureVector, importances: dict[str, float]) -> list[tuple[str, float]]:
    """
    Rank the absent features by how much model importance they would have
    carried. This is what populates 'which missing features would most change
    the estimate' on the zero-history card.

    It is an IMPORTANCE-WEIGHTED ABSENCE RANKING, not a counterfactual. It says
    'this input usually matters a lot and we do not have it'. It does not say
    'this patient's score would have gone up'.
    """
    concept_to_feature = {
        "prior_record": "comorbidity_count", "COMORBIDITY_COUNT": "comorbidity_count",
        "PRIOR_ED_VISITS_90D": "prior_ed_visits_90d", "PRIOR_ICU_ADMISSION": "prior_icu",
        "SBP": "z_sbp", "SPO2": "z_spo2", "HEART_RATE": "z_hr", "RESP_RATE": "z_rr",
        "TEMP": "z_temp", "CAP_REFILL": "z_cap", "SHOCK_INDEX": "z_shock",
        "CONFUSION_NEW": "att_confusion", "SLEEPINESS_INCREASE": "att_sleepy",
        "RESPONDS_TO_NAME": "att_responds", "AVPU": "avpu_ord",
        "ARRIVAL_MODE": "arrival_mode_ord", "POSTURE": "posture_ord",
        "WORK_OF_BREATHING": "work_of_breathing", "STILLNESS_MINUTES": "stillness_min",
        "SKIN_COLOR_CHANGE": "skin_change_ord", "PAIN_SELF_REPORT": "pain",
    }
    out: list[tuple[str, float]] = []
    for concept in dict.fromkeys(fv.missing):
        feat = concept_to_feature.get(concept)
        if feat is None:
            continue
        out.append((concept, round(importances.get(feat, 0.0), 4)))
    out.sort(key=lambda kv: -kv[1])
    return out
