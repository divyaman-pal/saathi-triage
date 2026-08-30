"""
Confidence decomposition and abstention.

NO ARBITRARY "87% CONFIDENT" NUMBERS. Confidence here is five separately
computed, separately displayed, separately auditable quantities, because they
answer five different questions and can fail independently:

    signal_quality     Could the sensors see what they claim to have seen?
    completeness       How much of the picture do we actually have?
    applicability      Was this model ever built for a patient like this one?
    channel_agreement  Do the three independent views tell the same story?
    calibration_status Do our probabilities mean what they say for THIS subgroup?

They are never collapsed into a single displayed number, because:

    signal quality != model confidence != clinical certainty != escalation priority

INFORMATIVE MISSINGNESS
-----------------------
Camera occlusion rises with waiting-room crowding, and crowding is exactly when
triage matters most. Treating that missingness as random would systematically
overstate confidence at the moment confidence should be lowest. The
signal_quality term therefore carries an explicit crowding penalty proportional
to the share of camera windows lost to occlusion, and the surge test asserts
that confidence FALLS as volume rises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contract_loader import Contract
from .features import FeatureVector
from .models import (
    AbstentionResult,
    Channel,
    ConfidenceComponents,
    Evidence,
    FreshnessStatus,
    QualityStatus,
)

QUALITY_SCORE = {
    QualityStatus.GOOD: 1.00,
    QualityStatus.ACCEPTABLE: 0.78,
    QualityStatus.DEGRADED: 0.45,
    QualityStatus.FAILED: 0.00,
    # A nurse's manual count has no sensor that can fail. Scoring it 'unknown'
    # would penalise the single most reliable channel in the department.
    QualityStatus.NOT_APPLICABLE: 1.00,
}

# Development-set size per calibration group at which we consider the model to
# have seen enough of this kind of patient. Below it, applicability degrades
# proportionally rather than stepping off a cliff.
APPLICABILITY_SUPPORT_TARGET = 500


@dataclass
class ConfidenceDetail:
    components: ConfidenceComponents
    explanations: dict[str, str] = field(default_factory=dict)
    failing: list[str] = field(default_factory=list)
    informative_missing_fraction: float = 0.0


def _channel_quality(usable: list[Evidence], rejected: list[Evidence]) -> tuple[float, float, dict[str, float]]:
    """
    Returns (quality_score, informative_missing_fraction, per_channel).

    Rejected items count. A camera that produced forty windows and had
    thirty-five thrown out is not a healthy camera, and averaging only the five
    survivors would report it as one.
    """
    per_channel: dict[str, list[float]] = {}
    for ev in usable:
        per_channel.setdefault(ev.source_channel.value, []).append(
            QUALITY_SCORE.get(ev.signal_quality.status, 0.5))
    for ev in rejected:
        if not ev.signal_quality.passed_floor:
            per_channel.setdefault(ev.source_channel.value, []).append(0.0)

    per_channel.pop(Channel.SYSTEM.value, None)
    if not per_channel:
        return 0.0, 1.0, {}

    means = {ch: sum(v) / len(v) for ch, v in per_channel.items()}
    score = sum(means.values()) / len(means)

    camera_items = [e for e in usable + rejected if e.source_channel is Channel.CAMERA]
    occluded = [e for e in camera_items
                if e.signal_quality.occlusion_pct is not None and e.signal_quality.occlusion_pct > 30]
    frac = (len(occluded) / len(camera_items)) if camera_items else 0.0

    # The crowding penalty. Missingness correlated with load is not missingness
    # at random, and must not be silently averaged away.
    score *= (1.0 - 0.35 * frac)
    return round(max(0.0, min(1.0, score)), 3), round(frac, 3), {k: round(v, 3) for k, v in means.items()}


def _completeness(fv: FeatureVector, importances: dict[str, float], names: list[str]) -> tuple[float, str]:
    """Importance-weighted feature presence. A missing high-gain feature costs more."""
    if not names:
        return 0.0, "no feature set"
    weights = {n: max(importances.get(n, 0.0), 1e-4) for n in names}
    total = sum(weights.values())
    have = sum(w for n, w in weights.items() if n in fv.present)
    plain = sum(1 for n in names if n in fv.present) / len(names)
    weighted = have / total if total else plain
    score = round(0.5 * plain + 0.5 * weighted, 3)
    return score, (f"{sum(1 for n in names if n in fv.present)}/{len(names)} features present "
                   f"(importance-weighted {weighted:.2f})")


def _applicability(
    calibrated: bool, group_support: int, live_channels: int, expected_channels: int, route: str
) -> tuple[float, str]:
    calib_term = 1.0 if calibrated else 0.30
    support_term = min(1.0, group_support / APPLICABILITY_SUPPORT_TARGET)
    channel_term = min(1.0, live_channels / max(1, expected_channels))
    score = 0.40 * calib_term + 0.30 * support_term + 0.30 * channel_term
    why = (f"route {route}; "
           f"{'calibrated' if calibrated else 'NOT calibrated'} for this age group; "
           f"development support n={group_support} (target {APPLICABILITY_SUPPORT_TARGET}); "
           f"{live_channels}/{expected_channels} channels live")
    return round(max(0.0, min(1.0, score)), 3), why


def compute(
    contract: Contract,
    fv: FeatureVector,
    usable: list[Evidence],
    rejected: list[Evidence],
    *,
    importances: dict[str, float],
    feature_name_list: list[str],
    agreement: float,
    calibration_status: str,
    group_support: int,
    live_channels: int,
    expected_channels: int,
    tier_ceiling: float = 1.0,
) -> ConfidenceDetail:
    sq, frac_occl, per_channel = _channel_quality(usable, rejected)
    comp, comp_why = _completeness(fv, importances, feature_name_list)
    calibrated = calibration_status.startswith("CALIBRATED")
    app, app_why = _applicability(calibrated, group_support, live_channels, expected_channels, fv.route)

    # A deployment tier with fewer channels caps how confident the system is
    # permitted to be, however clean the channels it does have happen to be.
    sq = round(min(sq, tier_ceiling), 3)
    app = round(min(app, tier_ceiling), 3)

    components = ConfidenceComponents(
        signal_quality=sq,
        completeness=comp,
        applicability=app,
        channel_agreement=round(max(0.0, min(1.0, agreement)), 3),
        calibration_status=calibration_status,
    )
    detail = ConfidenceDetail(
        components=components,
        informative_missing_fraction=frac_occl,
        explanations={
            "signal_quality": (f"per-channel means {per_channel}; "
                               f"{frac_occl:.0%} of camera windows lost to occlusion "
                               f"(crowding penalty applied - missingness here is NOT at random)"),
            "completeness": comp_why,
            "applicability": app_why,
            "channel_agreement": f"1 - normalised channel spread = {agreement:.2f}",
            "calibration_status": calibration_status,
        },
    )
    detail.failing = components.failing(contract.confidence_floors)
    return detail


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


PRIORITY_QUESTIONS = {
    "asr_failed": "Can the patient speak a full sentence without pausing for breath?",
    "camera_blind": "Is the patient sitting upright, and is their breathing visibly laboured?",
    "vitals_stale": "Please take a fresh set of vitals - the last set is beyond its validity window.",
    "no_attendant": "Is anyone with this patient who can tell you how they were an hour ago?",
    "default": "Can the patient tell you their own name and where they are?",
}


def evaluate_abstention(
    contract: Contract,
    detail: ConfidenceDetail,
    usable: list[Evidence],
    rejected: list[Evidence],
    *,
    attendant_present: bool,
    has_prior_record: bool,
    nurse_vitals_age_minutes: float | None,
    asr_confidence: float | None,
) -> AbstentionResult:
    """
    Decide whether the system should decline to score.

    ABSTENTION IS A FEATURE. It raises attention, never lowers it: acuity is
    held, the queue position is protected, the re-check timer is zeroed and a
    priority question is emitted for the nurse. A patient the system cannot see
    is a patient a human must see.

    Note what is deliberately NOT here: there is no path that produces a low
    risk estimate from degraded signals. The alternative to a confident answer
    is silence plus a request, never a hedged number.
    """
    c = detail.components
    tripped: list[str] = []
    failing_signals: list[str] = []
    missing_channels: list[str] = []
    stale_items: list[str] = []

    live_channels = {e.source_channel for e in usable} - {Channel.SYSTEM}
    fresh_nurse_vitals = any(
        e.source_channel is Channel.NURSE
        and e.concept_id in ("HEART_RATE", "RESP_RATE", "SPO2", "SBP")
        and e.freshness is not None
        and e.freshness.status is not FreshnessStatus.STALE
        for e in usable
    )

    for ev in rejected:
        if not ev.signal_quality.passed_floor and ev.signal_quality.floor_detail:
            d = ev.signal_quality.floor_detail
            if d not in failing_signals:
                failing_signals.append(d)
        if ev.freshness is not None and ev.freshness.status is FreshnessStatus.STALE:
            label = f"{ev.concept_id} ({ev.freshness.age_minutes:.0f} min, threshold {ev.freshness.max_staleness_minutes:.0f} min)"
            if not any(s.startswith(ev.concept_id) for s in stale_items):
                stale_items.append(label)

    if not attendant_present:
        missing_channels.append("attendant channel")
    if not has_prior_record:
        missing_channels.append("prior record")
    if Channel.CAMERA not in live_channels:
        missing_channels.append("camera channel")

    # ABST_SIGNAL_FLOOR
    if len(live_channels) < 1 and not fresh_nurse_vitals:
        tripped.append("ABST_SIGNAL_FLOOR")
    if not fresh_nurse_vitals and Channel.CAMERA not in live_channels:
        tripped.append("ABST_SIGNAL_FLOOR")

    # ABST_COMPLETENESS
    if c.completeness < 0.35:
        tripped.append("ABST_COMPLETENESS")

    # ABST_APPLICABILITY
    if c.applicability < 0.40:
        tripped.append("ABST_APPLICABILITY")

    # ABST_ASR_AND_NO_ATTENDANT
    # The `len(live_channels) < 2` clause is load-bearing. A non-verbal patient
    # with no family and no record, but with fresh nurse vitals and a working
    # camera, is a patient we can still assess PHYSIOLOGICALLY. Abstaining there
    # would abstain on a large share of Indian ED arrivals and make the system
    # useless. No history is not the same epistemic state as no observation.
    if (asr_confidence is not None and asr_confidence < 0.40
            and not attendant_present and not has_prior_record
            and len(live_channels) < 2):
        tripped.append("ABST_ASR_AND_NO_ATTENDANT")

    # ABST_CAMERA_OCCLUSION_SUSTAINED
    occl_failures = [e for e in rejected
                     if e.source_channel is Channel.CAMERA
                     and e.signal_quality.occlusion_pct is not None
                     and e.signal_quality.occlusion_pct > 55]
    if len(occl_failures) >= 4 and not fresh_nurse_vitals:
        tripped.append("ABST_CAMERA_OCCLUSION_SUSTAINED")

    tripped = list(dict.fromkeys(tripped))
    if not tripped:
        return AbstentionResult(abstained=False, missing_channels=missing_channels,
                                stale_items=stale_items, failing_signals=failing_signals)

    if asr_confidence is not None and asr_confidence < 0.40:
        q = PRIORITY_QUESTIONS["asr_failed"]
    elif not fresh_nurse_vitals:
        q = PRIORITY_QUESTIONS["vitals_stale"]
    elif Channel.CAMERA not in live_channels:
        q = PRIORITY_QUESTIONS["camera_blind"]
    elif not attendant_present:
        q = PRIORITY_QUESTIONS["no_attendant"]
    else:
        q = PRIORITY_QUESTIONS["default"]

    return AbstentionResult(
        abstained=True,
        gates_tripped=tripped,
        failing_signals=failing_signals,
        missing_channels=missing_channels,
        stale_items=stale_items,
        priority_question=q,
    )
