"""
Signal quality gating and freshness evaluation.

Two rules govern this module, and both exist because the alternative is a system
that quietly lies:

  1. REJECT, DO NOT DOWN-WEIGHT. A value below its contract quality floor is
     removed from the feature set. Down-weighting a measurement we do not
     believe still lets it move the score, and still lets it be shown to a
     clinician as if it were an observation.

  2. NO SILENT IMPUTATION. A missing value stays missing. It is never replaced
     by a mean, a last-observation-carried-forward, or a model prediction and
     then presented as if it had been observed. Missingness propagates into the
     completeness component of confidence, where it belongs.

Missingness here is emphatically NOT at random: camera occlusion rises with
crowding, and crowding is exactly when triage matters most. That correlation is
modelled explicitly in core/confidence.py rather than assumed away.
"""

from __future__ import annotations

from datetime import datetime

from .contract_loader import Contract
from .models import (
    AcquisitionMethod,
    Evidence,
    Freshness,
    FreshnessStatus,
    QualityStatus,
    SignalQuality,
)


def evaluate_quality(
    contract: Contract,
    method: AcquisitionMethod,
    *,
    snr_db: float | None = None,
    occlusion_pct: float | None = None,
    asr_confidence: float | None = None,
    motion_index: float | None = None,
) -> SignalQuality:
    """Build a SignalQuality and decide whether it clears the contract floor."""
    metrics = {
        "snr_db": snr_db,
        "occlusion_pct": occlusion_pct,
        "asr_confidence": asr_confidence,
        "motion_index": motion_index,
    }
    present = {k: v for k, v in metrics.items() if v is not None}

    if not present:
        return SignalQuality(status=QualityStatus.NOT_APPLICABLE, passed_floor=True)

    # Overall status is the WORST band across the metrics we have. A window with
    # excellent SNR and 70% occlusion is a bad window.
    order = [QualityStatus.GOOD, QualityStatus.ACCEPTABLE, QualityStatus.DEGRADED, QualityStatus.FAILED]
    worst = QualityStatus.GOOD
    for metric, value in present.items():
        band = contract.quality_band(metric, value)
        if band is QualityStatus.NOT_APPLICABLE:
            continue
        if order.index(band) > order.index(worst):
            worst = band

    passed = True
    detail: str | None = None
    floor = contract.quality_floor(method)
    if floor:
        metric = floor["metric"]
        observed = present.get(metric)
        if observed is None:
            passed = False
            detail = (
                f"{method.value} requires {metric} to demonstrate it cleared its floor; "
                f"none was reported. Rejected rather than assumed adequate."
            )
        elif "min" in floor and observed < floor["min"]:
            passed = False
            detail = f"{metric}={observed:.2f} below floor {floor['min']} for {method.value} - value REJECTED"
        elif "max" in floor and observed > floor["max"]:
            passed = False
            detail = f"{metric}={observed:.1f} above ceiling {floor['max']} for {method.value} - value REJECTED"

    if not passed:
        worst = QualityStatus.FAILED

    return SignalQuality(
        snr_db=snr_db,
        occlusion_pct=occlusion_pct,
        asr_confidence=asr_confidence,
        motion_index=motion_index,
        status=worst,
        passed_floor=passed,
        floor_detail=detail,
    )


def evaluate_freshness(
    contract: Contract, concept_id: str, observed_at: datetime, now: datetime
) -> Freshness:
    """Age a value against its contract-declared maximum staleness."""
    age_s = max(0.0, (now - observed_at).total_seconds())
    max_min = contract.max_staleness_minutes(concept_id)
    cadence = contract.refresh_cadence_seconds(concept_id)

    if age_s / 60.0 > max_min:
        status = FreshnessStatus.STALE
    elif cadence is not None and age_s > cadence * 1.5:
        status = FreshnessStatus.AGING
    elif age_s / 60.0 > max_min * 0.6:
        status = FreshnessStatus.AGING
    else:
        status = FreshnessStatus.CURRENT

    return Freshness(
        age_seconds=age_s,
        max_staleness_minutes=max_min,
        status=status,
        expected_refresh_seconds=cadence,
    )


def gate(contract: Contract, evidence: list[Evidence], now: datetime) -> tuple[list[Evidence], list[Evidence]]:
    """
    Split evidence into (usable, rejected).

    Rejected items are NOT discarded. They travel with the assessment so the
    nurse can see that the camera tried and failed, rather than seeing a blank
    where a measurement should be. Degradation must be visible.
    """
    usable: list[Evidence] = []
    rejected: list[Evidence] = []
    for ev in evidence:
        if ev.freshness is None:
            ev = ev.model_copy(
                update={"freshness": evaluate_freshness(contract, ev.concept_id, ev.observed_at, now)}
            )
        (usable if ev.usable else rejected).append(ev)
    return usable, rejected


def latest_per_concept(evidence: list[Evidence]) -> dict[str, Evidence]:
    """
    Reconcile many grains to one patient-time grain.

    Where several channels report the same concept in the window, the one with
    the highest (reliability_weight, recency) wins as the value of record. The
    losers are retained for the discordance computation - they are not deleted,
    because the disagreement between them is itself a signal.
    """
    best: dict[str, Evidence] = {}
    for ev in evidence:
        cur = best.get(ev.concept_id)
        if cur is None:
            best[ev.concept_id] = ev
            continue
        if (ev.reliability_weight, ev.observed_at) > (cur.reliability_weight, cur.observed_at):
            best[ev.concept_id] = ev
    return best


def all_per_concept(evidence: list[Evidence]) -> dict[str, list[Evidence]]:
    out: dict[str, list[Evidence]] = {}
    for ev in evidence:
        out.setdefault(ev.concept_id, []).append(ev)
    return out
