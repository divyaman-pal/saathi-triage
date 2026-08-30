"""
Materiality engine, attendant routing, and SLA / re-check scheduling.

STATISTICAL EVIDENCE AND CLINICAL MATERIALITY ARE SEPARATE AXES
---------------------------------------------------------------
    WEAK evidence   + CRITICAL materiality  -> ESCALATE
    STRONG evidence + LOW materiality       -> LOG, DO NOT ALERT

A single family report of new confusion has almost no statistical power in any
tabular model and enormous clinical materiality. A statistically robust 2 bpm
heart-rate drift detected across forty windows has excellent statistical
properties and no clinical meaning whatsoever. A system that ranks by model
movement alone gets both of these wrong, in opposite and equally harmful
directions.

The matrix lives in contract/cost_policy.yaml. This module applies it.

ATTENDANT NON-GAMING
--------------------
attendant_escalation() is the enforcement point for the promise on the slide.
ATTENDANT_CONCERN routes to nurse_recheck_trigger and to nothing else. It has no
return path into acuity, EWER, or queue order. Pressing it fifty times buys
fifty nurse re-checks, capped by the re-check budget, and zero queue positions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contract_loader import Contract
from .fusion import Trajectory
from .models import (
    Channel,
    Evidence,
    MaterialityClass,
    MaterialityFinding,
    SLAStatus,
    StatisticalStrength,
    ThresholdBand,
)


# ---------------------------------------------------------------------------
# Materiality
# ---------------------------------------------------------------------------


def evaluate_materiality(
    contract: Contract,
    usable: list[Evidence],
    trajs: list[Trajectory],
    *,
    age_band: str,
    model_moved: bool,
    reporter_weight: float = 0.70,
    reporter_detail: str = "",
) -> list[MaterialityFinding]:
    """
    reporter_weight is the ANTI-GAMING GATE, and it is deliberately narrow.

    A proxy report from a reporter whose change-claims have been repeatedly
    contradicted by nurse checks is DOWNGRADED from ESCALATE to RECHECK. They
    still get the nurse. They do not get the acuity.

    What this gate does NOT touch:
      - red flags, which fire from the attendant channel regardless of weight
        (RF_UNRESPONSIVE_PROXY is not suppressible by anything, including this)
      - the nurse re-check itself, which every report still buys
      - a reporter with no adjudicated history, who sits at base weight and is
        believed
    """
    findings: list[MaterialityFinding] = []
    low_trust = reporter_weight <= 0.40

    # -- point-value materiality -------------------------------------------
    # ONLY for concepts whose contract entry declares `materiality_escalation`
    # in contributes_to. Those are the proxy-report concepts - new confusion,
    # increased sleepiness, unresponsiveness - which carry enormous clinical
    # weight and almost no statistical power, and which therefore have to reach
    # the escalation engine by a route that does not run through a model.
    #
    # Vital signs deliberately do NOT take this route, even though several of
    # them now carry a materiality_class. A concerning heart rate is already
    # fully represented twice - in the arrival acuity rules and in the tabular
    # model - and letting it escalate a third time here would be double
    # counting dressed up as clinical judgement. Their materiality_class exists
    # for the TRAJECTORY path below, where a measured trend is something
    # neither of the other two layers can see.
    for ev in usable:
        mclass = contract.materiality_class(ev.concept_id)
        if mclass is None:
            continue
        if "materiality_escalation" not in contract.contributes_to(ev.concept_id):
            continue
        if ev.threshold_band not in (ThresholdBand.CONCERNING, ThresholdBand.CRITICAL):
            continue
        power = contract.statistical_power(ev.concept_id) or "WEAK"
        action = contract.materiality_action(power, mclass)

        detail = (
            f"{ev.concept_id} reported via {ev.source_channel.value}. "
            f"Statistical evidence: {power}. Clinical materiality: {mclass}. "
            f"Contract action: {action}."
        )
        if mclass == "CRITICAL" and not model_moved:
            detail += (" The vital-sign model did not move. This escalation is on clinical "
                       "significance, not on model output.")
        if age_band in ("age_65_80", "age_80_plus") and ev.concept_id == "CONFUSION_NEW":
            detail += (" Delirium is frequently the sole presenting sign of serious illness "
                       "in this age band.")

        if low_trust and ev.source_channel is Channel.ATTENDANT and action == "ESCALATE":
            action = "RECHECK"
            detail = (
                f"{ev.concept_id} reported via attendant. DOWNGRADED from ESCALATE to RECHECK: "
                f"this reporter's change-claims have been repeatedly contradicted by nurse "
                f"re-checks (calibrated weight {reporter_weight:.2f}). {reporter_detail} "
                "The report still buys a nurse re-check. It does not move the acuity or the "
                "queue position."
            )

        findings.append(MaterialityFinding(
            concept_id=ev.concept_id,
            statistical_strength=StatisticalStrength(power),
            materiality_class=MaterialityClass(mclass),
            action=action,
            detail=detail,
            evidence_ids=[ev.evidence_id],
        ))

    # -- trajectory materiality --------------------------------------------
    # A measured TREND is evidence in exactly the same sense a family's report
    # is: it has a statistical strength and a clinical materiality, and the
    # contract matrix decides what to do with the pair. Previously only the
    # immaterial trends were emitted here, which meant a respiratory rate
    # climbing 7 over 22 minutes - the single most useful early warning an
    # adult gives you - reached the escalation engine as nothing at all.
    for t in trajs:
        if t.material:
            worsening = (
                (t.concept_id in ("HEART_RATE", "RESP_RATE", "TEMP", "STILLNESS_MINUTES")
                 and t.delta > 0)
                or (t.concept_id in ("SPO2", "SBP") and t.delta < 0))
            if not worsening:
                continue
            mclass = contract.materiality_class(t.concept_id) or "MODERATE"
            power = contract.statistical_power(t.concept_id) or "MODERATE"
            action = contract.materiality_action(power, mclass)
            ratio = abs(t.delta) / max(1e-6, t.materiality_threshold or 1.0)
            findings.append(MaterialityFinding(
                concept_id=t.concept_id,
                statistical_strength=StatisticalStrength(power),
                materiality_class=MaterialityClass(mclass),
                action=action,
                detail=(f"{t.concept_id} moved {t.delta:+g} over {t.minutes:.0f} min - "
                        f"{ratio:.1f}x the contract materiality threshold of "
                        f"{t.materiality_threshold:g}. Statistical evidence: {power}. "
                        f"Clinical materiality: {mclass}. Contract action: {action}. "
                        "This is a measured trend, not a snapshot: it is the change a "
                        "triage-once system cannot see."),
                evidence_ids=t.evidence_ids,
            ))
            continue
        # Detected, statistically real, clinically meaningless. Logged so the
        # audit trail shows the system saw it and correctly declined to alert.
        findings.append(MaterialityFinding(
            concept_id=t.concept_id,
            statistical_strength=StatisticalStrength.STRONG,
            materiality_class=MaterialityClass.LOW,
            action="LOG_DO_NOT_ALERT",
            detail=(f"{t.concept_id} changed {t.delta:+g} over {t.minutes:.0f} min. "
                    f"Below the contract materiality threshold of {t.materiality_threshold:g}. "
                    "Detected and logged; deliberately not alerted on."),
            evidence_ids=t.evidence_ids,
        ))

    order = {"ESCALATE": 0, "RECHECK": 1, "LOG_ONLY": 2, "LOG_DO_NOT_ALERT": 3}
    findings.sort(key=lambda f: order.get(f.action, 9))
    return findings


def materiality_escalates(findings: list[MaterialityFinding]) -> bool:
    return any(f.action == "ESCALATE" for f in findings)


ACCUMULATION_THRESHOLD = 3


def accumulation_escalates(findings: list[MaterialityFinding]) -> tuple[bool, list[str]]:
    """
    Three or more concurrent RECHECK-class findings, all worsening, escalate.

    None of them is individually enough, and that is the point. A family
    reporting increased sleepiness, fourteen unbroken minutes of stillness, and
    a heart rate trending upward are each a reason to walk over and look.
    Arriving together they are a reason to move the patient up the queue.

    This is also the honest answer to a specific failure mode: when the camera
    loses the respiratory-rate trend to a signal-quality drop - which happens
    exactly when the room is crowded, which is exactly when it matters - the
    single strongest trajectory signal disappears. Accumulation is what stops
    that from silently returning the patient to baseline.
    """
    rechecks = [f for f in findings if f.action == "RECHECK"]
    concepts = sorted({f.concept_id for f in rechecks})
    return len(concepts) >= ACCUMULATION_THRESHOLD, concepts


def materiality_boost(findings: list[MaterialityFinding]) -> float:
    boost = 0.0
    for f in findings:
        if f.action == "ESCALATE":
            boost += 0.30
        elif f.action == "RECHECK":
            boost += 0.12
    return round(min(0.60, boost), 3)


# ---------------------------------------------------------------------------
# Attendant routing - the non-gaming enforcement point
# ---------------------------------------------------------------------------


@dataclass
class AttendantEscalation:
    presses: int = 0
    recheck_triggered: bool = False
    acuity_effect: int = 0                # STRUCTURALLY ZERO. Never assigned.
    queue_effect: int = 0                 # STRUCTURALLY ZERO. Never assigned.
    ewer_effect: float = 0.0              # STRUCTURALLY ZERO. Never assigned.
    receipt: str = ""
    budget_note: str = ""


def attendant_escalation(usable: list[Evidence], *, rechecks_used_this_hour: int = 0,
                         recheck_budget_per_hour: int = 6) -> AttendantEscalation:
    """
    Convert attendant concern presses into nurse re-checks, and nothing else.

    The three `*_effect` fields are returned as literal zeros and are never
    written to. That is not a comment - it is the guarantee, and
    tests/test_invariants.py::test_attendant_non_gaming asserts that a patient's
    acuity, EWER rank and queue position are byte-identical with 0 presses and
    with 50.
    """
    presses = [e for e in usable if e.concept_id == "ATTENDANT_CONCERN" and bool(e.value)]
    n = len(presses)
    if n == 0:
        return AttendantEscalation()

    within_budget = rechecks_used_this_hour < recheck_budget_per_hour
    return AttendantEscalation(
        presses=n,
        recheck_triggered=within_budget,
        receipt=(f"{n} report{'s' if n > 1 else ''} received. "
                 + ("A nurse has been asked to check." if within_budget
                    else "A nurse has been asked to check; you are in the queue for re-checks.")),
        budget_note=(f"{rechecks_used_this_hour}/{recheck_budget_per_hour} attendant-triggered "
                     f"re-checks used this hour."),
    )


# ---------------------------------------------------------------------------
# SLA and re-check scheduling
# ---------------------------------------------------------------------------


def compute_sla(
    contract: Contract,
    *,
    level: int,
    waited_minutes: float,
    minutes_since_human_contact: float,
    has_prior_record: bool,
    abstained: bool,
    consent_declined: bool,
    low_confidence: bool,
    surge_active: bool,
    channel_degraded: bool,
    attendant_absent: bool,
) -> SLAStatus:
    """
    Every modifier in the contract is <= 1.0. There is no configuration of this
    function in which reduced information buys LESS human attention.
    """
    spec = contract.sla_for(level)
    base = float(spec["recheck_interval_minutes"])
    max_wait = float(spec["max_wait_minutes"])

    mods: dict[str, float] = {}
    if not has_prior_record:
        mods["no_prior_record"] = contract.recheck_modifier("no_prior_record")
    if consent_declined:
        mods["consent_declined"] = contract.recheck_modifier("consent_declined")
    if low_confidence:
        mods["low_confidence"] = contract.recheck_modifier("low_confidence")
    if surge_active:
        mods["surge_active"] = contract.recheck_modifier("surge_active")
    if channel_degraded:
        mods["channel_degraded"] = contract.recheck_modifier("channel_degraded")
    if attendant_absent:
        mods["attendant_absent"] = contract.recheck_modifier("attendant_absent")

    interval = base
    for f in mods.values():
        interval *= f
    if abstained:
        mods["abstention_active"] = 0.0
        interval = 0.0
    else:
        # The floor exists so modifiers cannot stack down to an unworkable
        # 30-second cadence. It must never RAISE the interval above the base -
        # level 1 has a base of 0 (see the clinician immediately), and a naive
        # max() would quietly convert that into "check back in 5 minutes".
        interval = min(base, max(contract.recheck_floor_minutes, interval))

    assert interval <= base + 1e-9, (
        "RECHECK MODIFIER VIOLATION: a modifier lengthened the interval. "
        "Reduced information must never buy less attention."
    )

    return SLAStatus(
        acuity_level=level,
        waited_minutes=round(waited_minutes, 1),
        max_wait_minutes=max_wait,
        breached=waited_minutes > max_wait,
        recheck_interval_minutes=round(interval, 1),
        recheck_due_in_minutes=round(max(0.0, interval - minutes_since_human_contact), 1),
        modifiers_applied={k: round(v, 3) for k, v in mods.items()},
    )


def sla_pressure(sla: SLAStatus) -> float:
    """0 at arrival, 1.0 at the SLA ceiling, rising beyond it."""
    if sla.max_wait_minutes <= 0:
        return 1.0
    return round(min(2.0, sla.waited_minutes / sla.max_wait_minutes), 3)


def live_channel_count(usable: list[Evidence]) -> int:
    return len({e.source_channel for e in usable} - {Channel.SYSTEM})
