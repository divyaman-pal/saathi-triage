"""
EWER - Evidence-Weighted Escalation Rank.

WHAT IT IS
    A transparent engineering ranking of which patients deserve human attention
    next. Its components are always displayed. A nurse can see why this patient
    rose above that one.

WHAT IT IS NOT
    Not a probability. Not a clinical certainty. Not a diagnosis. Not a severity
    score in any published triage scheme. It is deliberately NEVER rendered as a
    percentage and NEVER converted into a triage level - the UI shows it as a
    rank position plus its component bars, and the formatter below refuses to
    produce a "%" string.

Note that EWER and acuity are different objects with different jobs. Acuity is
a clinical classification with a scheme and a version. EWER is an ordering over
a queue at an instant. Two patients at the same acuity have an order; that
order is EWER's only job.
"""

from __future__ import annotations

from .models import AbstentionResult, ConfidenceComponents, EWERComponents, SLAStatus

# Weights are engineering parameters for an ORDERING, not clinical coefficients.
# They are visible, adjustable, and carry no probabilistic interpretation.
WEIGHTS = {
    "model_risk": 1.00,
    "confidence_penalty": 0.45,
    "discordance": 0.55,
    "since_human": 0.35,
    "sla_pressure": 0.50,
    "red_flag": 6.00,          # dominates everything, by construction
    "trajectory": 0.70,
    "materiality": 0.90,
    "abstention": 2.20,        # an unassessable patient ranks high, not low
}


def compute(
    *,
    model_risk: float | None,
    confidence: ConfidenceComponents,
    discordance_escalating: float,
    discordance_de_escalating: float,
    minutes_since_human_contact: float,
    sla: SLAStatus,
    red_flag_fired: bool,
    trajectory: float,
    materiality: float,
    abstention: AbstentionResult,
) -> EWERComponents:
    # A patient we cannot assess must not fall to the bottom of the list for
    # want of a number. Absent a model output we substitute a mid-scale
    # placeholder and let the abstention term carry the rank.
    risk = 0.35 if model_risk is None else float(model_risk)

    # Low confidence RAISES the rank. It never lowers it. This is the
    # degraded-mode escalation invariant expressed in the ordering.
    conf_penalty = round(1.0 - min(confidence.signal_quality,
                                   confidence.completeness,
                                   confidence.applicability), 3)

    since_human = round(min(1.5, minutes_since_human_contact / 40.0), 3)
    pressure = round(min(2.0, sla.waited_minutes / max(1.0, sla.max_wait_minutes)), 3) \
        if sla.max_wait_minutes > 0 else 1.0

    net_discordance = round(max(0.0, discordance_escalating - 0.5 * discordance_de_escalating), 3)

    c = EWERComponents(
        model_risk=round(risk, 3),
        confidence_penalty=conf_penalty,
        discordance=net_discordance,
        minutes_since_human_contact=since_human,
        sla_pressure=pressure,
        red_flag_boost=1.0 if red_flag_fired else 0.0,
        trajectory=round(trajectory, 3),
        materiality_boost=round(materiality, 3),
    )

    c.rank_value = round(
        WEIGHTS["model_risk"] * c.model_risk
        + WEIGHTS["confidence_penalty"] * c.confidence_penalty
        + WEIGHTS["discordance"] * c.discordance
        + WEIGHTS["since_human"] * c.minutes_since_human_contact
        + WEIGHTS["sla_pressure"] * c.sla_pressure
        + WEIGHTS["red_flag"] * c.red_flag_boost
        + WEIGHTS["trajectory"] * c.trajectory
        + WEIGHTS["materiality"] * c.materiality_boost
        + WEIGHTS["abstention"] * (1.0 if abstention.abstained else 0.0),
        4,
    )
    return c


def format_components(c: EWERComponents) -> list[tuple[str, float, float]]:
    """(label, raw component, weighted contribution) - for the UI bars."""
    keys = [
        ("model risk estimate", c.model_risk, WEIGHTS["model_risk"]),
        ("confidence penalty", c.confidence_penalty, WEIGHTS["confidence_penalty"]),
        ("channel discordance", c.discordance, WEIGHTS["discordance"]),
        ("time since human contact", c.minutes_since_human_contact, WEIGHTS["since_human"]),
        ("wait vs SLA", c.sla_pressure, WEIGHTS["sla_pressure"]),
        ("red flag", c.red_flag_boost, WEIGHTS["red_flag"]),
        ("trajectory", c.trajectory, WEIGHTS["trajectory"]),
        ("clinical materiality", c.materiality_boost, WEIGHTS["materiality"]),
    ]
    return [(label, round(raw, 3), round(raw * w, 3)) for label, raw, w in keys]


def render(c: EWERComponents) -> str:
    """
    Deliberately NOT a percentage and NOT a level.

    Attempting to format EWER as either is a category error, so the only
    renderer provided prints a bare index with an explicit disclaimer attached.
    """
    return f"EWER {c.rank_value:.2f} (ranking index - not a probability, not an acuity level)"
