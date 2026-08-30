"""
Deterministic red-flag rule engine.

This module has no imports from risk_model, fusion, confidence, cost_engine or
llm, and it never will. That absence is the architecture: there is no code path
by which a model output can reach this evaluation, and therefore no way for a
model to suppress a clinician-authored rule.

Two deliberate design choices:

  1. RED FLAGS IGNORE STALENESS. A rule fires on any evidence that cleared its
     QUALITY floor, whether or not it has since gone stale. Stridor heard forty
     minutes ago has not been cancelled by the clock; it has been cancelled only
     when a human says so. A red flag that silently expires is a safety hole.
     Staleness is recorded on the hit and displayed, but it does not gate.

  2. RED FLAGS LATCH. Once fired during an encounter, a flag remains fired until
     a human clears it with a recorded reason. See RedFlagLatch.

Tested by:
  tests/test_invariants.py::test_red_flag_supremacy
  tests/test_invariants.py::test_red_flags_fire_with_all_models_disabled
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contract_loader import Contract, evaluate_band
from .models import Acuity, Channel, Evidence, RedFlagHit, ThresholdBand


@dataclass
class RuleContext:
    age_years: float
    age_band: str
    complaint_symptoms: list[str] = field(default_factory=list)
    attendant_symptoms: list[str] = field(default_factory=list)
    avpu_baseline: str = "A"


def _values_for(evidence: list[Evidence], concept_id: str) -> list[Evidence]:
    return [e for e in evidence if e.concept_id == concept_id and e.signal_quality.passed_floor]


def _latest(evidence: list[Evidence], concept_id: str) -> Evidence | None:
    items = _values_for(evidence, concept_id)
    return max(items, key=lambda e: e.observed_at) if items else None


def _num(ev: Evidence) -> float | None:
    try:
        return float(ev.value)
    except (TypeError, ValueError):
        return None


def _check(
    contract: Contract, cond: dict, evidence: list[Evidence], ctx: RuleContext
) -> tuple[bool, list[str]]:
    """Evaluate one machine-readable condition. Returns (fired, evidence_ids)."""
    concept = cond["concept"]
    op = cond["op"]
    ref = cond.get("value")

    # -- pseudo-concepts ---------------------------------------------------
    if concept == "_age_band":
        return (ctx.age_band in ref if op == "in" else ctx.age_band == ref), []
    if concept == "_age_years":
        if op == "gte":
            return ctx.age_years >= ref, []
        if op == "lt":
            return ctx.age_years < ref, []
        return False, []
    if concept == "COMPLAINT_SYMPTOM":
        return (ref in ctx.complaint_symptoms), []
    if concept == "ATTENDANT_SYMPTOM":
        return (ref in ctx.attendant_symptoms), []

    # -- real concepts -----------------------------------------------------
    if op == "acquisition_in":
        items = [e for e in _values_for(evidence, concept) if e.acquisition_method.value in ref]
        return bool(items), [e.evidence_id for e in items[-1:]]

    if op == "worsened_from_baseline":
        items = _values_for(evidence, concept)
        if not items:
            return False, []
        order = ["A", "V", "P", "U"]
        worst = max(items, key=lambda e: order.index(str(e.value)) if str(e.value) in order else -1)
        base_i = order.index(ctx.avpu_baseline) if ctx.avpu_baseline in order else 0
        cur_i = order.index(str(worst.value)) if str(worst.value) in order else 0
        return cur_i > base_i, [worst.evidence_id]

    ev = _latest(evidence, concept)
    if ev is None:
        return False, []
    eid = [ev.evidence_id]

    if op == "age_band_critical":
        band, _ = evaluate_band(contract, concept, ev.value, ctx.age_band)
        return band is ThresholdBand.CRITICAL, eid
    if op == "age_band_at_least_concerning":
        band, _ = evaluate_band(contract, concept, ev.value, ctx.age_band)
        return band in (ThresholdBand.CONCERNING, ThresholdBand.CRITICAL), eid

    v = _num(ev)
    if op == "eq":
        return ev.value == ref, eid
    if op == "in":
        return ev.value in ref, eid
    if op == "contains":
        return ref in str(ev.value), eid
    if v is None:
        return False, []
    if op == "gte":
        return v >= ref, eid
    if op == "gt":
        return v > ref, eid
    if op == "lte":
        return v <= ref, eid
    if op == "lt":
        return v < ref, eid
    return False, []


def evaluate(contract: Contract, evidence: list[Evidence], ctx: RuleContext) -> list[RedFlagHit]:
    """
    Run the whole clinician-authored ruleset. Pure function of the contract and
    the evidence. Takes no model, no confidence, no configuration.
    """
    hits: list[RedFlagHit] = []
    for rule in contract.red_flags["rules"]:
        trig = rule["trigger_machine"]
        fired = True
        eids: list[str] = []

        if "all" in trig:
            for cond in trig["all"]:
                ok, ids = _check(contract, cond, evidence, ctx)
                eids += ids
                if not ok:
                    fired = False
                    break
        if fired and "any" in trig:
            any_ok = False
            for cond in trig["any"]:
                ok, ids = _check(contract, cond, evidence, ctx)
                if ok:
                    any_ok = True
                    eids += ids
            fired = any_ok
        if "all" not in trig and "any" not in trig:
            fired = False

        if fired:
            hits.append(RedFlagHit(
                rule_id=rule["rule_id"],
                name=rule["name"],
                trigger_human=rule["trigger_human"].strip(),
                target_acuity=Acuity(scheme_id="ESI", scheme_version="v4",
                                     level=int(rule["target_acuity"])),
                fired_on_evidence=list(dict.fromkeys(eids)),
                source_channels=[Channel(c) for c in rule.get("source_channel", [])
                                 if c in {x.value for x in Channel}],
                rationale=rule.get("rationale", "").strip(),
                ruleset_version=contract.ruleset_version,
            ))
    hits.sort(key=lambda h: h.target_acuity.level)
    return hits


def highest(hits: list[RedFlagHit]) -> Acuity | None:
    return hits[0].target_acuity if hits else None


class RedFlagLatch:
    """
    Once fired, a flag stays fired for the encounter until a human clears it.

    A camera that stops seeing accessory muscle use has not established that the
    child's airway is fine; it has established that the camera stopped seeing.
    """

    def __init__(self) -> None:
        self._latched: dict[str, dict[str, RedFlagHit]] = {}
        self._cleared: dict[str, dict[str, str]] = {}

    def apply(self, patient_id: str, hits: list[RedFlagHit]) -> list[RedFlagHit]:
        store = self._latched.setdefault(patient_id, {})
        cleared = self._cleared.get(patient_id, {})
        for h in hits:
            if h.rule_id not in cleared:
                store[h.rule_id] = h
        out = list(store.values())
        out.sort(key=lambda h: h.target_acuity.level)
        return out

    def clear(self, patient_id: str, rule_id: str, clinician_id: str, reason: str) -> None:
        self._cleared.setdefault(patient_id, {})[rule_id] = f"{clinician_id}: {reason}"
        self._latched.get(patient_id, {}).pop(rule_id, None)

    def cleared_flags(self, patient_id: str) -> dict[str, str]:
        return dict(self._cleared.get(patient_id, {}))


# ---------------------------------------------------------------------------
# Deterministic time rules
# ---------------------------------------------------------------------------


@dataclass
class TimeRuleHit:
    rule_id: str
    name: str
    action: str
    escalates_acuity: bool
    detail: str
    factor: float | None = None


def evaluate_time_rules(
    contract: Contract,
    *,
    waited_minutes: float,
    max_wait_minutes: float,
    minutes_since_human_contact: float,
    recheck_interval_minutes: float,
    minutes_since_any_observation: float,
    consent_declined: bool,
) -> list[TimeRuleHit]:
    """
    Clock-only rules. They force human attention; with one exception they do NOT
    raise acuity, because nothing clinical has changed - only time has passed.
    Conflating 'waited too long' with 'sicker' would corrupt the acuity signal.
    """
    out: list[TimeRuleHit] = []
    by_id = {r["rule_id"]: r for r in contract.red_flags["time_rules"]}

    if waited_minutes > max_wait_minutes:
        r = by_id["TR_SLA_BREACH"]
        out.append(TimeRuleHit(
            r["rule_id"], r["name"], r["action"], r["escalates_acuity"],
            f"Waited {waited_minutes:.0f} min against a {max_wait_minutes:.0f} min ceiling for this acuity."))

    if minutes_since_human_contact > 2 * max(1.0, recheck_interval_minutes):
        r = by_id["TR_NO_HUMAN_CONTACT"]
        out.append(TimeRuleHit(
            r["rule_id"], r["name"], r["action"], r["escalates_acuity"],
            f"No human assessment for {minutes_since_human_contact:.0f} min "
            f"(twice the {recheck_interval_minutes:.0f} min re-check interval)."))

    if minutes_since_any_observation > 10:
        r = by_id["TR_ALL_CHANNELS_SILENT"]
        out.append(TimeRuleHit(
            r["rule_id"], r["name"], r["action"], r["escalates_acuity"],
            f"No channel has produced an observation for {minutes_since_any_observation:.0f} min. "
            "Silence is absence of information, not stability."))

    if consent_declined:
        r = by_id["TR_CONSENT_DECLINED_COMPENSATION"]
        out.append(TimeRuleHit(
            r["rule_id"], r["name"], r["action"], r["escalates_acuity"],
            "Camera and attendant channels declined. Human re-check interval halved to compensate. "
            "Queue position unaffected.", factor=float(r.get("factor", 0.5))))

    return out
