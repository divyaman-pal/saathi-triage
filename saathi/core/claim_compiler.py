"""
The Clinical Claim Compiler.

This is the central defensible innovation. It determines WHAT MAY BE SAID from
WHAT THE EVIDENCE SUPPORTS, before any language is generated.

    Evidence -> quality gating -> risk + confidence -> cost decision
             -> SAFETY CONTRACT           <- this module
             -> narrative plan (per persona)
             -> LLM rendering             <- llm.py, replaceable, non-load-bearing
             -> VALIDATOR                 <- validator.py, rejects out-of-contract claims
             -> output

The Safety Contract is produced entirely by deterministic code. The LLM renders
it. If the LLM is switched off, the contract is byte-identical and the
deterministic template renderer below produces the text instead - which is why
disabling the LLM costs you prose and nothing else.

THE NUMBER WHITELIST
    allowed_numbers is the anti-hallucination mechanism. Every numeric token in
    the rendered text must appear in this list. A model that invents
    "respiratory rate 32" when the contract says 27 is caught deterministically,
    by set membership, without needing a second model to judge the first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .contract_loader import Contract
from .fusion import Discordance, Trajectory
from .models import (
    AbstentionResult,
    Acuity,
    ConfidenceComponents,
    CostDecision,
    EpistemicState,
    EWERComponents,
    Evidence,
    MaterialityFinding,
    OverrideRecord,
    RedFlagHit,
    Role,
    SafetyContract,
    SLAStatus,
    next_id,
)


# ---------------------------------------------------------------------------
# Epistemic state
# ---------------------------------------------------------------------------


def determine_state(
    *,
    red_flags: list[RedFlagHit],
    abstention: AbstentionResult,
    override: OverrideRecord | None,
    materiality: list[MaterialityFinding],
    model_moved: bool,
    discordance: Discordance,
    escalating: bool,
    has_prior_record: bool,
    failing_confidence: list[str],
) -> EpistemicState:
    """
    Precedence is deliberate and is the safety ordering, most-constraining first.

      1. RED_FLAG_FIRED       - a clinician-authored rule fired. Nothing a model
                                says can soften the language.
      2. INSUFFICIENT_SIGNAL  - we cannot see. Admitting that outranks every
                                other thing we might want to say.
      3. HUMAN_OVERRIDE       - a human has decided. The system reports, it does
                                not argue.
      4. MATERIALITY          - escalated on clinical significance without model
                                movement. Must be SAID that way, or the nurse
                                will read a model claim that was never made.
      5. DISCORDANT_CHANNELS  - the disagreement is the finding, and attribution
                                language is forbidden because it would hide it.
      6. OBSERVATION_ONLY     - reduced feature set; association claims barred.
      7/8. escalation, high or moderate confidence.
      9. STABLE_NO_CHANGE     - and even here, reassurance is banned.
    """
    if red_flags:
        return EpistemicState.RED_FLAG_FIRED
    if abstention.abstained:
        return EpistemicState.INSUFFICIENT_SIGNAL
    if override is not None:
        return EpistemicState.HUMAN_OVERRIDE_APPLIED
    if any(m.action == "ESCALATE" for m in materiality) and not model_moved:
        return EpistemicState.MATERIALITY_ESCALATION
    if discordance.magnitude >= 0.35:
        return EpistemicState.DISCORDANT_CHANNELS
    if escalating and not has_prior_record:
        return EpistemicState.OBSERVATION_ONLY
    if escalating:
        return (EpistemicState.MODERATE_CONFIDENCE_ESCALATION if failing_confidence
                else EpistemicState.HIGH_CONFIDENCE_ESCALATION)
    if not has_prior_record:
        return EpistemicState.OBSERVATION_ONLY
    return EpistemicState.STABLE_NO_CHANGE


# ---------------------------------------------------------------------------
# Number and entity whitelists
# ---------------------------------------------------------------------------


def _nums_from(values) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        out.append(round(f, 2))
        out.append(round(f))
    return out


ENTITY_NAMES = {
    "HEART_RATE": "heart rate", "RESP_RATE": "respiratory rate", "SPO2": "oxygen saturation",
    "SBP": "systolic blood pressure", "TEMP": "temperature", "CAP_REFILL": "capillary refill",
    "AVPU": "responsiveness", "ARRIVAL_MODE": "mode of arrival", "POSTURE": "posture",
    "WORK_OF_BREATHING": "work of breathing", "STILLNESS_MINUTES": "movement",
    "SKIN_COLOR_CHANGE": "skin colour", "CONFUSION_NEW": "attendant report of new confusion",
    "SLEEPINESS_INCREASE": "attendant report of increased sleepiness",
    "RESPONDS_TO_NAME": "attendant report of responsiveness",
    "PAIN_SELF_REPORT": "patient-reported pain", "SHOCK_INDEX": "shock index",
    "WAIT_MINUTES": "wait time", "MINUTES_SINCE_HUMAN_CONTACT": "time since last human check",
    "COMORBIDITY_COUNT": "recorded comorbidities", "PRIOR_ED_VISITS_90D": "recent ED visits",
    "PRIOR_ICU_ADMISSION": "prior intensive care admission",
    "ATTENDANT_CONCERN": "attendant report",
}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(
    contract: Contract,
    *,
    patient_id: str,
    persona: Role,
    access_scope: list[str],
    acuity_arrival: Acuity,
    acuity_previous: Acuity | None,
    acuity_current: Acuity,
    change_reason: str,
    state: EpistemicState,
    supporting: list[Evidence],
    contradicting: list[Evidence],
    red_flags: list[RedFlagHit],
    materiality: list[MaterialityFinding],
    confidence: ConfidenceComponents,
    cost_decision: CostDecision | None,
    ewer: EWERComponents | None,
    sla: SLAStatus | None,
    trajectories: list[Trajectory],
    abstention: AbstentionResult,
    missing_data: list[str],
    stale_data: list[str],
    degraded_channels: list[str],
    absent_channels: list[str],
    model_versions: dict[str, str],
    lineage_ref: str,
    override: OverrideRecord | None = None,
    generated_at: datetime | None = None,
) -> SafetyContract:
    spec = contract.epistemic_spec(state.value)
    abstaining = state is EpistemicState.INSUFFICIENT_SIGNAL

    # -- numbers ----------------------------------------------------------
    numbers: list[float] = []
    if abstaining:
        # ABSTENTION PURITY: only the failing quality metrics and the staleness
        # ages. No risk number, no score, no probability may enter the whitelist,
        # so the renderer cannot emit one even if it tries.
        for line in abstention.failing_signals:
            numbers += [round(float(m), 2) for m in re.findall(r"\d+\.?\d*", line)]
        for line in abstention.stale_items:
            numbers += [round(float(m), 2) for m in re.findall(r"\d+\.?\d*", line)]
        if sla is not None:
            numbers += [round(sla.waited_minutes), 0]
    else:
        numbers += _nums_from(e.value for e in supporting + contradicting)
        numbers += _nums_from(t.delta for t in trajectories if t.material)
        numbers += _nums_from(t.minutes for t in trajectories if t.material)
        numbers += [acuity_current.level, acuity_arrival.level]
        if acuity_previous:
            numbers.append(acuity_previous.level)
        if sla is not None:
            numbers += [round(sla.waited_minutes), round(sla.max_wait_minutes),
                        round(sla.recheck_interval_minutes)]
    numbers = sorted({float(n) for n in numbers if n is not None})

    # -- entities ---------------------------------------------------------
    entities = {patient_id}
    for e in supporting + contradicting:
        entities.add(ENTITY_NAMES.get(e.concept_id, e.concept_id.lower().replace("_", " ")))
    for t in trajectories:
        if t.material:
            entities.add(ENTITY_NAMES.get(t.concept_id, t.concept_id.lower().replace("_", " ")))
    for m in materiality:
        if m.action in ("ESCALATE", "RECHECK"):
            entities.add(ENTITY_NAMES.get(m.concept_id, m.concept_id.lower().replace("_", " ")))
    for r in red_flags:
        entities.add(r.name.lower())
    entities |= {"nurse re-check", "clinician review", "wait time", "prior record",
                 "camera", "attendant", "nurse"}

    # -- required disclosures ---------------------------------------------
    disclosures: list[str] = []
    if "prior_record" in missing_data or "COMORBIDITY_COUNT" in missing_data:
        disclosures.append("prior_record_missing")
    for ch in degraded_channels:
        disclosures.append(f"{ch}_degraded")
    for ch in absent_channels:
        disclosures.append(f"{ch}_absent")
    if stale_data:
        disclosures.append("stale_data_present")
    if not confidence.calibration_status.startswith("CALIBRATED"):
        disclosures.append("uncalibrated_subgroup")
    if state is EpistemicState.MATERIALITY_ESCALATION:
        disclosures.append("escalated_on_materiality_not_model")
    if state is EpistemicState.OBSERVATION_ONLY:
        disclosures.append("observation_only_model")
    if abstaining:
        disclosures.append("insufficient_signal")

    sc = SafetyContract(
        contract_id=next_id("SC", patient_id),
        patient_id=patient_id,
        persona=persona,
        access_scope=access_scope,
        acuity_current=acuity_current,
        acuity_previous=acuity_previous,
        acuity_arrival=acuity_arrival,
        change_reason=change_reason,
        escalation_only=override is None,
        epistemic_status=state,
        evidence_ids=[e.evidence_id for e in supporting],
        contradictory_evidence_ids=[e.evidence_id for e in contradicting],
        red_flags_fired=[r.rule_id for r in red_flags],
        materiality_findings=[f"{m.concept_id}:{m.action}" for m in materiality
                              if m.action in ("ESCALATE", "RECHECK")],
        confidence=confidence,
        cost_decision=None if abstaining else cost_decision,
        ewer=ewer,
        sla=sla,
        missing_data=sorted(set(missing_data)),
        stale_data=sorted(set(stale_data)),
        degraded_channels=sorted(set(degraded_channels)),
        absent_channels=sorted(set(absent_channels)),
        allowed_claim_types=list(spec["allowed_claim_types"]),
        forbidden_claim_types=list(spec["forbidden_claim_types"]),
        allowed_numbers=numbers,
        allowed_entities=sorted(entities),
        required_disclosures=sorted(set(disclosures)),
        max_words=int(spec["max_words"]),
        priority_question=abstention.priority_question,
        physical_check_instruction=_physical_check(state, red_flags, supporting, abstention),
        lineage_ref=lineage_ref,
        model_versions=model_versions,
        contract_version=contract.version,
        grammar_version=contract.grammar_version,
        generated_at=generated_at or datetime.now(),
    )
    return sc


def _physical_check(
    state: EpistemicState, red_flags: list[RedFlagHit], supporting: list[Evidence],
    abstention: AbstentionResult,
) -> str | None:
    """
    'What I should physically go and check.' A nurse cannot act on a score; they
    can act on an instruction to go and put a hand on someone.
    """
    if red_flags:
        return f"Go now. {red_flags[0].trigger_human}"
    if abstention.abstained:
        return "Go and take a fresh set of vitals and a direct look. The system cannot see this patient."
    concepts = {e.concept_id for e in supporting}
    if "CONFUSION_NEW" in concepts or "RESPONDS_TO_NAME" in concepts:
        return "Ask the patient their name and today's date. Compare with how the family describes them."
    if "RESP_RATE" in concepts or "WORK_OF_BREATHING" in concepts:
        return "Count the respiratory rate yourself for a full 30 seconds and look for accessory muscle use."
    if "STILLNESS_MINUTES" in concepts or "POSTURE" in concepts:
        return "Rouse the patient and check they can sit up and speak a full sentence."
    if "SPO2" in concepts or "SBP" in concepts:
        return "Repeat the blood pressure and saturation on the other arm."
    return None


# ---------------------------------------------------------------------------
# Narrative plan - what the renderer is asked to say, per persona
# ---------------------------------------------------------------------------


@dataclass
class NarrativePlan:
    persona: Role
    state: EpistemicState
    bullets: list[str] = field(default_factory=list)
    contrast: list[str] = field(default_factory=list)
    action: str = ""
    disclosures: list[str] = field(default_factory=list)
    max_words: int = 90


DISCLOSURE_TEXT = {
    "prior_record_missing": "No prior record available.",
    "camera_degraded": "Camera signal degraded.",
    "attendant_degraded": "Attendant channel degraded.",
    "camera_absent": "No camera at this site.",
    "attendant_absent": "No attendant present.",
    "prior_record_absent": "No linked record at this site.",
    "stale_data_present": "Some values are past their validity window.",
    "uncalibrated_subgroup": "Model is not calibrated for this age subgroup.",
    "escalated_on_materiality_not_model": "Escalated on clinical significance, not on model output.",
    "observation_only_model": "Observation-only model used (reduced feature set).",
    "insufficient_signal": "Insufficient signal to assess.",
}


def plan(
    contract: Contract, sc: SafetyContract, supporting: list[Evidence],
    contradicting: list[Evidence], trajectories: list[Trajectory],
    red_flags: list[RedFlagHit], materiality: list[MaterialityFinding],
    discordance: Discordance,
) -> NarrativePlan:
    p = NarrativePlan(persona=sc.persona, state=sc.epistemic_status, max_words=sc.max_words)

    if sc.epistemic_status is EpistemicState.INSUFFICIENT_SIGNAL:
        p.bullets = ["Cannot produce a reliable estimate for this patient."]
        p.action = "Mandatory nurse re-check. Acuity held. Queue position protected."
        p.disclosures = [DISCLOSURE_TEXT.get(d, d) for d in sc.required_disclosures]
        return p

    if red_flags:
        for r in red_flags[:2]:
            p.bullets.append(f"{r.name}: {r.trigger_human}")
        p.action = f"Immediate clinician review. Acuity {sc.acuity_current.level}."
        return p

    for t in trajectories[:3]:
        if t.material:
            p.bullets.append(t.describe(contract.unit(t.concept_id)))
    for e in supporting[:4]:
        if len(p.bullets) >= 4:
            break
        name = ENTITY_NAMES.get(e.concept_id, e.concept_id.lower().replace("_", " "))
        p.bullets.append(f"{name} {e.value} ({e.source_channel.value}, "
                         f"{e.signal_quality.status.value.lower()}, {e.threshold_band.value.lower()} for age)")
    for m in materiality:
        if m.action == "ESCALATE":
            name = ENTITY_NAMES.get(m.concept_id, m.concept_id)
            p.bullets.append(f"{name} - weak statistical evidence, {m.materiality_class.value.lower()} clinical significance")
            break

    for e in contradicting[:2]:
        name = ENTITY_NAMES.get(e.concept_id, e.concept_id.lower().replace("_", " "))
        p.contrast.append(f"{name} {e.value} is within age-band limits ({e.source_channel.value})")
    if discordance.de_escalating > 0.05:
        p.contrast.append("attendant reports the patient is at their usual baseline")

    if sc.sla is not None:
        p.action = (f"Nurse re-check in {sc.sla.recheck_interval_minutes:.0f} min."
                    if sc.sla.recheck_interval_minutes > 0 else "Nurse re-check now.")
    p.disclosures = [DISCLOSURE_TEXT.get(d, d) for d in sc.required_disclosures]
    return p


# ---------------------------------------------------------------------------
# Deterministic template renderer
# ---------------------------------------------------------------------------


def render_deterministic(sc: SafetyContract, p: NarrativePlan) -> str:
    """
    A pure function of the Safety Contract. Cannot violate the grammar by
    construction - it only emits vocabulary the grammar permits.

    This is what runs when the LLM is off, when the LLM times out, and when the
    validator rejects the LLM's output twice. It is the floor beneath the prose,
    and the reason the LLM is never load-bearing for safety.
    """
    if sc.persona is Role.ATTENDANT:
        return _render_attendant(sc)

    parts: list[str] = []
    if sc.epistemic_status is EpistemicState.INSUFFICIENT_SIGNAL:
        parts.append("INSUFFICIENT SIGNAL. Cannot produce a reliable estimate for this patient.")
        if p.disclosures:
            parts.append(" ".join(p.disclosures))
        parts.append(p.action)
        if sc.priority_question:
            parts.append(f"Priority question for the nurse: {sc.priority_question}")
        return " ".join(parts)

    if sc.epistemic_status is EpistemicState.HUMAN_OVERRIDE_APPLIED:
        return (f"Acuity set to {sc.acuity_current.level} by a clinician. "
                f"System recommendation was {sc.acuity_previous.level if sc.acuity_previous else sc.acuity_arrival.level} "
                f"and is retained in the record.")

    if sc.acuity_current.level < sc.acuity_arrival.level:
        parts.append(f"Escalated {sc.acuity_arrival.level} to {sc.acuity_current.level}.")
    else:
        parts.append(f"Acuity {sc.acuity_current.level}, unchanged from arrival.")

    if p.bullets:
        parts.append("Observed: " + "; ".join(p.bullets[:4]) + ".")
    if p.contrast:
        parts.append("Arguing against: " + "; ".join(p.contrast[:2]) + ".")
    if p.disclosures:
        parts.append(" ".join(p.disclosures))
    if p.action:
        parts.append(p.action)
    return " ".join(parts)


def _render_attendant(sc: SafetyContract) -> str:
    """
    Allow-list surface. No acuity, no score, no queue position, no risk word, no
    reassurance. A concrete observable task and an honest receipt.
    """
    if sc.priority_question:
        return ("Thank you. Your report has reached the nurse. "
                "Please stay with the patient. A nurse has been asked to check.")
    return ("Thank you. Your report has reached the nurse at the desk. "
            "Please stay with the patient and tell us if anything changes.")
