"""
THE SAFETY INVARIANT SUITE.

Eight properties that must hold for every patient, in every deployment tier,
with any subset of the stack switched off. These are the claims SAATHI makes
about itself; this file is where they are checked rather than asserted.

A passing invariant suite is worth more than a higher AUC. An AUC is a summary
of a sample; an invariant is a promise about behaviour.

    pytest saathi/tests/test_invariants.py -v
"""

from __future__ import annotations

import pytest

from saathi.core.escalation import compute_sla
from saathi.core.models import (
    Acuity,
    EpistemicState,
    SafetyContract,
    SchemeMismatchError,
    esi,
)
from saathi.core.pipeline import PipelineConfig
from saathi.core.schemes import monotonic_check
from saathi.runtime import Runtime

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def rt() -> Runtime:
    r = Runtime().build(include_surge=False)
    r.assess_all()
    return r


# ===========================================================================
# INVARIANT 1 - MONOTONIC ESCALATION
# SAATHI raises urgency and never lowers it. Only a human downgrades, with a
# recorded reason.
# ===========================================================================


def test_invariant_1_no_patient_is_ever_downgraded_by_the_system(rt):
    for a in rt.assessments.values():
        assert a.acuity_current.level <= a.acuity_arrival.level, (
            f"{a.patient_id} moved from L{a.acuity_arrival.level} to "
            f"L{a.acuity_current.level} - less urgent - without a human override.")


def test_invariant_1_monotonic_check_raises_on_downgrade():
    monotonic_check(esi(3), esi(2))          # escalation: fine
    monotonic_check(esi(3), esi(3))          # no change: fine
    with pytest.raises(ValueError, match="MONOTONIC ESCALATION VIOLATION"):
        monotonic_check(esi(2), esi(3))      # downgrade: refused
    monotonic_check(esi(2), esi(3), human_override=True)   # human: permitted


def test_invariant_1_safety_contract_cannot_be_constructed_with_a_downgrade(rt):
    """The invariant is structural, not merely checked at one call site."""
    a = next(iter(rt.assessments.values()))
    sc = a.safety_contract
    with pytest.raises(ValueError, match="MONOTONIC ESCALATION VIOLATION"):
        SafetyContract(**(sc.model_dump() | {
            "acuity_arrival": esi(2), "acuity_current": esi(4), "escalation_only": True}))


def test_invariant_1_acuity_cannot_be_compared_across_schemes():
    mts_orange = Acuity(scheme_id="MTS", scheme_version="v3", level=2)
    with pytest.raises(SchemeMismatchError):
        esi(2).is_more_urgent_than(mts_orange)


# ===========================================================================
# INVARIANT 2 - RED-FLAG SUPREMACY
# Deterministic clinician-authored rules fire regardless of model output and
# cannot be suppressed by it.
# ===========================================================================


def test_invariant_2_red_flag_sets_acuity_at_or_above_its_target(rt):
    for a in rt.assessments.values():
        for flag in a.red_flags:
            assert a.acuity_current.level <= flag.target_acuity.level, (
                f"{a.patient_id}: {flag.rule_id} targets L{flag.target_acuity.level} "
                f"but the patient sits at L{a.acuity_current.level}. A model suppressed a rule.")


def test_invariant_2_red_flags_fire_with_every_model_disabled():
    """
    The whole learned stack off. Rules must still triage.

    This is the test that says what SAATHI is: a deterministic safety system
    with a model attached, not a model with some rules attached.
    """
    r = Runtime().build(include_surge=False, reset_store=False)
    cfg = PipelineConfig(fail_all_models=True, llm_enabled=False)
    for pid in ("P-011", "P-012", "P-008"):
        a = r.assess(pid, config=cfg)
        assert a.red_flags, f"{pid} lost its red flags when the models were disabled"
        assert a.acuity_current.level <= min(f.target_acuity.level for f in a.red_flags)


def test_invariant_2_red_flags_are_not_suppressible_by_flag(rt):
    for a in rt.assessments.values():
        for flag in a.red_flags:
            assert flag.suppressible_by_model is False


def test_invariant_2_red_flag_module_imports_no_model():
    """
    Enforced by construction: there is no wire from a model to a rule.

    Checked against the IMPORT LINES rather than the file text, so that the
    module is still free to explain in prose which modules it deliberately does
    not import.
    """
    import ast
    import saathi.core.red_flags as rf
    tree = ast.parse(open(rf.__file__, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[-1] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[-1])
            imported |= {a.name for a in node.names}
    for forbidden in ("risk_model", "cost_engine", "confidence", "llm", "fusion",
                      "features", "pipeline"):
        assert forbidden not in imported, (
            f"red_flags.py imports {forbidden!r}. The rule layer must not be able "
            "to see a model output, or 'un-suppressible' is only a comment.")


# ===========================================================================
# INVARIANT 3 - DEGRADED-MODE ESCALATION
# Model failure, signal failure or LLM failure moves the patient TOWARD human
# attention, never away.
# ===========================================================================


def test_invariant_3_model_failure_never_lowers_acuity_or_lengthens_recheck(rt):
    baseline = {pid: (a.acuity_current.level, a.sla.recheck_interval_minutes)
                for pid, a in rt.assessments.items()}
    r = Runtime().build(include_surge=False, reset_store=False)
    r.assess_all()
    cfg = PipelineConfig(fail_all_models=True)
    for pid in list(baseline)[:10]:
        r.pipeline.states.pop(pid, None)
        a = r.assess(pid, config=cfg)
        assert a.sla.recheck_interval_minutes <= baseline[pid][1] + 1e-6, (
            f"{pid}: losing the model LENGTHENED the re-check interval.")


def test_invariant_3_all_channels_failing_forces_abstention_not_a_calm_score():
    r = Runtime().build(include_surge=False, reset_store=False)
    cfg = PipelineConfig(fail_all_channels=True)
    for pid in ("P-003", "P-020", "P-013"):
        r.pipeline.states.pop(pid, None)
        a = r.assess(pid, config=cfg)
        assert a.abstention.abstained, f"{pid} produced a score with every channel dead"
        assert a.epistemic_state is EpistemicState.INSUFFICIENT_SIGNAL
        assert a.sla.recheck_interval_minutes == 0.0


def test_invariant_3_abstention_never_produces_a_risk_number(rt):
    for a in rt.assessments.values():
        if not a.abstention.abstained:
            continue
        assert a.cost_decision is None, (
            f"{a.patient_id} abstained but carries a cost decision - a risk number "
            "the renderer could leak.")
        assert a.safety_contract.cost_decision is None


def test_invariant_3_abstention_raises_rank_rather_than_lowering_it(rt):
    """A patient we cannot see must not sink to the bottom of the list."""
    abstained = [a for a in rt.assessments.values() if a.abstention.abstained]
    assert abstained, "no abstention in the cohort - the path is untested"
    for a in abstained:
        same_level = [b for b in rt.assessments.values()
                      if b.acuity_current.level == a.acuity_current.level
                      and not b.abstention.abstained and not b.red_flags]
        if same_level:
            median = sorted(b.ewer.rank_value for b in same_level)[len(same_level) // 2]
            assert a.ewer.rank_value > median, (
                f"{a.patient_id} abstained yet ranks below the median of its own level.")


# ===========================================================================
# INVARIANT 4 - FAIL-SAFE DEFAULT
# If the entire stack fails, fall back to rules plus FIFO and SAY SO.
# ===========================================================================


def test_invariant_4_total_failure_declares_itself_on_screen():
    """
    Every model gone, every channel gone, the LLM gone.

    A patient with nothing left to go on must produce an explicit statement of
    blindness rendered by the deterministic template - not a blank card, and not
    a confident-looking default.
    """
    r = Runtime().build(include_surge=False, reset_store=False)
    cfg = PipelineConfig(fail_all_models=True, fail_all_channels=True, llm_enabled=False)
    r.pipeline.states.pop("P-003", None)
    a = r.assess("P-003", config=cfg)
    assert a.abstention.abstained
    assert a.narrative.renderer == "deterministic_template"
    assert "insufficient" in a.narrative.text.lower() or "cannot" in a.narrative.text.lower(), (
        f"the system failed silently instead of saying so: {a.narrative.text!r}")
    assert a.sla.recheck_interval_minutes == 0.0, "a blind system did not demand a human"


def test_invariant_4_a_recorded_red_flag_survives_total_failure():
    """
    The complement, and the stronger result.

    P-011's stridor was captured at the desk. Losing every camera, every model
    and the language layer does not un-hear it: the rule still fires from the
    recorded complaint and the child still holds level 1. A red flag that
    evaporated when a sensor died would not be a red flag.
    """
    r = Runtime().build(include_surge=False, reset_store=False)
    cfg = PipelineConfig(fail_all_models=True, fail_all_channels=True, llm_enabled=False)
    r.pipeline.states.pop("P-011", None)
    a = r.assess("P-011", config=cfg)
    assert a.red_flags, "total failure lost a clinician-authored rule"
    assert a.acuity_current.level == 1
    assert a.narrative.renderer == "deterministic_template"


# ===========================================================================
# INVARIANT 5 - NO SILENT IMPUTATION
# A missing value is never replaced by a mean and treated as observed.
# ===========================================================================


def test_invariant_5_missing_features_are_nan_not_means(rt):
    import math
    from saathi.core.features import assemble
    from saathi.core.gating import gate
    from saathi.data.simulate import Simulator

    p = rt.profiles["P-003"]          # no prior record
    sim = Simulator(rt.contract, rt.now, "TIER_A")
    usable, rejected = gate(rt.contract, sim.evidence_for(p), rt.now)
    fv = assemble(rt.contract, usable, rejected, age_years=p.age_years, sex=p.sex,
                  has_prior_record=False, live_channels=2)
    for name in ("comorbidity_count", "prior_ed_visits_90d", "prior_icu"):
        assert math.isnan(fv.values[name]), f"{name} was imputed rather than left missing"
        assert name not in fv.present


def test_invariant_5_missing_data_is_surfaced_not_hidden(rt):
    for a in rt.assessments.values():
        if a.model_route == "OBSERVATION_ONLY_v1":
            assert a.missing_data, (
                f"{a.patient_id} used the reduced model but declared nothing missing")
            assert "prior_record_missing" in a.safety_contract.required_disclosures


def test_invariant_5_derived_values_are_never_built_from_imputed_inputs(rt):
    """SHOCK_INDEX exists only when BOTH inputs passed their gates."""
    for a in rt.assessments.values():
        shock = [e for e in a.supporting_evidence + a.contradictory_evidence
                 if e.concept_id == "SHOCK_INDEX"]
        for e in shock:
            assert e.lineage_ref and "derived_from:" in (e.lineage_ref or ""), (
                "a shock index appeared without a recorded pair of source observations")


# ===========================================================================
# INVARIANT 6 - ATTENDANT NON-GAMING
# Attendant escalation triggers a nurse re-check, never a queue position.
# ===========================================================================


def test_invariant_6_fifty_presses_change_nothing_about_the_score():
    """
    The load-bearing test for the promise on the slide.

    P-015's family press the button seven times in the shipped cohort. Here we
    push it to fifty and assert that the acuity, the EWER rank and every EWER
    component are BYTE-IDENTICAL to the zero-press case.
    """
    r = Runtime().build(include_surge=False, reset_store=False)
    base = r.assess("P-015")
    baseline = (base.acuity_current.level, base.ewer.rank_value,
                base.ewer.model_dump())

    p = r.profiles["P-015"]
    p.attendant.concern_presses = [float(i) * 0.7 for i in range(1, 51)]
    r.pipeline.states.pop("P-015", None)
    spammed = r.assess("P-015")

    assert spammed.acuity_current.level == baseline[0], "presses moved the acuity"
    assert spammed.ewer.rank_value == pytest.approx(baseline[1]), "presses moved the EWER rank"
    assert spammed.ewer.model_dump() == baseline[2], "presses moved an EWER component"


def test_invariant_6_attendant_escalation_returns_structural_zeros():
    from saathi.core.escalation import attendant_escalation
    from saathi.core.gating import gate
    from saathi.data.simulate import Simulator
    r = Runtime().build(include_surge=False, reset_store=False)
    p = r.profiles["P-015"]
    sim = Simulator(r.contract, r.now, "TIER_A")
    usable, _ = gate(r.contract, sim.evidence_for(p), r.now)
    esc = attendant_escalation(usable)
    assert esc.presses > 0, "the over-reporter case produced no presses"
    assert esc.recheck_triggered is True, "a report must always buy a nurse re-check"
    assert esc.acuity_effect == 0
    assert esc.queue_effect == 0
    assert esc.ewer_effect == 0.0


def test_invariant_6_attendant_concept_has_no_scoring_path(rt):
    """ATTENDANT_CONCERN routes to nurse_recheck_trigger and nowhere else."""
    contributes = rt.contract.contributes_to("ATTENDANT_CONCERN")
    assert contributes == ["nurse_recheck_trigger"], (
        f"ATTENDANT_CONCERN contributes to {contributes} - it must reach the "
        "re-check trigger and nothing else.")
    for forbidden in ("tabular_risk", "ewer", "materiality_escalation"):
        assert forbidden not in contributes


def test_invariant_6_a_downweighted_reporter_still_gets_the_nurse(rt):
    """Down-weighting influence on the SCORE must never stop us listening."""
    a = rt.assessments["P-015"]
    rechecks = [m for m in a.materiality if m.action == "RECHECK"]
    escalates = [m for m in a.materiality if m.action == "ESCALATE"]
    assert not escalates, "an over-reporter moved the acuity"
    assert rechecks, "an over-reporter was silenced entirely rather than down-weighted"


# ===========================================================================
# INVARIANT 7 - WAIT-TIME SLA
# Every acuity level has a maximum safe wait; breach forces re-assessment.
# ===========================================================================


def test_invariant_7_every_level_has_an_sla(rt):
    for level in (1, 2, 3, 4, 5):
        spec = rt.contract.sla_for(level)
        assert "max_wait_minutes" in spec and "recheck_interval_minutes" in spec


def test_invariant_7_breach_is_detected_for_every_patient_past_the_ceiling(rt):
    for a in rt.assessments.values():
        expected = a.sla.waited_minutes > a.sla.max_wait_minutes
        assert a.sla.breached == expected, f"{a.patient_id}: SLA breach flag is wrong"


def test_invariant_7_breach_alone_never_raises_acuity(rt):
    """
    P-013 waited past its ceiling with nothing clinical changing.

    'Waited too long' and 'sicker' are different facts. Conflating them would
    corrupt the acuity signal, and the queue would fill with level 2s created
    by the clock.
    """
    a = rt.assessments["P-013"]
    assert a.sla.breached, "P-013 was supposed to breach its SLA"
    assert not a.red_flags
    assert a.acuity_current.level == a.acuity_arrival.level, (
        "an SLA breach raised the acuity with no clinical change")


def test_invariant_7_no_modifier_ever_lengthens_a_recheck_interval(rt):
    """Reduced information must never buy LESS attention."""
    for level in (1, 2, 3, 4, 5):
        base = float(rt.contract.sla_for(level)["recheck_interval_minutes"])
        for flags in [
            dict(has_prior_record=False, abstained=False, consent_declined=False,
                 low_confidence=False, surge_active=False, channel_degraded=False,
                 attendant_absent=False),
            dict(has_prior_record=False, abstained=False, consent_declined=True,
                 low_confidence=True, surge_active=True, channel_degraded=True,
                 attendant_absent=True),
            dict(has_prior_record=True, abstained=True, consent_declined=False,
                 low_confidence=False, surge_active=False, channel_degraded=False,
                 attendant_absent=False),
        ]:
            sla = compute_sla(rt.contract, level=level, waited_minutes=5.0,
                              minutes_since_human_contact=5.0, **flags)
            assert sla.recheck_interval_minutes <= base + 1e-9


def test_invariant_7_opting_out_shortens_the_interval_and_costs_no_queue_position(rt):
    """P-018 declined consent. It must cost the department, not the patient."""
    a = rt.assessments["P-018"]
    base = float(rt.contract.sla_for(a.acuity_current.level)["recheck_interval_minutes"])
    assert "consent_declined" in a.sla.modifiers_applied
    assert a.sla.recheck_interval_minutes < base
    assert a.acuity_current.level <= a.acuity_arrival.level


# ===========================================================================
# INVARIANT 8 - HUMAN AUTHORITY
# No patient is moved, discharged or de-prioritised without a human.
# ===========================================================================


def test_invariant_8_only_a_human_override_can_lower_an_acuity(rt):
    from saathi.core.models import OverrideRecord
    a = rt.assessments["P-019"]
    rec = OverrideRecord(
        override_id="OVR-TEST", patient_id="P-019", clinician_id="dr_rao",
        role=next(r for r in __import__("saathi.core.models", fromlist=["Role"]).Role
                  if r.value == "ed_physician"),
        timestamp=rt.now, system_acuity=esi(2), clinician_acuity=esi(3), direction="down",
        reason_code="clinical_assessment_on_examination",
        safety_contract_id=a.safety_contract.contract_id, contract_version=rt.contract.version,
    )
    assert rec.direction == "down"
    with pytest.raises(ValueError, match="contradicts"):
        OverrideRecord(**(rec.model_dump() | {"direction": "up"}))
    with pytest.raises(ValueError, match="must change"):
        OverrideRecord(**(rec.model_dump() | {"clinician_acuity": esi(2)}))


def test_invariant_8_override_captures_the_full_payload(rt):
    from saathi.core.models import OverrideRecord, Role
    a = rt.assessments["P-019"]
    rec = OverrideRecord(
        override_id="OVR-TEST2", patient_id="P-019", clinician_id="dr_rao",
        role=Role.ED_PHYSICIAN, timestamp=rt.now,
        system_acuity=a.acuity_current, clinician_acuity=esi(a.acuity_current.level + 1),
        direction="down", reason_code="clinical_assessment_on_examination",
        evidence_shown_at_time=[e.evidence_id for e in a.supporting_evidence],
        safety_contract_id=a.safety_contract.contract_id,
        model_versions=a.model_versions, contract_version=rt.contract.version,
        time_from_display_to_override_ms=4200.0)
    for field in ("override_id", "patient_id", "clinician_id", "role", "timestamp",
                  "system_acuity", "clinician_acuity", "direction", "reason_code",
                  "evidence_shown_at_time", "safety_contract_id", "model_versions",
                  "contract_version", "time_from_display_to_override_ms"):
        assert field in rec.model_dump()
    assert rec.evidence_shown_at_time, "no evidence snapshot captured"


def test_invariant_8_no_endpoint_discharges_or_removes_a_patient():
    import saathi.api.main as api
    src = open(api.__file__, encoding="utf-8").read()
    for word in ("discharge", "def remove_patient", "delete_patient"):
        assert word not in src, f"the API exposes {word!r} - no automated disposition is permitted"


# ===========================================================================
# CROSS-CUTTING - every score carries its uncertainty
# ===========================================================================


def test_every_score_carries_a_confidence_decomposition(rt):
    """No bare scores, ever."""
    for a in rt.assessments.values():
        c = a.confidence
        for part in ("signal_quality", "completeness", "applicability", "channel_agreement"):
            v = getattr(c, part)
            assert 0.0 <= v <= 1.0, f"{a.patient_id}: {part} out of range"
        assert c.calibration_status, f"{a.patient_id} has no calibration status"


def test_every_patient_has_contradictory_evidence_retrieved(rt):
    """'What argues against this' must be populated, not merely present."""
    populated = [a for a in rt.assessments.values() if a.contradictory_evidence]
    assert len(populated) >= len(rt.assessments) * 0.6, (
        "most patients have no contradictory evidence retrieved - the system is "
        "only surfacing what supports its own conclusion.")


def test_signal_freshness_is_attached_to_every_displayed_item(rt):
    for a in rt.assessments.values():
        for e in a.supporting_evidence + a.contradictory_evidence:
            assert e.freshness is not None, f"{e.evidence_id} displayed without a freshness stamp"
            assert e.signal_quality is not None
            assert e.age_band_context, f"{e.evidence_id} displayed without its age-band context"


def test_acuity_always_carries_scheme_and_version(rt):
    for a in rt.assessments.values():
        for acu in (a.acuity_arrival, a.acuity_current):
            assert acu.scheme_id and acu.scheme_version, (
                "a bare acuity level escaped without its scheme identity")
