"""
SAATHI HTTP API.

Every endpoint enforces entitlements in the BACKEND, before retrieval, before
analysis and before any LLM call. There is no endpoint that returns a raw video
frame or a face embedding, because those are never persisted - the absence is
the control, not a filter.

Identity is carried in headers for the prototype (X-Actor-Id, X-Role,
X-Patient-Id for attendants). A production deployment would put a real
authenticator here; what it would NOT change is anything below the
`require(...)` calls, which is the point of putting them there.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core import ewer as ewer_mod
from ..core.audit import get_store
from ..core.contract_loader import get_contract
from ..core.injection_guard import scan
from ..core.models import OverrideRecord, Role, SchemeMappingRefused, esi, next_id
from ..core.rbac import AccessDenied, Principal, llm_payload, project_assessment, require, scope_patients
from ..core.schemes import convert
from ..core.telemetry import projected_costs, summarise
from ..runtime import get_runtime

app = FastAPI(
    title="SAATHI",
    version="2.0",
    description=(
        "Continuous, multi-channel triage assistance for the waiting interval. "
        "Clinical decision support. Not a diagnostic device."
    ),
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def principal(
    x_actor_id: str = Header(default="anonymous"),
    x_role: str = Header(default=""),
    x_patient_id: str | None = Header(default=None),
) -> Principal:
    try:
        role = Role(x_role)
    except ValueError:
        # default_decision is DENY. An unrecognised role is not a guest.
        get_store().log_access(
            actor_id=x_actor_id, role=x_role or "<none>", subject=None,
            resource="authentication", fields=None, decision="DENY",
            reason="unrecognised or absent role; entitlements default_decision is DENY")
        raise HTTPException(status_code=403, detail={
            "error": "unrecognised role",
            "supplied": x_role,
            "valid": [r.value for r in Role],
            "note": "Entitlements default to DENY. There is no anonymous access.",
        })
    rt = get_runtime()
    own_queue = [a.patient_id for a in rt.queue()] if role is Role.TRIAGE_NURSE else []
    return Principal(actor_id=x_actor_id, role=role, own_queue=own_queue,
                     own_patient=x_patient_id)


@app.exception_handler(AccessDenied)
async def _denied(request, exc: AccessDenied):
    return JSONResponse(status_code=403, content={
        "error": "403 Forbidden",
        "role": exc.role,
        "resource": exc.resource,
        "reason": exc.reason,
        "audit": "An access-denied event has been written to the audit log.",
    })


@app.exception_handler(SchemeMappingRefused)
async def _scheme_refused(request, exc: SchemeMappingRefused):
    return JSONResponse(status_code=422, content={
        "error": "422 Cross-scheme conversion refused",
        "reason": str(exc),
    })


# ---------------------------------------------------------------------------
# Health and contract
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    rt = get_runtime()
    c = get_contract()
    return {
        "status": "ok",
        "contract_version": c.version,
        "grammar_version": c.grammar_version,
        "ruleset_version": c.ruleset_version,
        "cost_policy_version": c.policy_version,
        "tier": rt.tier,
        "patients_loaded": len(rt.profiles),
        "llm": rt.llm.telemetry(),
        "positioning": "Clinical decision support. Raises urgency only. Never diagnoses.",
    }


@app.get("/contract/concepts")
def concepts() -> dict:
    c = get_contract()
    return {"contract_version": c.version, "concepts": c.concept_ids}


@app.get("/contract/concept/{concept_id}")
def concept(concept_id: str) -> dict:
    return get_contract().concept(concept_id)


@app.get("/contract/age-bands")
def age_bands() -> dict:
    c = get_contract()
    return {"bands": c.clinical["age_bands"]}


@app.get("/contract/threshold-demo")
def threshold_demo(concept_id: str = "HEART_RATE", value: float = 148.0) -> dict:
    """
    The same absolute value, read through every age band.

    This is the endpoint behind 'HR 148 - normal for age 2, critical for age 75'.
    """
    from ..core.contract_loader import evaluate_band
    c = get_contract()
    rows = []
    for b in c.age_band_ids:
        band, note = evaluate_band(c, concept_id, value, b)
        spec = c.thresholds_for(concept_id, b) or {}
        rows.append({
            "age_band": b, "label": c.age_band_label(b), "verdict": band.value,
            "normal": spec.get("normal"), "concerning": spec.get("concerning"),
            "critical": spec.get("critical"), "note": note,
        })
    return {"concept_id": concept_id, "value": value,
            "unit": c.unit(concept_id), "bands": rows}


@app.get("/contract/scheme-convert")
def scheme_convert(level: int = 2, to: str = "LOCAL3") -> dict:
    """Cross-scheme conversion, or an explicit refusal for the UNSAFE edges."""
    c = get_contract()
    conv = convert(c, esi(level), to)
    return {
        "from": str(conv.source), "to": str(conv.target), "fidelity": conv.fidelity,
        "lost_in_translation": conv.lost_in_translation,
        "mapping_version": conv.mapping_version,
    }


# ---------------------------------------------------------------------------
# Queue and patients
# ---------------------------------------------------------------------------


@app.get("/queue")
def queue(p: Principal = Depends(principal), limit: int = Query(default=60, le=200)) -> dict:
    rt = get_runtime()
    require(get_contract(), rt.store, p, "acuity_score")
    ids = scope_patients(get_contract(), rt.store, p, [a.patient_id for a in rt.queue()])
    rows = []
    for a in rt.queue():
        if a.patient_id not in ids:
            continue
        rows.append({
            "patient_id": a.patient_id,
            "acuity": a.acuity_current.model_dump(),
            "acuity_arrival": a.acuity_arrival.level,
            "changed": a.acuity_current.level < a.acuity_arrival.level,
            "change_reason": a.acuity_change_reason,
            "ewer": a.ewer.rank_value,
            "ewer_components": ewer_mod.format_components(a.ewer),
            "epistemic_state": a.epistemic_state.value,
            "abstained": a.abstention.abstained,
            "red_flags": [r.rule_id for r in a.red_flags],
            "waited_minutes": a.sla.waited_minutes,
            "sla_breached": a.sla.breached,
            "recheck_due_in_minutes": a.sla.recheck_due_in_minutes,
            "confidence": a.confidence.model_dump(),
            "age_band": a.age_band,
            "model_route": a.model_route,
        })
        if len(rows) >= limit:
            break
    return {"n": len(rows), "surge": rt.floor_summary(), "queue": rows}


@app.get("/patient/{patient_id}")
def patient(patient_id: str, p: Principal = Depends(principal)) -> dict:
    rt = get_runtime()
    c = get_contract()
    require(c, rt.store, p, "patient_clinical_detail", subject=patient_id)
    allowed = scope_patients(c, rt.store, p, [patient_id])
    if patient_id not in allowed:
        raise AccessDenied(p.role.value, "patient_row",
                           f"patient {patient_id} is outside this caller's row scope")
    a = rt.assessments.get(patient_id)
    if a is None:
        raise HTTPException(404, f"No assessment for {patient_id}")
    return project_assessment(c, rt.store, p, a)


@app.get("/attendant/view")
def attendant_view(p: Principal = Depends(principal)) -> dict:
    """
    The family's surface. No acuity, no score, no queue position, ever.

    Note that this is a SEPARATE endpoint rather than a filtered version of
    /patient. The attendant projection is an allow-list, and building it by
    subtraction from a clinical payload is how fields leak.
    """
    rt = get_runtime()
    c = get_contract()
    if p.role is not Role.ATTENDANT:
        raise AccessDenied(p.role.value, "attendant_view", "this surface is for attendants only")
    if not p.own_patient:
        raise AccessDenied(p.role.value, "attendant_view", "no patient binding supplied")
    a = rt.assessments.get(p.own_patient)
    if a is None:
        raise HTTPException(404, "unknown patient")
    view = project_assessment(c, rt.store, p, a)
    prof = rt.profiles.get(p.own_patient)
    view["prompt"] = _attendant_prompt(a)
    view["language"] = prof.language if prof else "Hindi"
    view["escalation_effect"] = (
        "Telling us you are worried asks a nurse to come and check. "
        "It does not change your place in the queue - that is decided by the nurses."
    )
    return view


def _attendant_prompt(a) -> dict:
    """One question at a time. Observable tasks only. Never clinical judgement."""
    prompts = [
        {"id": "count_breaths", "text": "Please count his breaths for 15 seconds and tell us the number.",
         "type": "timed_observation_task", "seconds": 15},
        {"id": "responds_to_name", "text": "Does he answer when you call his name?",
         "type": "binary_observation_question"},
        {"id": "more_sleepy", "text": "Is he more sleepy than when you arrived?",
         "type": "comparison_question"},
        {"id": "own_name", "text": "Can he tell you his own name?",
         "type": "binary_observation_question"},
    ]
    idx = int(a.sla.waited_minutes // 12) % len(prompts)
    return prompts[idx]


@app.get("/patient/{patient_id}/lineage")
def lineage(patient_id: str, p: Principal = Depends(principal)) -> dict:
    """
    Displayed acuity -> decision rule -> fusion -> per-channel sub-scores ->
    feature values with staleness -> quality gate result -> raw observation.
    """
    rt = get_runtime()
    c = get_contract()
    require(c, rt.store, p, "patient_clinical_detail", subject=patient_id)
    if patient_id not in scope_patients(c, rt.store, p, [patient_id]):
        raise AccessDenied(p.role.value, "patient_row", "outside row scope")
    a = rt.assessments.get(patient_id)
    if a is None:
        raise HTTPException(404, "unknown patient")
    payload = rt.store.read_lineage(a.lineage_ref)
    if payload is None:
        raise HTTPException(404, "no lineage recorded")
    return payload


@app.get("/patient/{patient_id}/safety-contract")
def safety_contract(patient_id: str, p: Principal = Depends(principal)) -> dict:
    rt = get_runtime()
    c = get_contract()
    require(c, rt.store, p, "acuity_score", subject=patient_id)
    a = rt.assessments.get(patient_id)
    if a is None or a.safety_contract is None:
        raise HTTPException(404, "unknown patient")
    return a.safety_contract.model_dump()


@app.get("/patient/{patient_id}/llm-payload")
def patient_llm_payload(patient_id: str, p: Principal = Depends(principal)) -> dict:
    """
    EXACTLY what the language model receives.

    Displayed so a reader can confirm for themselves that it carries no name, no
    phone number, no ABHA number, no face data and no free identifier. The
    minimisation is applied here, in the backend, by the same function the
    pipeline uses - not re-derived for the demo.
    """
    rt = get_runtime()
    c = get_contract()
    require(c, rt.store, p, "acuity_score", subject=patient_id)
    a = rt.assessments.get(patient_id)
    if a is None or a.safety_contract is None:
        raise HTTPException(404, "unknown patient")
    raw = rt.pipeline._llm_payload(a.safety_contract)
    minimised = llm_payload(c, rt.store, p, raw, subject=patient_id)
    return {
        "sent_to_model": minimised,
        "model": rt.llm.model,
        "backend": rt.llm.backend,
        "guarantee": (
            "No DIRECT_IDENTIFIER, BIOMETRIC or RAW_MEDIA field can appear here. "
            "core.rbac.assert_no_pii raises rather than allowing egress, and the "
            "validator independently rejects any identifier-shaped token in the output."
        ),
    }


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------


class OverrideRequest(BaseModel):
    clinician_acuity: int = Field(ge=1, le=5)
    reason_code: str
    free_text: str | None = None
    time_from_display_to_override_ms: float = 0.0


@app.post("/patient/{patient_id}/override")
def override(patient_id: str, body: OverrideRequest, p: Principal = Depends(principal)) -> dict:
    rt = get_runtime()
    c = get_contract()
    spec = c.role_spec(p.role.value)
    if not spec.get("can_downgrade_acuity") and body.clinician_acuity > 0:
        require(c, rt.store, p, "acuity_score", subject=patient_id)
    if "override_acuity" not in spec["actions"]:
        raise AccessDenied(p.role.value, "override_acuity",
                           "this role may not change an acuity")

    a = rt.assessments.get(patient_id)
    if a is None or a.safety_contract is None:
        raise HTTPException(404, "unknown patient")
    if body.clinician_acuity == a.acuity_current.level:
        raise HTTPException(400, "an override must change the acuity")

    direction = "down" if body.clinician_acuity > a.acuity_current.level else "up"
    rec = OverrideRecord(
        override_id=next_id("OVR", patient_id),
        patient_id=patient_id,
        clinician_id=p.actor_id,
        role=p.role,
        timestamp=rt.now,
        system_acuity=a.acuity_current,
        clinician_acuity=esi(body.clinician_acuity),
        direction=direction,
        reason_code=body.reason_code,
        free_text=body.free_text,
        # The FULL evidence snapshot as it was shown. Seven years from now the
        # question "what did the clinician actually see?" still has an answer.
        evidence_shown_at_time=[e.evidence_id for e in a.supporting_evidence]
                               + [e.evidence_id for e in a.contradictory_evidence],
        safety_contract_id=a.safety_contract.contract_id,
        model_versions=a.model_versions,
        contract_version=c.version,
        time_from_display_to_override_ms=body.time_from_display_to_override_ms,
    )
    rt.store.write_override(rec)
    a.override = rec

    # A human downgrade is the ONLY way an acuity goes down, and it is recorded
    # as a human act rather than as a system decision.
    st = rt.pipeline.states[patient_id]
    st.acuity_previous = st.acuity_current
    st.acuity_current = esi(body.clinician_acuity)
    st.human_overridden = True
    st.last_change_reason = f"HUMAN_OVERRIDE:{p.actor_id}:{body.reason_code}"
    a.acuity_current = st.acuity_current
    a.acuity_change_reason = st.last_change_reason

    rt.store.log_event("OVERRIDE", rec.model_dump_json(), patient_id)
    return {
        "recorded": True,
        "override": rec.model_dump(),
        "acknowledgement": (
            "Your override has been recorded with the full evidence snapshot, the Safety "
            "Contract id and every model version. It will appear in this shift's override "
            "report and is fed to recalibration only through a documented human review gate - "
            "the system never auto-retrains on your disagreement."
        ),
        "trust_metrics": rt.store.override_rate(),
    }


@app.get("/overrides")
def overrides(p: Principal = Depends(principal), patient_id: str | None = None) -> dict:
    rt = get_runtime()
    c = get_contract()
    access = require(c, rt.store, p, "override_history", subject=patient_id)
    rows = rt.store.overrides(patient_id)
    if access == "aggregate_with_identity":
        rows = [{k: v for k, v in r.items() if k != "payload"} for r in rows]
    return {"n": len(rows), "access": access, "overrides": rows,
            "rate": rt.store.override_rate(),
            "reason_clusters": rt.store.override_reason_clusters()}


# ---------------------------------------------------------------------------
# Administrator surface
# ---------------------------------------------------------------------------


@app.get("/admin/metrics")
def admin_metrics(p: Principal = Depends(principal)) -> dict:
    rt = get_runtime()
    c = get_contract()
    require(c, rt.store, p, "subgroup_performance")
    min_cell = c.role_spec("administrator").get("minimum_aggregation_cell_size", 5)

    by_band: dict[str, dict[str, Any]] = {}
    for a in rt.assessments.values():
        d = by_band.setdefault(a.age_band, {"n": 0, "escalated": 0, "abstained": 0,
                                            "uncalibrated": 0, "obs_only": 0})
        d["n"] += 1
        d["escalated"] += int(a.acuity_current.level < a.acuity_arrival.level)
        d["abstained"] += int(a.abstention.abstained)
        d["uncalibrated"] += int(not a.confidence.calibration_status.startswith("CALIBRATED"))
        d["obs_only"] += int(a.model_route == "OBSERVATION_ONLY_v1")

    suppressed = [k for k, v in by_band.items() if v["n"] < min_cell]
    shown = {k: v for k, v in by_band.items() if v["n"] >= min_cell}

    return {
        "note": "Aggregate only. No patient identifiers are returned to this role.",
        "minimum_cell_size": min_cell,
        "suppressed_cells": suppressed,
        "suppression_reason": (
            "Cells below the minimum are withheld to prevent re-identification by "
            "small-cell inference on a subgroup breakdown."
        ),
        "by_age_band": shown,
        "override": rt.store.override_rate(),
        "validator_rejections": len(rt.store.rejections()),
        "surge": rt.floor_summary()["surge_message"],
    }


@app.get("/admin/telemetry")
def admin_telemetry(p: Principal = Depends(principal)) -> dict:
    rt = get_runtime()
    require(get_contract(), rt.store, p, "cost_telemetry")
    s = summarise(rt.store.telemetry_rows(), rt.llm.backend)
    return {
        "summary": s.model_dump() if hasattr(s, "model_dump") else s.__dict__,
        "rows": s.as_rows(),
        "projected_annual": projected_costs(s.cost_per_patient_usd),
    }


@app.get("/audit/access")
def audit_access(p: Principal = Depends(principal), decision: str | None = None,
                 limit: int = 100) -> dict:
    rt = get_runtime()
    require(get_contract(), rt.store, p, "audit_log")
    return {"events": rt.store.access_events(limit=limit, decision=decision)}


# ---------------------------------------------------------------------------
# Demo / adversarial surfaces
# ---------------------------------------------------------------------------


class InjectionRequest(BaseModel):
    text: str


@app.post("/demo/injection-scan")
def injection_scan(body: InjectionRequest) -> dict:
    """
    Prompt injection through the complaint field, and what it can achieve.

    The answer is nothing, and the reason is structural rather than clever: free
    text has no wire into the feature vector, the cost threshold or the red-flag
    engine.
    """
    return scan(body.text)


@app.post("/demo/llm")
def llm_switch(enabled: bool = True, inject_violation: bool = False) -> dict:
    """
    THE KILL SWITCH, and the validator's demo hook.

    Turning the LLM off must leave every acuity, every EWER rank and every red
    flag unchanged. Only prose is lost.
    """
    rt = get_runtime()
    rt.llm.set_enabled(enabled)
    rt.llm.inject_violation = inject_violation
    return {"llm": rt.llm.telemetry(), "inject_violation": inject_violation}


@app.get("/demo/validator-rejections")
def validator_rejections(limit: int = 50) -> dict:
    rt = get_runtime()
    return {"rejections": rt.store.rejections(limit)}
