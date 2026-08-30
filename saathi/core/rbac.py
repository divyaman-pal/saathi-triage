"""
Backend-enforced role-based access control and PHI minimisation.

THREE ENFORCEMENT POINTS, IN ORDER
    1. BEFORE RETRIEVAL  - scope_patients() narrows the query itself. Rows the
                           caller may not see are never read out of the store.
    2. BEFORE ANALYSIS   - project() strips fields the role may not hold before
                           anything computes on them.
    3. BEFORE ANY LLM CALL - llm_payload() builds the model prompt from an
                           already-minimised projection. We never hand a model
                           data the caller may not see and trust it to withhold.

Hiding a widget is not access control. Every rule here is exercised by an
unauthorised HTTP request in tests/test_rbac.py, which asserts a 403 AND an
audit row.

RAW MEDIA HAS NO READ PATH
    There is no function in this module, and none in the API, that returns a
    video frame or an audio waveform - because there is nothing to return.
    Frames are processed in memory and discarded within the window. The
    "raw_video_frames: denied" row in the entitlements table is a statement
    about a capability that does not exist, for any role, including operators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditStore
from .contract_loader import Contract
from .models import Assessment, Evidence, Role


class AccessDenied(Exception):
    def __init__(self, role: str, resource: str, reason: str):
        self.role, self.resource, self.reason = role, resource, reason
        super().__init__(f"403 {role} may not access {resource}: {reason}")


@dataclass
class Principal:
    actor_id: str
    role: Role
    floor_id: str = "ED-1"
    own_queue: list[str] = field(default_factory=list)
    own_patient: str | None = None       # attendants are scoped to exactly one


# ---------------------------------------------------------------------------
# 1. Before retrieval
# ---------------------------------------------------------------------------


def scope_patients(
    contract: Contract, store: AuditStore, principal: Principal,
    candidate_ids: list[str], *, request_id: str = "",
) -> list[str]:
    spec = contract.role_spec(principal.role.value)
    scope = spec["row_scope"]

    if scope == "own_queue":
        allowed = [p for p in candidate_ids if not principal.own_queue or p in principal.own_queue]
    elif scope == "own_floor":
        allowed = list(candidate_ids)
    elif scope == "own_patient":
        allowed = [p for p in candidate_ids if p == principal.own_patient]
    elif scope == "aggregate":
        allowed = []       # administrators never retrieve individual rows
    else:
        allowed = []

    denied = [p for p in candidate_ids if p not in allowed]
    if denied:
        store.log_access(
            actor_id=principal.actor_id, role=principal.role.value,
            subject=",".join(denied[:20]), resource="patient_rows", fields=None,
            decision="DENY",
            reason=f"row_scope={scope}; {len(denied)} patient row(s) outside scope, excluded before retrieval",
            request_id=request_id)
    if allowed:
        store.log_access(
            actor_id=principal.actor_id, role=principal.role.value,
            subject=",".join(allowed[:20]), resource="patient_rows", fields=None,
            decision="ALLOW", reason=f"row_scope={scope}", request_id=request_id)
    return allowed


def require(
    contract: Contract, store: AuditStore, principal: Principal, resource: str,
    *, subject: str | None = None, request_id: str = "",
) -> str:
    """Assert an entitlement. Raises AccessDenied and writes an audit row."""
    rule = contract.resource_rule(principal.role.value, resource)
    access = rule.get("access", "deny")
    if access in ("denied", "deny", "none"):
        store.log_access(
            actor_id=principal.actor_id, role=principal.role.value, subject=subject,
            resource=resource, fields=None, decision="DENY",
            reason=rule.get("reason", "not granted; default_decision is DENY"),
            request_id=request_id)
        raise AccessDenied(principal.role.value, resource,
                           rule.get("reason", "not granted; default_decision is DENY"))

    if access == "own_only" and subject and principal.own_patient and subject != principal.own_patient:
        store.log_access(
            actor_id=principal.actor_id, role=principal.role.value, subject=subject,
            resource=resource, fields=None, decision="DENY",
            reason=f"own_only scope; caller is bound to {principal.own_patient}",
            request_id=request_id)
        raise AccessDenied(principal.role.value, resource,
                           f"own_only scope; caller is bound to {principal.own_patient}")

    store.log_access(
        actor_id=principal.actor_id, role=principal.role.value, subject=subject,
        resource=resource, fields=None, decision="ALLOW",
        reason=f"access={access}, scope={rule.get('scope')}", request_id=request_id)
    return access


def permits(contract: Contract, principal: Principal, resource: str) -> bool:
    """Non-raising, non-logging check - for deciding what to render."""
    return contract.resource_rule(principal.role.value, resource).get("access", "deny") \
        not in ("denied", "deny", "none")


# ---------------------------------------------------------------------------
# 2. Before analysis - field projection
# ---------------------------------------------------------------------------

RESOURCE_FOR_CONCEPT = {
    "COMPLAINT_TEXT": "raw_complaint_text",
    "ATTENDANT_FREE_TEXT": "attendant_free_text",
}
CAMERA_DERIVED = {"HEART_RATE", "RESP_RATE", "STILLNESS_MINUTES", "POSTURE",
                  "WORK_OF_BREATHING", "SKIN_COLOR_CHANGE", "ARRIVAL_MODE"}


def project_evidence(
    contract: Contract, store: AuditStore, principal: Principal,
    evidence: list[Evidence], *, subject: str | None = None, request_id: str = "",
) -> list[Evidence]:
    """Drop every evidence item the role may not hold. Log the drops."""
    out: list[Evidence] = []
    dropped: list[str] = []
    for ev in evidence:
        resource = RESOURCE_FOR_CONCEPT.get(ev.concept_id)
        if resource is None:
            resource = ("derived_camera_features"
                        if ev.source_channel.value == "camera" and ev.concept_id in CAMERA_DERIVED
                        else "patient_clinical_detail")
        rule = contract.resource_rule(principal.role.value, resource)
        access = rule.get("access", "deny")
        if access in ("denied", "deny", "none"):
            dropped.append(f"{ev.concept_id}:{resource}")
            continue
        if access == "own_only" and principal.own_patient and ev.patient_id != principal.own_patient:
            dropped.append(f"{ev.concept_id}:own_only")
            continue
        if access == "aggregate":
            dropped.append(f"{ev.concept_id}:aggregate_only")
            continue
        out.append(ev)

    if dropped:
        store.log_access(
            actor_id=principal.actor_id, role=principal.role.value, subject=subject,
            resource="evidence_projection", fields=sorted(set(dropped))[:40], decision="REDACT",
            reason=f"{len(dropped)} evidence item(s) removed before analysis",
            request_id=request_id)
    return out


def project_assessment(
    contract: Contract, store: AuditStore, principal: Principal, a: Assessment,
    *, request_id: str = "",
) -> dict[str, Any]:
    """
    Build the role-appropriate view. Different roles get genuinely different
    information depth - not the same payload with different labels.
    """
    role = principal.role
    base: dict[str, Any] = {
        "patient_id": a.patient_id,
        "assessed_at": a.assessed_at,
        "tier": a.tier,
    }

    if role is Role.ATTENDANT:
        # No acuity. No score. No EWER. No queue position. No clinical detail.
        store.log_access(actor_id=principal.actor_id, role=role.value, subject=a.patient_id,
                         resource="attendant_view", fields=["wait_time_own", "receipt_of_report"],
                         decision="ALLOW",
                         reason="attendant projection: acuity, score, EWER, queue position and "
                                "clinical detail all withheld by entitlement",
                         request_id=request_id)
        return base | {
            "waited_minutes": a.sla.waited_minutes,
            "reports_received": True,
            "acuity": None,
            "score": None,
            "queue_position": None,
        }

    if role is Role.ADMINISTRATOR:
        store.log_access(actor_id=principal.actor_id, role=role.value, subject=None,
                         resource="aggregate_metrics", fields=["acuity_distribution", "override_rate"],
                         decision="ALLOW", reason="aggregate_only projection; no patient identifiers",
                         request_id=request_id)
        return {
            "tier": a.tier,
            "age_band": a.age_band,
            "acuity_level": a.acuity_current.level,
            "epistemic_state": a.epistemic_state.value,
            "abstained": a.abstention.abstained,
            "model_route": a.model_route,
            "calibration_status": a.confidence.calibration_status,
            "patient_id": None,     # explicitly nulled, not merely omitted
        }

    payload = base | {
        "age_years": a.age_years,
        "age_band": a.age_band,
        "sex": a.sex,
        "acuity_arrival": a.acuity_arrival.model_dump(),
        "acuity_current": a.acuity_current.model_dump(),
        "acuity_change_reason": a.acuity_change_reason,
        "epistemic_state": a.epistemic_state.value,
        "confidence": a.confidence.model_dump(),
        "abstention": a.abstention.model_dump(),
        "ewer": a.ewer.model_dump(),
        "sla": a.sla.model_dump(),
        "red_flags": [r.model_dump() for r in a.red_flags],
        "materiality": [m.model_dump() for m in a.materiality],
        "supporting_evidence": [e.model_dump() for e in a.supporting_evidence],
        "contradictory_evidence": [e.model_dump() for e in a.contradictory_evidence],
        "channel_status": [c.model_dump() for c in a.channel_status],
        "missing_data": a.missing_data,
        "stale_data": a.stale_data,
        "model_route": a.model_route,
        "model_versions": a.model_versions,
        "narrative": a.narrative.model_dump() if a.narrative else None,
        "lineage_ref": a.lineage_ref,
        "stage_latency_ms": a.stage_latency_ms,
    }
    if role is Role.ED_PHYSICIAN:
        payload["cost_decision"] = a.cost_decision.model_dump() if a.cost_decision else None
        payload["override"] = a.override.model_dump() if a.override else None
    else:
        payload["cost_decision"] = a.cost_decision.model_dump() if a.cost_decision else None
    return payload


# ---------------------------------------------------------------------------
# 3. Before any LLM call - PHI minimisation
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    (re.compile(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b\d{2}-\d{4}-\d{4}-\d{4}\b"), "[ABHA_REDACTED]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[ID_REDACTED]"),
    (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"), "[EMAIL_REDACTED]"),
]


def redact(text: str) -> str:
    out = text
    for pattern, repl in PII_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def llm_payload(
    contract: Contract, store: AuditStore, principal: Principal, payload: dict,
    *, subject: str | None = None, request_id: str = "",
) -> dict:
    """
    Final minimisation before the prompt is built.

    Removes every DIRECT_IDENTIFIER, BIOMETRIC and RAW_MEDIA field, redacts
    free-text, and asserts the result is clean. The exact object returned here
    is what the demo displays as "what the LLM actually receives".
    """
    forbidden_classes = {"DIRECT_IDENTIFIER", "BIOMETRIC", "RAW_MEDIA", "GOVERNANCE"}
    forbidden_fields: set[str] = set()
    for cname in forbidden_classes:
        spec = contract.entitlements["field_classes"].get(cname, {})
        forbidden_fields |= set(spec.get("fields", []))

    removed: list[str] = []

    def clean(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in forbidden_fields:
                    removed.append(k)
                    continue
                out[k] = clean(v)
            return out
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, str):
            r = redact(obj)
            if r != obj:
                removed.append("free_text_pii")
            return r
        return obj

    minimised = clean(payload)

    store.log_access(
        actor_id=principal.actor_id, role=principal.role.value, subject=subject,
        resource="llm_egress", fields=sorted(set(removed))[:40], decision="REDACT",
        reason=(f"PHI minimisation before model call: {len(removed)} field(s)/match(es) removed. "
                "Classes DIRECT_IDENTIFIER, BIOMETRIC, RAW_MEDIA, GOVERNANCE are FORBIDDEN egress."),
        request_id=request_id)

    assert_no_pii(minimised)
    return minimised


def assert_no_pii(obj) -> None:
    """Belt and braces. Raises rather than shipping an identifier to a model."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("patient_name", "phone_number", "abha_number", "address",
                         "attendant_name", "attendant_phone", "face_embedding",
                         "video_frame", "audio_waveform"):
                    raise AssertionError(f"PHI EGRESS BLOCKED: field {k!r} reached the LLM payload.")
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for pattern, _ in PII_PATTERNS:
                if pattern.search(o):
                    raise AssertionError(f"PHI EGRESS BLOCKED: identifier-shaped text in {o[:60]!r}")
    walk(obj)
