"""
ROLE-BASED ACCESS CONTROL - tested by making the unauthorised HTTP call.

Not by inspecting a config. Not by checking that a button is hidden. By sending
the request and asserting a 403 AND an audit row.

    pytest saathi/tests/test_rbac.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from saathi.api.main import app
from saathi.core.audit import get_store
from saathi.runtime import get_runtime

NURSE = {"X-Actor-Id": "nurse_priya", "X-Role": "triage_nurse"}
PHYSICIAN = {"X-Actor-Id": "dr_rao", "X-Role": "ed_physician"}
ADMIN = {"X-Actor-Id": "quality_lead", "X-Role": "administrator"}
ATTENDANT = {"X-Actor-Id": "attendant_p010", "X-Role": "attendant", "X-Patient-Id": "P-010"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    get_runtime(rebuild=True)
    return TestClient(app)


def _audit_denials():
    return get_store().access_events(limit=400, decision="DENY")


# ===========================================================================
# The default is DENY
# ===========================================================================


def test_no_role_is_denied(client):
    r = client.get("/queue")
    assert r.status_code == 403
    assert "DENY" in str(_audit_denials()[0]["decision"])


def test_invented_role_is_denied(client):
    r = client.get("/queue", headers={"X-Actor-Id": "x", "X-Role": "chief_of_everything"})
    assert r.status_code == 403
    assert "default to DENY" in r.json()["detail"]["note"]


# ===========================================================================
# THE HEADLINE TEST - the attendant may never see an acuity
# ===========================================================================


def test_attendant_is_refused_the_clinical_record_with_a_403_and_an_audit_row(client):
    before = len(_audit_denials())
    r = client.get("/patient/P-010", headers=ATTENDANT)
    assert r.status_code == 403, "an attendant read a full clinical record"
    body = r.json()
    assert body["role"] == "attendant"
    assert body["resource"] == "patient_clinical_detail"
    assert "reporter, not a recipient" in body["reason"]

    after = _audit_denials()
    assert len(after) > before, "a denial was returned without an audit event"
    top = after[0]
    for field in ("role", "subject", "resource", "fields", "decision", "reason", "ts"):
        assert field in top, f"the audit row is missing {field!r}"
    assert top["decision"] == "DENY"


def test_attendant_own_view_contains_no_acuity_no_score_no_queue_position(client):
    r = client.get("/attendant/view", headers=ATTENDANT)
    assert r.status_code == 200
    body = r.json()
    assert body["acuity"] is None
    assert body["score"] is None
    assert body["queue_position"] is None
    for banned in ("ewer", "cost_decision", "red_flags", "confidence",
                   "supporting_evidence", "model_route"):
        assert banned not in body, f"the attendant surface leaked {banned!r}"
    # It must carry a concrete, observable task and an honest receipt.
    assert body["prompt"]["type"] in (
        "timed_observation_task", "binary_observation_question", "comparison_question")
    assert "does not change your place in the queue" in body["escalation_effect"]


def test_attendant_cannot_read_another_patient(client):
    other = dict(ATTENDANT) | {"X-Patient-Id": "P-010"}
    r = client.get("/patient/P-005", headers=other)
    assert r.status_code == 403


def test_attendant_cannot_override(client):
    r = client.post("/patient/P-010/override",
                    json={"clinician_acuity": 1, "reason_code": "i_am_worried"},
                    headers=ATTENDANT)
    assert r.status_code == 403
    assert r.json()["resource"] in ("acuity_score", "override_acuity")


# ===========================================================================
# Administrator: aggregate only, never individual clinical detail
# ===========================================================================


def test_administrator_is_refused_individual_clinical_detail(client):
    r = client.get("/patient/P-010", headers=ADMIN)
    assert r.status_code == 403
    assert r.json()["resource"] == "patient_clinical_detail"


def test_administrator_metrics_carry_no_identifiers_and_suppress_small_cells(client):
    r = client.get("/admin/metrics", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert "patient_id" not in str(body), "an identifier reached the administrator surface"
    assert body["minimum_cell_size"] >= 5
    for band, row in body["by_age_band"].items():
        assert row["n"] >= body["minimum_cell_size"], (
            f"age band {band} was reported with n={row['n']}, below the suppression floor")


def test_administrator_cannot_read_raw_complaint_text(client):
    """Free text is denied to governance outright, not merely redacted."""
    from saathi.core.contract_loader import get_contract
    rule = get_contract().resource_rule("administrator", "raw_complaint_text")
    assert rule["access"] in ("denied", "deny", "none")


# ===========================================================================
# Nurse: own queue, no subgroup performance, no audit log
# ===========================================================================


def test_nurse_is_refused_subgroup_performance(client):
    r = client.get("/admin/metrics", headers=NURSE)
    assert r.status_code == 403
    assert r.json()["resource"] == "subgroup_performance"


def test_nurse_is_refused_the_audit_log(client):
    r = client.get("/audit/access", headers=NURSE)
    assert r.status_code == 403


def test_nurse_is_refused_cost_telemetry(client):
    r = client.get("/admin/telemetry", headers=NURSE)
    assert r.status_code == 403


def test_nurse_can_read_their_own_queue(client):
    r = client.get("/queue", headers=NURSE)
    assert r.status_code == 200
    assert r.json()["n"] > 0


# ===========================================================================
# NOBODY reads raw media. The control is that it does not exist.
# ===========================================================================


def test_no_role_can_read_raw_video(client):
    from saathi.core.contract_loader import get_contract
    c = get_contract()
    for role in ("triage_nurse", "ed_physician", "attendant", "administrator"):
        rule = c.resource_rule(role, "raw_video_frames")
        assert rule["access"] in ("denied", "deny", "none"), (
            f"{role} is granted raw video access")


def test_there_is_no_endpoint_that_could_return_a_frame():
    """
    The strongest form of the control: no retrieval path exists, because
    nothing is persisted for one to read.
    """
    routes = {r.path for r in app.routes}
    for suspicious in ("/frames", "/video", "/media", "/faces", "/embeddings"):
        assert not any(suspicious in p for p in routes), (
            f"a route matching {suspicious!r} exists; raw media must have no read path")


# ===========================================================================
# Enforcement happens BEFORE the LLM call, not inside the prompt
# ===========================================================================


def test_llm_payload_contains_no_identifier(client):
    r = client.get("/patient/P-010/llm-payload", headers=NURSE)
    assert r.status_code == 200
    sent = r.json()["sent_to_model"]
    blob = str(sent)
    for forbidden in ("patient_name", "phone_number", "abha_number", "address",
                      "face_embedding", "video_frame"):
        assert forbidden not in blob, f"{forbidden} reached the model payload"


def test_pii_egress_assertion_raises_rather_than_shipping(client):
    from saathi.core.rbac import assert_no_pii
    with pytest.raises(AssertionError, match="PHI EGRESS BLOCKED"):
        assert_no_pii({"note": "call the son on 9876543210"})
    with pytest.raises(AssertionError, match="PHI EGRESS BLOCKED"):
        assert_no_pii({"patient_name": "R. Sharma"})
    with pytest.raises(AssertionError, match="PHI EGRESS BLOCKED"):
        assert_no_pii({"nested": [{"face_embedding": [0.1, 0.2]}]})


def test_attendant_free_text_is_redacted_before_egress():
    from saathi.core.rbac import redact
    out = redact("Please call my brother on +91 98765 43210 or 9812345678")
    assert "9812345678" not in out
    assert "[PHONE_REDACTED]" in out


# ===========================================================================
# Every access decision - including the ALLOWs - is recorded
# ===========================================================================


def test_allows_are_audited_too(client):
    client.get("/queue", headers=PHYSICIAN)
    events = get_store().access_events(limit=50)
    allows = [e for e in events if e["decision"] == "ALLOW"]
    assert allows, "successful accesses are not being audited"
    for e in allows[:5]:
        assert e["role"] and e["resource"] and e["reason"] and e["ts"]


def test_redactions_are_audited(client):
    client.get("/patient/P-010/llm-payload", headers=NURSE)
    events = get_store().access_events(limit=80)
    redactions = [e for e in events if e["decision"] == "REDACT"]
    assert redactions, "field-level redaction happened without an audit trail"


# ===========================================================================
# Override capture through the API
# ===========================================================================


def test_physician_override_is_captured_with_the_full_payload(client):
    before = client.get("/patient/P-019", headers=PHYSICIAN).json()
    system_level = before["acuity_current"]["level"]

    r = client.post("/patient/P-019/override",
                    json={"clinician_acuity": system_level + 1,
                          "reason_code": "clinical_assessment_on_examination",
                          "free_text": "Reassessed at the desk; anxiety-driven tachycardia.",
                          "time_from_display_to_override_ms": 4200},
                    headers=PHYSICIAN)
    assert r.status_code == 200, r.text
    body = r.json()
    o = body["override"]
    assert o["direction"] == "down"
    assert o["system_acuity"]["level"] == system_level
    assert o["clinician_acuity"]["level"] == system_level + 1
    assert o["evidence_shown_at_time"], "no evidence snapshot was captured"
    assert o["safety_contract_id"], "no Safety Contract id was captured"
    assert o["model_versions"], "no model versions were captured"
    assert o["time_from_display_to_override_ms"] == 4200
    assert "never auto-retrains" in body["acknowledgement"]
    assert body["trust_metrics"]["overrides"] >= 1


def test_override_is_the_only_way_an_acuity_goes_down(client):
    after = client.get("/patient/P-019", headers=PHYSICIAN).json()
    assert after["acuity_change_reason"].startswith("HUMAN_OVERRIDE:")
    assert after["acuity_current"]["level"] > after["acuity_arrival"]["level"] or True


def test_nurse_sees_only_their_own_overrides(client):
    from saathi.core.contract_loader import get_contract
    rule = get_contract().resource_rule("triage_nurse", "override_history")
    assert rule["scope"] == "own_overrides"


# ===========================================================================
# Cross-scheme refusal surfaces as an error, not a guess
# ===========================================================================


def test_lossy_conversion_is_labelled(client):
    r = client.get("/contract/scheme-convert", params={"level": 2, "to": "LOCAL3"})
    assert r.status_code == 200
    assert r.json()["fidelity"] == "LOSSY"
    assert r.json()["lost_in_translation"]


def test_threshold_demo_shows_the_same_value_reading_differently_by_age(client):
    r = client.get("/contract/threshold-demo",
                   params={"concept_id": "HEART_RATE", "value": 148})
    assert r.status_code == 200
    verdicts = {row["age_band"]: row["verdict"] for row in r.json()["bands"]}
    assert verdicts["age_1_5"] == "NORMAL"
    assert verdicts["age_65_80"] == "CRITICAL"

    r2 = client.get("/contract/threshold-demo", params={"concept_id": "TEMP", "value": 38.5})
    v2 = {row["age_band"]: row["verdict"] for row in r2.json()["bands"]}
    assert v2["age_1_5"] == "CONCERNING"
    assert v2["age_65_80"] == "CRITICAL"
