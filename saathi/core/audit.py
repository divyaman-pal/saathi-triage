"""
Audit, lineage and override store (SQLite).

Three obligations are met here:

  ACCESS ACCOUNTABILITY   every access decision writes
                          (role, subject, resource, fields, decision, reason, timestamp)
                          - including the DENIALS, which are the ones that matter.

  CLINICAL ACCOUNTABILITY every displayed acuity is traceable back to the raw
                          observation that produced it, through the decision
                          rule, the fusion output, the per-channel sub-scores,
                          the feature values with their staleness, the quality
                          gate result and the model + contract versions.

  DECISION ACCOUNTABILITY every override is captured with the full evidence
                          snapshot AS IT WAS SHOWN, so the question "what did
                          the clinician actually see when they decided that?"
                          has an answer seven years later.

Retention periods come from contract/entitlements.yaml. purge_expired() applies
them, and reports what it deleted.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .models import OverrideRecord, SafetyContract

DB_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "saathi_audit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    role TEXT NOT NULL,
    subject TEXT,
    resource TEXT NOT NULL,
    fields TEXT,
    decision TEXT NOT NULL,
    reason TEXT,
    request_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_access_ts ON access_log(ts);
CREATE INDEX IF NOT EXISTS ix_access_decision ON access_log(decision);

CREATE TABLE IF NOT EXISTS lineage (
    lineage_ref TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_lineage_patient ON lineage(patient_id);

CREATE TABLE IF NOT EXISTS safety_contracts (
    contract_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    persona TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sc_patient ON safety_contracts(patient_id);

CREATE TABLE IF NOT EXISTS overrides (
    override_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    clinician_id TEXT NOT NULL,
    role TEXT NOT NULL,
    direction TEXT NOT NULL,
    system_level INTEGER NOT NULL,
    clinician_level INTEGER NOT NULL,
    reason_code TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ovr_patient ON overrides(patient_id);

CREATE TABLE IF NOT EXISTS validator_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    patient_id TEXT,
    contract_id TEXT,
    check_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    detail TEXT,
    offending_text TEXT
);

CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    patient_id TEXT,
    stage TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    model_calls INTEGER DEFAULT 0,
    model_type TEXT,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    tier TEXT,
    surge INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_tel_stage ON telemetry(stage);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    patient_id TEXT,
    kind TEXT NOT NULL,
    detail TEXT
);
"""


class AuditStore:
    """
    Append-mostly audit, lineage and telemetry store.

    CONNECTION LIFECYCLE
        One connection is held open for the life of the store rather than a new
        one opened per write. The original per-write connect/commit/close cost
        23 of the 27 seconds needed to assess a 62-patient cohort - 1002
        connections and 1002 full commits, against roughly 3 seconds of actual
        clinical computation. On a cloud container with a slow disk that was
        enough to stop the app booting at all.

    DURABILITY TRADEOFF, STATED PLAINLY
        journal_mode=WAL with synchronous=NORMAL. A committed row survives the
        process being killed, but the last transaction can be lost if the
        machine loses power mid-write. For a clinical audit trail that is a real
        tradeoff and not a free win, so it is recorded here rather than buried:
        a deployment that must survive power loss without losing the final audit
        row should set synchronous=FULL and accept the write cost, or put the
        audit store on a database with its own durability guarantees. The
        prototype's threat model is a demo container, not a hospital.

    THREADING
        Streamlit runs each session's script in its own thread, so the shared
        connection is serialised by a reentrant lock. It is reentrant because
        some callers already hold the lock before entering `_conn` and others
        do not; a plain Lock would deadlock the former.
    """

    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cx = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._cx.row_factory = sqlite3.Row
        self._cx.execute("PRAGMA journal_mode=WAL")
        self._cx.execute("PRAGMA synchronous=NORMAL")
        with self._conn() as cx:
            cx.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._cx
                self._cx.commit()
            except Exception:
                self._cx.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._cx.close()

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="milliseconds")

    # -- access ------------------------------------------------------------

    def log_access(
        self, *, actor_id: str, role: str, subject: str | None, resource: str,
        fields: list[str] | None, decision: str, reason: str = "", request_id: str = "",
    ) -> None:
        with self._lock, self._conn() as cx:
            cx.execute(
                "INSERT INTO access_log (ts, actor_id, role, subject, resource, fields, decision, reason, request_id) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (self._now(), actor_id, role, subject, resource,
                 json.dumps(fields or []), decision, reason, request_id),
            )

    def access_events(self, limit: int = 200, decision: str | None = None) -> list[dict]:
        q = "SELECT * FROM access_log"
        args: list[Any] = []
        if decision:
            q += " WHERE decision = ?"
            args.append(decision)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as cx:
            return [dict(r) for r in cx.execute(q, args)]

    # -- lineage -----------------------------------------------------------

    def write_lineage(self, lineage_ref: str, patient_id: str, payload: dict) -> None:
        with self._lock, self._conn() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO lineage (lineage_ref, ts, patient_id, payload) VALUES (?,?,?,?)",
                (lineage_ref, self._now(), patient_id, json.dumps(payload, default=str)),
            )

    def read_lineage(self, lineage_ref: str) -> dict | None:
        with self._conn() as cx:
            r = cx.execute("SELECT payload FROM lineage WHERE lineage_ref = ?", (lineage_ref,)).fetchone()
            return json.loads(r["payload"]) if r else None

    # -- safety contracts --------------------------------------------------

    def write_contract(self, sc: SafetyContract) -> None:
        with self._lock, self._conn() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO safety_contracts (contract_id, ts, patient_id, persona, payload) "
                "VALUES (?,?,?,?,?)",
                (sc.contract_id, self._now(), sc.patient_id, sc.persona.value,
                 sc.model_dump_json()),
            )

    def read_contract(self, contract_id: str) -> dict | None:
        with self._conn() as cx:
            r = cx.execute("SELECT payload FROM safety_contracts WHERE contract_id = ?",
                           (contract_id,)).fetchone()
            return json.loads(r["payload"]) if r else None

    # -- overrides ---------------------------------------------------------

    def write_override(self, o: OverrideRecord) -> None:
        with self._lock, self._conn() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO overrides (override_id, ts, patient_id, clinician_id, role, "
                "direction, system_level, clinician_level, reason_code, payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (o.override_id, o.timestamp.isoformat(), o.patient_id, o.clinician_id, o.role.value,
                 o.direction, o.system_acuity.level, o.clinician_acuity.level, o.reason_code,
                 o.model_dump_json()),
            )

    def overrides(self, patient_id: str | None = None, limit: int = 200) -> list[dict]:
        q = "SELECT * FROM overrides"
        args: list[Any] = []
        if patient_id:
            q += " WHERE patient_id = ?"
            args.append(patient_id)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        with self._conn() as cx:
            return [dict(r) for r in cx.execute(q, args)]

    def override_rate(self) -> dict[str, float]:
        with self._conn() as cx:
            total = cx.execute("SELECT COUNT(*) c FROM safety_contracts").fetchone()["c"]
            n = cx.execute("SELECT COUNT(*) c FROM overrides").fetchone()["c"]
            down = cx.execute("SELECT COUNT(*) c FROM overrides WHERE direction='down'").fetchone()["c"]
            up = cx.execute("SELECT COUNT(*) c FROM overrides WHERE direction='up'").fetchone()["c"]
        return {
            "contracts_generated": total,
            "overrides": n,
            "override_rate": (n / total) if total else 0.0,
            "downgrades": down,
            "upgrades": up,
        }

    def override_reason_clusters(self) -> list[dict]:
        with self._conn() as cx:
            return [dict(r) for r in cx.execute(
                "SELECT reason_code, direction, COUNT(*) n, AVG(system_level - clinician_level) mean_delta "
                "FROM overrides GROUP BY reason_code, direction ORDER BY n DESC")]

    # -- validator ---------------------------------------------------------

    def log_rejection(self, *, patient_id: str | None, contract_id: str | None,
                      check_id: str, severity: str, detail: str, offending_text: str) -> None:
        with self._lock, self._conn() as cx:
            cx.execute(
                "INSERT INTO validator_rejections (ts, patient_id, contract_id, check_id, severity, detail, offending_text) "
                "VALUES (?,?,?,?,?,?,?)",
                (self._now(), patient_id, contract_id, check_id, severity, detail, offending_text[:2000]),
            )

    def rejections(self, limit: int = 100) -> list[dict]:
        with self._conn() as cx:
            return [dict(r) for r in cx.execute(
                "SELECT * FROM validator_rejections ORDER BY id DESC LIMIT ?", (limit,))]

    # -- telemetry ---------------------------------------------------------

    def log_stage(self, *, patient_id: str | None, stage: str, latency_ms: float,
                  model_calls: int = 0, model_type: str = "", tokens_in: int = 0,
                  tokens_out: int = 0, cost_usd: float = 0.0, tier: str = "",
                  surge: bool = False) -> None:
        with self._lock, self._conn() as cx:
            cx.execute(
                "INSERT INTO telemetry (ts, patient_id, stage, latency_ms, model_calls, model_type, "
                "tokens_in, tokens_out, cost_usd, tier, surge) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (self._now(), patient_id, stage, latency_ms, model_calls, model_type,
                 tokens_in, tokens_out, cost_usd, tier, int(surge)),
            )

    def telemetry_rows(self, limit: int = 20000) -> list[dict]:
        with self._conn() as cx:
            return [dict(r) for r in cx.execute(
                "SELECT * FROM telemetry ORDER BY id DESC LIMIT ?", (limit,))]

    # -- events ------------------------------------------------------------

    def log_event(self, kind: str, detail: str = "", patient_id: str | None = None) -> None:
        with self._lock, self._conn() as cx:
            cx.execute("INSERT INTO events (ts, patient_id, kind, detail) VALUES (?,?,?,?)",
                       (self._now(), patient_id, kind, detail))

    def events(self, limit: int = 200) -> list[dict]:
        with self._conn() as cx:
            return [dict(r) for r in cx.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]

    # -- retention ---------------------------------------------------------

    def purge_expired(self, retention: dict) -> dict[str, int]:
        """
        Apply the contract's retention periods. DPDP Act 2023 storage limitation.

        Governance and audit records are excluded from erasure under the
        legal-obligation basis; that exclusion is disclosed at consent.
        """
        deleted: dict[str, int] = {}
        now = datetime.now()
        plan = [
            ("telemetry", retention["telemetry"]["retention_days"]),
            ("lineage", retention["clinical_derived"]["retention_days"]),
            ("safety_contracts", retention["clinical_derived"]["retention_days"]),
            ("overrides", retention["governance"]["retention_days"]),
            ("access_log", retention["governance"]["retention_days"]),
        ]
        with self._lock, self._conn() as cx:
            for table, days in plan:
                cutoff = (now - timedelta(days=int(days))).isoformat()
                cur = cx.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
                deleted[table] = cur.rowcount
        return deleted

    def reset(self) -> None:
        with self._lock, self._conn() as cx:
            for t in ("access_log", "lineage", "safety_contracts", "overrides",
                      "validator_rejections", "telemetry", "events"):
                cx.execute(f"DELETE FROM {t}")


_STORE: AuditStore | None = None


def get_store(path: Path | str = DB_PATH) -> AuditStore:
    global _STORE
    if _STORE is None or Path(path) != _STORE.path:
        _STORE = AuditStore(path)
    return _STORE
