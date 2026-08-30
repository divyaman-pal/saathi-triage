"""
Runtime telemetry: measured latency, model calls, tokens, and cost.

Every number here is MEASURED at runtime, not estimated from a spreadsheet. The
pipeline times each stage with perf_counter and writes a row per stage per
patient into the audit store.

THE LATENCY BUDGET THAT MATTERS
    The nurse-facing decision must render in under 2 seconds WITHOUT the LLM.
    LLM narration is additive and droppable. That is why `decision_p95_ms`
    below excludes the llm_render and validate stages and reports them
    separately - a system whose triage decision waits on a network call to a
    language model has put the model on the safety path, which is exactly what
    the architecture forbids.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

# Stages that are ON the decision path. The nurse can act as soon as these are
# done. Everything after is prose.
DECISION_STAGES = [
    "signal_acquisition", "quality_gating", "feature_assembly", "model_inference",
    "fusion", "red_flags", "confidence", "cost_decision", "ewer", "contract_generation",
]
NARRATION_STAGES = ["llm_render", "validate"]


@dataclass
class StageTimer:
    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[stage] = self.stages.get(stage, 0.0) + (time.perf_counter() - t0) * 1000.0

    @property
    def decision_ms(self) -> float:
        return round(sum(v for k, v in self.stages.items() if k in DECISION_STAGES), 3)

    @property
    def narration_ms(self) -> float:
        return round(sum(v for k, v in self.stages.items() if k in NARRATION_STAGES), 3)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 3)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 3)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


@dataclass
class TelemetrySummary:
    n_patients: int
    per_stage: dict[str, dict[str, float]]
    decision_p50_ms: float
    decision_p95_ms: float
    decision_p99_ms: float
    end_to_end_p50_ms: float
    end_to_end_p95_ms: float
    end_to_end_p99_ms: float
    narration_p50_ms: float
    narration_p95_ms: float
    model_calls_per_patient: dict[str, float]
    tokens_in_total: int
    tokens_out_total: int
    cost_total_usd: float
    cost_per_patient_usd: float
    cost_per_1000_visits_usd: float
    llm_backend: str
    budget_met: bool
    budget_ms: float = 2000.0

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("Patients assessed", f"{self.n_patients}"),
            ("Decision latency p50 / p95 / p99 (no LLM)",
             f"{self.decision_p50_ms:.1f} / {self.decision_p95_ms:.1f} / {self.decision_p99_ms:.1f} ms"),
            ("End-to-end p50 / p95 / p99 (with narration)",
             f"{self.end_to_end_p50_ms:.1f} / {self.end_to_end_p95_ms:.1f} / {self.end_to_end_p99_ms:.1f} ms"),
            ("Narration p50 / p95", f"{self.narration_p50_ms:.1f} / {self.narration_p95_ms:.1f} ms"),
            ("2 s decision budget (LLM excluded)", "MET" if self.budget_met else "BREACHED"),
            ("Model calls per patient", ", ".join(f"{k} {v:.2f}" for k, v in self.model_calls_per_patient.items())),
            ("Tokens in / out", f"{self.tokens_in_total:,} / {self.tokens_out_total:,}"),
            ("LLM backend", self.llm_backend),
            ("Cost per patient", f"${self.cost_per_patient_usd:.6f}"),
            ("Cost per 1,000 ED visits", f"${self.cost_per_1000_visits_usd:.2f}"),
        ]


def summarise(rows: list[dict], llm_backend: str = "stub") -> TelemetrySummary:
    """Aggregate raw telemetry rows from the audit store."""
    by_patient_stage: dict[str, dict[str, float]] = defaultdict(dict)
    per_stage_values: dict[str, list[float]] = defaultdict(list)
    model_calls: dict[str, int] = defaultdict(int)
    tin = tout = 0
    cost = 0.0

    for r in rows:
        pid = r.get("patient_id") or "-"
        stage = r["stage"]
        by_patient_stage[pid][stage] = by_patient_stage[pid].get(stage, 0.0) + r["latency_ms"]
        per_stage_values[stage].append(r["latency_ms"])
        if r.get("model_calls"):
            model_calls[r.get("model_type") or stage] += int(r["model_calls"])
        tin += int(r.get("tokens_in") or 0)
        tout += int(r.get("tokens_out") or 0)
        cost += float(r.get("cost_usd") or 0.0)

    n = max(1, len(by_patient_stage))
    decision = [sum(v for k, v in s.items() if k in DECISION_STAGES) for s in by_patient_stage.values()]
    total = [sum(s.values()) for s in by_patient_stage.values()]
    narration = [sum(v for k, v in s.items() if k in NARRATION_STAGES) for s in by_patient_stage.values()]

    per_stage = {
        stage: {
            "p50": percentile(vals, 0.50),
            "p95": percentile(vals, 0.95),
            "p99": percentile(vals, 0.99),
            "mean": round(sum(vals) / len(vals), 3),
            "n": len(vals),
        }
        for stage, vals in sorted(per_stage_values.items())
    }

    d95 = percentile(decision, 0.95)
    return TelemetrySummary(
        n_patients=len(by_patient_stage),
        per_stage=per_stage,
        decision_p50_ms=percentile(decision, 0.50),
        decision_p95_ms=d95,
        decision_p99_ms=percentile(decision, 0.99),
        end_to_end_p50_ms=percentile(total, 0.50),
        end_to_end_p95_ms=percentile(total, 0.95),
        end_to_end_p99_ms=percentile(total, 0.99),
        narration_p50_ms=percentile(narration, 0.50),
        narration_p95_ms=percentile(narration, 0.95),
        model_calls_per_patient={k: round(v / n, 3) for k, v in sorted(model_calls.items())},
        tokens_in_total=tin,
        tokens_out_total=tout,
        cost_total_usd=round(cost, 6),
        cost_per_patient_usd=round(cost / n, 8),
        cost_per_1000_visits_usd=round(cost / n * 1000, 4),
        llm_backend=llm_backend,
        budget_met=d95 < 2000.0,
    )


def projected_costs(cost_per_patient: float) -> list[tuple[str, int, float]]:
    """Annual run-rate at the three deployment tiers, from the measured unit cost."""
    return [
        ("Tier A - 500 visits/day", 500 * 365, round(cost_per_patient * 500 * 365, 2)),
        ("Tier B - 250 visits/day", 250 * 365, round(cost_per_patient * 250 * 365, 2)),
        ("Tier C - 100 visits/day", 100 * 365, round(cost_per_patient * 100 * 365, 2)),
    ]
