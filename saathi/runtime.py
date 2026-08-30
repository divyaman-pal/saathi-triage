"""
Runtime harness: builds the cohort, runs the pipeline over it, and holds the
results for the API and the UI.

One object so that the FastAPI app, the Streamlit UI, the test suite and the
evaluation scripts all exercise the SAME pipeline instance and the same audit
store. If the demo and the tests ran different code, neither would mean much.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .core.audit import AuditStore, get_store
from .core.contract_loader import Contract, get_contract
from .core.llm import LLMClient, get_llm
from .core.models import Assessment, Role, esi, reset_ids
from .core.pipeline import Pipeline, PipelineConfig, detect_surge
from .core.risk_model import RiskModel
from .core.models import AcquisitionMethod
from .data.cohort import PatientProfile, full_cohort, mandatory_cases, surge_fill
from .data.simulate import Simulator

DEMO_NOW = datetime(2026, 4, 12, 11, 31, 0)


@dataclass
class Runtime:
    tier: str = "TIER_A"
    now: datetime = DEMO_NOW
    surge_n: int = 42
    occupancy: float = 3.0
    contract: Contract = field(default_factory=get_contract)
    store: AuditStore = field(default_factory=get_store)
    llm: LLMClient = field(default_factory=get_llm)

    profiles: dict[str, PatientProfile] = field(default_factory=dict)
    assessments: dict[str, Assessment] = field(default_factory=dict)
    pipeline: Pipeline | None = None
    surge_active: bool = False
    surge_ratio: float = 1.0
    surge_message: str = ""

    def build(self, *, include_surge: bool = True, reset_store: bool = True) -> "Runtime":
        if reset_store:
            self.store.reset()
        reset_ids()
        model = RiskModel.load() if RiskModel.is_built() else RiskModel()
        self.pipeline = Pipeline(self.contract, model, self.store, self.llm)
        cases = full_cohort(self.surge_n, self.occupancy) if include_surge else mandatory_cases()
        self.profiles = {p.patient_id: p for p in cases}
        return self

    # -- surge -------------------------------------------------------------

    def evaluate_surge(self) -> None:
        arrivals = sum(1 for p in self.profiles.values() if p.arrival_minutes_ago <= 60)
        self.surge_active, self.surge_ratio, self.surge_message = detect_surge(
            self.contract, arrivals, self.tier)

    # -- assessment --------------------------------------------------------

    def assess_all(self, persona: Role = Role.TRIAGE_NURSE) -> dict[str, Assessment]:
        assert self.pipeline is not None, "call build() first"
        self.evaluate_surge()
        cfg = PipelineConfig(tier=self.tier, surge_active=self.surge_active, persona=persona)
        for pid, p in self.profiles.items():
            self.assessments[pid] = self._assess_one(p, cfg)
        return self.assessments

    def assess(self, patient_id: str, persona: Role = Role.TRIAGE_NURSE,
               *, config: PipelineConfig | None = None,
               elapsed_minutes: float | None = None) -> Assessment:
        assert self.pipeline is not None, "call build() first"
        p = self.profiles[patient_id]
        cfg = config or PipelineConfig(tier=self.tier, surge_active=self.surge_active, persona=persona)
        a = self._assess_one(p, cfg, elapsed_minutes=elapsed_minutes)
        self.assessments[patient_id] = a
        return a

    def _assess_one(self, p: PatientProfile, cfg: PipelineConfig,
                    elapsed_minutes: float | None = None) -> Assessment:
        assert self.pipeline is not None
        elapsed = p.arrival_minutes_ago if elapsed_minutes is None else elapsed_minutes
        sim = Simulator(self.contract, self.now, cfg.tier)
        evidence = sim.evidence_for(p, elapsed_minutes=elapsed)
        arrival_at = self.now - timedelta(minutes=elapsed)

        # Triage happened at ARRIVAL. If this is the patient's first assessment
        # and we are already well into their wait, establish the arrival acuity
        # from arrival-time evidence rather than from data that has since gone
        # stale. Without this, P-009 - whose nurse vitals are 41 minutes old -
        # would be assigned a level from an empty feature vector and then
        # "held" there by the abstention path.
        symptoms = self._extracted_symptoms(p, cfg.tier)
        if p.patient_id not in self.pipeline.states:
            arrival_sim = Simulator(self.contract, arrival_at, cfg.tier)
            self.pipeline.seed_arrival(
                p.patient_id, p.age_years, p.sex,
                arrival_sim.evidence_for(p, elapsed_minutes=1.0), arrival_at,
                has_prior_record=p.has_prior_record,
                complaint_text=(p.complaint_text if p.consent_given else ""),
                extracted_symptoms=symptoms,
            )
        return self.pipeline.assess(
            patient_id=p.patient_id, age_years=p.age_years, sex=p.sex,
            evidence=evidence, now=self.now, arrival_at=arrival_at,
            complaint_text=(p.complaint_text if p.consent_given else ""),
            asr_confidence=p.asr_confidence,
            attendant_present=p.attendant.present and p.consent_given,
            has_prior_record=p.has_prior_record,
            consent_declined=not p.consent_given,
            arrival_acuity_hint=None,
            extracted_symptoms=symptoms,
            config=cfg,
        )

    def _extracted_symptoms(self, p: PatientProfile, tier: str) -> list[str]:
        """
        Stand-in for the multilingual ASR + LLM symptom-extraction step.

        SIMULATED - we do not run speech recognition. What is REAL is the gate:
        the contract's ASR confidence floor is applied here, so a complaint
        captured below it produces NO structured symptoms and is marked
        partially unparsed. That is P-017 (Gondi, 0.41) and P-009 (Santali,
        0.34), and it is why they degrade differently.
        """
        if not p.consent_given:
            return []
        if not self.contract.profile(tier)["channels"]["nurse"].get("asr"):
            return []
        floor = (self.contract.quality_floor(AcquisitionMethod.ASR) or {}).get("min", 0.0)
        if p.asr_confidence < floor:
            return []
        return list(p.complaint_symptoms)

    # -- replay ------------------------------------------------------------

    def replay(self, patient_id: str, step_minutes: float = 4.0,
               persona: Role = Role.TRIAGE_NURSE) -> list[Assessment]:
        """
        Walk the wait forward in steps, so the arrival-vs-queue distinction is
        visible rather than asserted. This is what makes P-010 and P-014 legible:
        the arrival score was correct, and the patient changed afterwards.
        """
        assert self.pipeline is not None
        p = self.profiles[patient_id]
        self.pipeline.states.pop(patient_id, None)
        # Clear ONLY this patient's latched red flags. The latch is keyed by
        # patient and a fired flag persists for the encounter, so rebuilding the
        # whole latch object here would silently un-latch every OTHER patient in
        # the queue as a side effect of replaying one of them.
        self.pipeline.latch._latched.pop(patient_id, None)
        self.pipeline.latch._cleared.pop(patient_id, None)
        cfg = PipelineConfig(tier=self.tier, surge_active=self.surge_active, persona=persona)
        out: list[Assessment] = []
        t = 0.0
        while t <= p.arrival_minutes_ago + 1e-6:
            out.append(self._assess_one(p, cfg, elapsed_minutes=min(t, p.arrival_minutes_ago)))
            t += step_minutes
        if out and out[-1].sla.waited_minutes < p.arrival_minutes_ago - 1e-6:
            out.append(self._assess_one(p, cfg, elapsed_minutes=p.arrival_minutes_ago))
        self.assessments[patient_id] = out[-1]
        return out

    # -- queue -------------------------------------------------------------

    def queue(self) -> list[Assessment]:
        """
        Rank order. Acuity first (it is a clinical classification), then EWER
        within a level (it is an ordering over a queue). The two are different
        objects with different jobs and are never collapsed.
        """
        return sorted(self.assessments.values(),
                      key=lambda a: (a.acuity_current.level, -a.ewer.rank_value))

    def floor_summary(self) -> dict:
        qs = self.queue()
        breached = [a for a in qs if a.sla.breached]
        abstained = [a for a in qs if a.abstention.abstained]
        flagged = [a for a in qs if a.red_flags]
        escalated = [a for a in qs if a.acuity_current.level < a.acuity_arrival.level]
        by_level: dict[int, int] = {}
        for a in qs:
            by_level[a.acuity_current.level] = by_level.get(a.acuity_current.level, 0) + 1
        return {
            "n": len(qs),
            "by_level": dict(sorted(by_level.items())),
            "sla_breached": [a.patient_id for a in breached],
            "abstained": [a.patient_id for a in abstained],
            "red_flagged": [a.patient_id for a in flagged],
            "escalated_since_arrival": [a.patient_id for a in escalated],
            "surge_active": self.surge_active,
            "surge_ratio": round(self.surge_ratio, 2),
            "surge_message": self.surge_message,
            "alert_budget": (self.contract.alert_budget["surge"] if self.surge_active
                             else self.contract.alert_budget["normal"]),
        }


_RUNTIME: Runtime | None = None


def get_runtime(tier: str = "TIER_A", rebuild: bool = False) -> Runtime:
    global _RUNTIME
    if _RUNTIME is None or rebuild or _RUNTIME.tier != tier:
        _RUNTIME = Runtime(tier=tier).build()
        _RUNTIME.assess_all()
    return _RUNTIME
