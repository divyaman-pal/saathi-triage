"""
SAATHI core data model.

These Pydantic models are the executable specification. Anything that cannot be
expressed here does not exist in the system. Two properties are enforced
structurally rather than by convention:

  1. An acuity value is ALWAYS a (scheme_id, scheme_version, level) triple.
     A bare integer acuity cannot be constructed. This is what stops an ESI 2,
     an MTS Orange and a local "Red" from being accidentally compared.

  2. A clinical value is ALWAYS accompanied by its signal quality, its
     freshness, its acquisition method and its lineage. There is no way to
     construct an Evidence object that carries a number without carrying the
     epistemics of that number.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Channel(str, Enum):
    NURSE = "nurse"
    CAMERA = "camera"
    ATTENDANT = "attendant"
    PRIOR_RECORD = "prior_record"
    SYSTEM = "system"


class AcquisitionMethod(str, Enum):
    MANUAL_COUNT = "manual_count"
    CUFF = "cuff"
    CLINICAL_OBSERVATION = "clinical_observation"
    RPPG_HR = "rppg_hr"
    RPPG_RR = "rppg_rr"
    POSE_ESTIMATION = "pose_estimation"
    GESTALT_ARRIVAL = "gestalt_arrival"
    GUIDED_COUNT_15S = "guided_count_15s"
    PROXY_REPORT = "proxy_report"
    SELF_REPORT = "self_report"
    ASR = "asr"
    RECORD_LOOKUP = "record_lookup"
    DETERMINISTIC = "deterministic"


class QualityStatus(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FreshnessStatus(str, Enum):
    CURRENT = "CURRENT"
    AGING = "AGING"
    STALE = "STALE"
    ABSENT = "ABSENT"


class ChannelAvailability(str, Enum):
    """The three states that must never be collapsed into 'no data'."""

    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    SILENT = "SILENT"
    ABSENT = "ABSENT"


class ThresholdBand(str, Enum):
    NORMAL = "NORMAL"
    CONCERNING = "CONCERNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class EpistemicState(str, Enum):
    RED_FLAG_FIRED = "RED_FLAG_FIRED"
    HIGH_CONFIDENCE_ESCALATION = "HIGH_CONFIDENCE_ESCALATION"
    MODERATE_CONFIDENCE_ESCALATION = "MODERATE_CONFIDENCE_ESCALATION"
    DISCORDANT_CHANNELS = "DISCORDANT_CHANNELS"
    MATERIALITY_ESCALATION = "MATERIALITY_ESCALATION"
    OBSERVATION_ONLY = "OBSERVATION_ONLY"
    INSUFFICIENT_SIGNAL = "INSUFFICIENT_SIGNAL"
    STABLE_NO_CHANGE = "STABLE_NO_CHANGE"
    HUMAN_OVERRIDE_APPLIED = "HUMAN_OVERRIDE_APPLIED"


class Role(str, Enum):
    TRIAGE_NURSE = "triage_nurse"
    ED_PHYSICIAN = "ed_physician"
    ATTENDANT = "attendant"
    ADMINISTRATOR = "administrator"


class MaterialityClass(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class StatisticalStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


class SupportDirection(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Acuity - the anti-conflation type
# ---------------------------------------------------------------------------


class Acuity(BaseModel):
    """
    A triage acuity value. Never a bare integer.

    Comparison between two Acuity values of different schemes raises rather than
    silently coercing. Cross-scheme conversion must go through
    core.schemes.convert(), which consults the versioned, declared-lossy mapping
    table and refuses UNSAFE edges.
    """

    model_config = ConfigDict(frozen=True)

    scheme_id: str
    scheme_version: str
    level: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.scheme_id}/{self.scheme_version} L{self.level}"

    def _assert_same_scheme(self, other: "Acuity") -> None:
        if (self.scheme_id, self.scheme_version) != (other.scheme_id, other.scheme_version):
            raise SchemeMismatchError(
                f"Refusing to compare {self} with {other}. "
                "Cross-scheme comparison must go through core.schemes.convert()."
            )

    def is_more_urgent_than(self, other: "Acuity") -> bool:
        self._assert_same_scheme(other)
        return self.level < other.level

    def is_less_urgent_than(self, other: "Acuity") -> bool:
        self._assert_same_scheme(other)
        return self.level > other.level


class SchemeMismatchError(Exception):
    """Raised when two acuity values from different schemes are compared."""


class SchemeMappingRefused(Exception):
    """Raised when a cross-scheme conversion is registered as UNSAFE."""


def esi(level: int) -> Acuity:
    """Convenience constructor for the native scheme."""
    return Acuity(scheme_id="ESI", scheme_version="v4", level=level)


# ---------------------------------------------------------------------------
# Signal quality and freshness
# ---------------------------------------------------------------------------


class SignalQuality(BaseModel):
    """
    Quality of the acquisition, not quality of the model.

    `passed_floor` is the load-bearing field. A value whose quality is below the
    contract's floor for its acquisition method is REJECTED - excluded from the
    feature set entirely - rather than down-weighted. Down-weighting a value we
    do not believe still lets it move the score.
    """

    snr_db: float | None = None
    occlusion_pct: float | None = None
    asr_confidence: float | None = None
    motion_index: float | None = None
    status: QualityStatus = QualityStatus.NOT_APPLICABLE
    passed_floor: bool = True
    floor_detail: str | None = None

    def summary(self) -> str:
        parts: list[str] = []
        if self.snr_db is not None:
            parts.append(f"SNR {self.snr_db:.1f}")
        if self.occlusion_pct is not None:
            parts.append(f"occl {self.occlusion_pct:.0f}%")
        if self.asr_confidence is not None:
            parts.append(f"ASR {self.asr_confidence:.2f}")
        if self.motion_index is not None:
            parts.append(f"motion {self.motion_index:.2f}")
        return f"{self.status.value}" + (f" ({', '.join(parts)})" if parts else "")


class Freshness(BaseModel):
    age_seconds: float
    max_staleness_minutes: float
    status: FreshnessStatus
    expected_refresh_seconds: float | None = None

    @property
    def age_minutes(self) -> float:
        return self.age_seconds / 60.0

    def display(self) -> str:
        if self.status is FreshnessStatus.ABSENT:
            return "absent"
        m = self.age_minutes
        if m < 1:
            return f"{int(self.age_seconds)}s ago"
        return f"{int(m)} min ago"


# ---------------------------------------------------------------------------
# Evidence - the atom of the system
# ---------------------------------------------------------------------------


class Contribution(BaseModel):
    """
    How much this evidence moved the model output.

    `method` is mandatory and constrains the language the claim compiler may
    use downstream. A 'shap' contribution licenses an ATTRIBUTION claim. It
    never licenses a CAUSATION claim. A 'rule' contribution licenses an
    imperative. A 'materiality' contribution licenses a materiality statement
    and explicitly forbids attribution language.
    """

    method: Literal["shap", "rule", "materiality", "trajectory", "discordance", "sla", "none"] = "none"
    value: float = 0.0
    direction: Literal["escalating", "de-escalating", "neutral"] = "neutral"
    population_note: str | None = None


class ConfidenceComponents(BaseModel):
    """
    Five named, inspectable axes. NEVER collapsed into a single displayed number.

        signal quality != model confidence != clinical certainty != escalation priority

    `overall()` exists only for internal gating and ranking. It is deliberately
    not rendered to any clinical user without its components alongside.
    """

    signal_quality: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    applicability: float = Field(ge=0.0, le=1.0)
    channel_agreement: float = Field(ge=0.0, le=1.0)
    calibration_status: str = "UNCALIBRATED_SUBGROUP"

    def overall(self) -> float:
        """Weakest-link aggregate. Internal gating only."""
        return min(self.signal_quality, self.completeness, self.applicability)

    def failing(self, floors: dict[str, float]) -> list[str]:
        out = []
        for name in ("signal_quality", "completeness", "applicability", "channel_agreement"):
            if name in floors and getattr(self, name) < floors[name]:
                out.append(name)
        return out


class Evidence(BaseModel):
    """One observation, with everything needed to defend it."""

    evidence_id: str
    patient_id: str
    source_channel: Channel
    acquisition_method: AcquisitionMethod
    device_id: str | None = None

    concept_id: str
    value: Any
    unit: str | None = None

    observation_window: tuple[datetime, datetime]
    grain: str

    signal_quality: SignalQuality = Field(default_factory=SignalQuality)
    freshness: Freshness | None = None
    reliability_weight: float = 1.0

    contribution: Contribution = Field(default_factory=Contribution)
    confidence_components: ConfidenceComponents | None = None
    supports_or_contradicts: SupportDirection = SupportDirection.NEUTRAL

    threshold_band: ThresholdBand = ThresholdBand.UNKNOWN
    age_band_context: str | None = None

    access_classification: str = "CLINICAL_DERIVED"
    model_version: str | None = None
    contract_version: str | None = None
    lineage_ref: str | None = None

    @property
    def observed_at(self) -> datetime:
        return self.observation_window[1]

    @property
    def usable(self) -> bool:
        """Passed its quality floor AND is not stale."""
        if not self.signal_quality.passed_floor:
            return False
        if self.freshness is not None and self.freshness.status is FreshnessStatus.STALE:
            return False
        return True

    def display_line(self) -> str:
        fresh = self.freshness.display() if self.freshness else "-"
        return (
            f"{self.concept_id} = {self.value}{(' ' + self.unit) if self.unit else ''} "
            f"[{self.source_channel.value}/{self.acquisition_method.value}] "
            f"{fresh} q={self.signal_quality.status.value}"
        )


# ---------------------------------------------------------------------------
# Red flags, materiality, SLA
# ---------------------------------------------------------------------------


class RedFlagHit(BaseModel):
    rule_id: str
    name: str
    trigger_human: str
    target_acuity: Acuity
    fired_on_evidence: list[str] = Field(default_factory=list)
    source_channels: list[Channel] = Field(default_factory=list)
    rationale: str = ""
    ruleset_version: str = ""
    suppressible_by_model: Literal[False] = False


class MaterialityFinding(BaseModel):
    concept_id: str
    statistical_strength: StatisticalStrength
    materiality_class: MaterialityClass
    action: Literal["ESCALATE", "RECHECK", "LOG_ONLY", "LOG_DO_NOT_ALERT"]
    detail: str
    evidence_ids: list[str] = Field(default_factory=list)


class SLAStatus(BaseModel):
    acuity_level: int
    waited_minutes: float
    max_wait_minutes: float
    breached: bool
    recheck_interval_minutes: float
    recheck_due_in_minutes: float
    modifiers_applied: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# EWER - Evidence-Weighted Escalation Rank
# ---------------------------------------------------------------------------


class EWERComponents(BaseModel):
    """
    A transparent engineering ranking of who deserves human attention next.

    It is NOT a probability, NOT a clinical certainty, NOT a diagnosis and NOT a
    severity level in any published scheme. It is never rendered as a percentage
    and never as a triage level. The components are always shown.
    """

    model_risk: float = 0.0
    confidence_penalty: float = 0.0
    discordance: float = 0.0
    minutes_since_human_contact: float = 0.0
    sla_pressure: float = 0.0
    red_flag_boost: float = 0.0
    trajectory: float = 0.0
    materiality_boost: float = 0.0

    rank_value: float = 0.0

    def breakdown(self) -> list[tuple[str, float]]:
        return [
            ("model risk estimate", self.model_risk),
            ("confidence penalty", self.confidence_penalty),
            ("channel discordance", self.discordance),
            ("time since human contact", self.minutes_since_human_contact),
            ("wait vs SLA", self.sla_pressure),
            ("red flag", self.red_flag_boost),
            ("trajectory", self.trajectory),
            ("clinical materiality", self.materiality_boost),
        ]


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


class CostDecision(BaseModel):
    """
    The escalate-or-hold decision, and what the accuracy-optimised threshold
    would have done with the same probability.

    `decision_asymmetric` and `decision_symmetric` are BINARY (1 = escalate,
    0 = hold). They are not acuity levels. The model estimates one thing -
    P(this patient needs escalation during the wait) - and a single probability
    cannot be honestly fanned out into five ordered clinical classes. Arrival
    acuity is assigned separately and deterministically by core/triage_rules.py.
    """

    probability: float
    threshold_used: float
    threshold_symmetric: float
    cost_under: float
    cost_over: float
    ratio: float
    decision_asymmetric: int            # 1 = escalate, 0 = hold
    decision_symmetric: int             # what a 0.5 threshold would have done
    target_level_asymmetric: int
    target_level_symmetric: int
    expected_cost_escalate: float
    expected_cost_hold: float
    policy_version: str

    @property
    def disagrees_with_symmetric(self) -> bool:
        return self.decision_asymmetric != self.decision_symmetric


class AbstentionResult(BaseModel):
    abstained: bool
    gates_tripped: list[str] = Field(default_factory=list)
    failing_signals: list[str] = Field(default_factory=list)
    missing_channels: list[str] = Field(default_factory=list)
    stale_items: list[str] = Field(default_factory=list)
    priority_question: str | None = None


# ---------------------------------------------------------------------------
# Safety Contract - generated BEFORE the LLM is called
# ---------------------------------------------------------------------------


class SafetyContract(BaseModel):
    """
    The machine-readable decision. The LLM renders this; it never authors it.

    Everything in this object is produced by deterministic code. If the LLM is
    switched off, this object is byte-identical and the deterministic template
    renderer produces the text instead.
    """

    contract_id: str
    patient_id: str
    persona: Role
    access_scope: list[str]

    acuity_current: Acuity
    acuity_previous: Acuity | None
    acuity_arrival: Acuity
    change_reason: str
    escalation_only: bool = True

    epistemic_status: EpistemicState

    evidence_ids: list[str] = Field(default_factory=list)
    contradictory_evidence_ids: list[str] = Field(default_factory=list)
    red_flags_fired: list[str] = Field(default_factory=list)
    materiality_findings: list[str] = Field(default_factory=list)

    confidence: ConfidenceComponents
    cost_decision: CostDecision | None = None
    ewer: EWERComponents | None = None
    sla: SLAStatus | None = None

    missing_data: list[str] = Field(default_factory=list)
    stale_data: list[str] = Field(default_factory=list)
    degraded_channels: list[str] = Field(default_factory=list)
    absent_channels: list[str] = Field(default_factory=list)

    allowed_claim_types: list[str] = Field(default_factory=list)
    forbidden_claim_types: list[str] = Field(default_factory=list)
    allowed_numbers: list[float] = Field(default_factory=list)
    allowed_entities: list[str] = Field(default_factory=list)
    required_disclosures: list[str] = Field(default_factory=list)
    max_words: int = 90

    priority_question: str | None = None
    physical_check_instruction: str | None = None

    lineage_ref: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    contract_version: str
    grammar_version: str
    generated_at: datetime

    @model_validator(mode="after")
    def _escalation_only_invariant(self) -> "SafetyContract":
        """
        INVARIANT 1 (monotonic escalation), enforced at construction.

        A SafetyContract with escalation_only=True cannot be built with a
        current acuity less urgent than the arrival acuity. The only way to
        produce a downgrade is a HUMAN_OVERRIDE_APPLIED contract, which sets
        escalation_only=False and carries an override record.
        """
        if self.escalation_only and self.acuity_current.level > self.acuity_arrival.level:
            raise ValueError(
                f"MONOTONIC ESCALATION VIOLATION: contract {self.contract_id} "
                f"would move {self.patient_id} from arrival L{self.acuity_arrival.level} "
                f"to L{self.acuity_current.level} (less urgent) without a human override."
            )
        return self

    @model_validator(mode="after")
    def _abstention_purity(self) -> "SafetyContract":
        """
        INVARIANT: an abstaining contract may not carry a risk number for the
        renderer to leak. allowed_numbers is restricted to quality/staleness
        figures upstream; here we assert no cost decision rides along.
        """
        if self.epistemic_status is EpistemicState.INSUFFICIENT_SIGNAL:
            if self.cost_decision is not None:
                raise ValueError(
                    "ABSTENTION PURITY VIOLATION: an INSUFFICIENT_SIGNAL contract "
                    "must not carry a cost decision - there is nothing to decide from."
                )
        return self


# ---------------------------------------------------------------------------
# Rendered output
# ---------------------------------------------------------------------------


class ValidationRejection(BaseModel):
    check_id: str
    detail: str
    offending_text: str
    severity: Literal["CRITICAL", "STANDARD"] = "STANDARD"


class RenderedNarrative(BaseModel):
    text: str
    renderer: Literal["llm", "deterministic_template"]
    attempts: int = 1
    rejections: list[ValidationRejection] = Field(default_factory=list)
    llm_enabled: bool = True
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    model_id: str | None = None


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------


class OverrideRecord(BaseModel):
    override_id: str
    patient_id: str
    clinician_id: str
    role: Role
    timestamp: datetime

    system_acuity: Acuity
    clinician_acuity: Acuity
    direction: Literal["up", "down"]

    reason_code: str
    free_text: str | None = None

    evidence_shown_at_time: list[str] = Field(default_factory=list)
    safety_contract_id: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    contract_version: str
    time_from_display_to_override_ms: float = 0.0
    outcome_link: str | None = None

    @model_validator(mode="after")
    def _direction_matches(self) -> "OverrideRecord":
        expected = "down" if self.clinician_acuity.level > self.system_acuity.level else "up"
        if self.system_acuity.level == self.clinician_acuity.level:
            raise ValueError("An override must change the acuity.")
        if self.direction != expected:
            raise ValueError(f"Override direction {self.direction} contradicts the acuity change.")
        return self


# ---------------------------------------------------------------------------
# Patient-facing assessment bundle
# ---------------------------------------------------------------------------


class ChannelStatus(BaseModel):
    channel: Channel
    availability: ChannelAvailability
    last_observation_at: datetime | None = None
    items_this_window: int = 0
    quality: QualityStatus = QualityStatus.NOT_APPLICABLE
    note: str | None = None


class Assessment(BaseModel):
    """Everything the pipeline produced for one patient at one instant."""

    patient_id: str
    display_name_available: bool = False
    age_years: float
    age_band: str
    sex: str
    tier: str

    assessed_at: datetime
    arrival_at: datetime

    acuity_arrival: Acuity
    acuity_previous: Acuity | None
    acuity_current: Acuity
    acuity_change_reason: str

    epistemic_state: EpistemicState
    confidence: ConfidenceComponents
    cost_decision: CostDecision | None
    abstention: AbstentionResult
    ewer: EWERComponents
    sla: SLAStatus

    red_flags: list[RedFlagHit] = Field(default_factory=list)
    materiality: list[MaterialityFinding] = Field(default_factory=list)

    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradictory_evidence: list[Evidence] = Field(default_factory=list)

    channel_status: list[ChannelStatus] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    stale_data: list[str] = Field(default_factory=list)

    model_route: str = "FULL_v1"
    model_versions: dict[str, str] = Field(default_factory=dict)

    safety_contract: SafetyContract | None = None
    narrative: RenderedNarrative | None = None
    lineage_ref: str = ""

    override: OverrideRecord | None = None
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)

    # Latest quality-passed value per vital, for the at-a-glance queue row.
    # Kept SEPARATE from supporting_evidence, which is a ranked explanation and
    # deliberately truncated - a nurse scanning a worklist needs the current
    # numbers whether or not they happened to make the top of the reasoning.
    # Each entry: {value, unit, band, age_seconds, quality, method, critical}.
    vitals: dict[str, dict[str, Any]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------


_COUNTERS: dict[str, int] = {}


def next_id(prefix: str, salt: str = "") -> str:
    _COUNTERS[prefix] = _COUNTERS.get(prefix, 0) + 1
    n = _COUNTERS[prefix]
    if salt:
        h = hashlib.sha256(f"{prefix}{salt}{n}".encode()).hexdigest()[:6].upper()
        return f"{prefix}-{n:06d}-{h}"
    return f"{prefix}-{n:06d}"


def reset_ids() -> None:
    _COUNTERS.clear()


def window(end: datetime, seconds: float) -> tuple[datetime, datetime]:
    return (end - timedelta(seconds=seconds), end)
