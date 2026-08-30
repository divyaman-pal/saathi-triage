"""
Loads the Clinical Semantic Contract and its sibling policy files, and exposes
them as the single source of truth for every threshold in SAATHI.

No vital-sign number, cost ratio, SLA minute or entitlement appears anywhere
else in the Python source. If you find one, it is a bug.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from .models import AcquisitionMethod, QualityStatus, ThresholdBand

CONTRACT_DIR = Path(__file__).resolve().parent.parent / "contract"


def _load(name: str) -> dict[str, Any]:
    with open(CONTRACT_DIR / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class ContractError(Exception):
    pass


class Contract:
    """Facade over the whole contract bundle."""

    def __init__(self) -> None:
        self.clinical = _load("clinical_contract.yaml")
        self.red_flags = _load("red_flags.yaml")
        self.cost = _load("cost_policy.yaml")
        self.schemes = _load("acuity_schemes.yaml")
        self.entitlements = _load("entitlements.yaml")
        self.grammar = _load("claim_grammar.yaml")
        self.profiles = _load("deployment_profiles.yaml")

        self.version: str = self.clinical["contract_version"]
        self.grammar_version: str = self.grammar["grammar_version"]
        self.ruleset_version: str = self.red_flags["ruleset_version"]
        self.policy_version: str = self.cost["policy_version"]

        self._concepts: dict[str, dict] = self.clinical["concepts"]
        self._bands: list[dict] = self.clinical["age_bands"]
        self._methods: dict[str, dict] = self.clinical["acquisition_methods"]
        self._quality_metrics: dict[str, dict] = self.clinical["quality_metrics"]

    # -- concepts ----------------------------------------------------------

    def concept(self, concept_id: str) -> dict:
        try:
            return self._concepts[concept_id]
        except KeyError as exc:
            raise ContractError(
                f"Concept {concept_id!r} is not in the Clinical Semantic Contract. "
                "Every clinical quantity SAATHI computes on must be declared."
            ) from exc

    def has_concept(self, concept_id: str) -> bool:
        return concept_id in self._concepts

    @property
    def concept_ids(self) -> list[str]:
        return list(self._concepts.keys())

    def unit(self, concept_id: str) -> str | None:
        return self.concept(concept_id).get("unit")

    def max_staleness_minutes(self, concept_id: str) -> float:
        return float(self.concept(concept_id).get("max_staleness_minutes", 15))

    def refresh_cadence_seconds(self, concept_id: str) -> float | None:
        v = self.concept(concept_id).get("refresh_cadence_seconds")
        return float(v) if v is not None else None

    def materiality_threshold(self, concept_id: str) -> float | None:
        v = self.concept(concept_id).get("materiality_threshold")
        return float(v) if v is not None else None

    def materiality_class(self, concept_id: str) -> str | None:
        return self.concept(concept_id).get("materiality_class")

    def statistical_power(self, concept_id: str) -> str | None:
        return self.concept(concept_id).get("statistical_power")

    def contributes_to(self, concept_id: str) -> list[str]:
        return list(self.concept(concept_id).get("contributes_to", []))

    def pii_class(self, concept_id: str) -> str | None:
        return self.concept(concept_id).get("pii_class")

    # -- age bands ---------------------------------------------------------

    @functools.lru_cache(maxsize=256)
    def age_band(self, age_years: float) -> str:
        for band in self._bands:
            if band["min_years"] <= age_years < band["max_years"]:
                return band["id"]
        return self._bands[-1]["id"]

    def age_band_label(self, band_id: str) -> str:
        for band in self._bands:
            if band["id"] == band_id:
                return band["label"]
        return band_id

    def age_band_note(self, band_id: str) -> str:
        for band in self._bands:
            if band["id"] == band_id:
                return band.get("physiology_note", "")
        return ""

    @property
    def age_band_ids(self) -> list[str]:
        return [b["id"] for b in self._bands]

    def thresholds_for(self, concept_id: str, band_id: str) -> dict | None:
        tbl = self.concept(concept_id).get("age_band_thresholds") or {}
        if "_all" in tbl:
            return tbl["_all"]
        return tbl.get(band_id)

    def threshold_note(self, concept_id: str, band_id: str) -> str | None:
        t = self.thresholds_for(concept_id, band_id)
        return t.get("note") if t else None

    # -- acquisition methods ----------------------------------------------

    def reliability_weight(self, method: AcquisitionMethod | str) -> float:
        key = method.value if isinstance(method, AcquisitionMethod) else method
        m = self._methods.get(key)
        if m is None:
            raise ContractError(f"Acquisition method {key!r} is not declared in the contract.")
        return float(m["reliability_weight"])

    def quality_floor(self, method: AcquisitionMethod | str) -> dict | None:
        key = method.value if isinstance(method, AcquisitionMethod) else method
        m = self._methods.get(key)
        if m is None:
            raise ContractError(f"Acquisition method {key!r} is not declared in the contract.")
        return m.get("quality_floor")

    def method_note(self, method: AcquisitionMethod | str) -> str:
        key = method.value if isinstance(method, AcquisitionMethod) else method
        return self._methods.get(key, {}).get("note", "")

    def method_channel(self, method: AcquisitionMethod | str) -> str:
        key = method.value if isinstance(method, AcquisitionMethod) else method
        return self._methods.get(key, {}).get("channel", "system")

    def quality_band(self, metric: str, value: float) -> QualityStatus:
        spec = self._quality_metrics.get(metric)
        if spec is None:
            return QualityStatus.NOT_APPLICABLE
        bands = spec["bands"]
        if spec["higher_is_better"]:
            for name in ("GOOD", "ACCEPTABLE", "DEGRADED"):
                if value >= bands[name]:
                    return QualityStatus(name)
            return QualityStatus.FAILED
        for name in ("GOOD", "ACCEPTABLE", "DEGRADED"):
            if value <= bands[name]:
                return QualityStatus(name)
        return QualityStatus.FAILED

    # -- cost policy -------------------------------------------------------

    @property
    def cost_ratio(self) -> float:
        return float(self.cost["asymmetric_cost"]["ratio"])

    @property
    def cost_under(self) -> float:
        return float(self.cost["asymmetric_cost"]["cost_under_triage"])

    @property
    def cost_over(self) -> float:
        return float(self.cost["asymmetric_cost"]["cost_over_triage"])

    def sla_for(self, level: int) -> dict:
        return self.cost["sla"][f"level_{level}"]

    def recheck_modifier(self, name: str) -> float:
        return float(self.cost["recheck_modifiers"][name])

    @property
    def recheck_floor_minutes(self) -> float:
        return float(self.cost["recheck_modifiers"]["clamp_minimum_minutes"])

    def level_threshold(self, level: int) -> float:
        return float(self.cost["level_thresholds"][f"to_level_{level}"])

    @property
    def confidence_floors(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.cost["confidence_floors"].items() if isinstance(v, (int, float))}

    @property
    def surge(self) -> dict:
        return self.cost["surge"]

    @property
    def alert_budget(self) -> dict:
        return self.cost["alert_budget"]

    def materiality_action(self, statistical: str, materiality: str) -> str:
        for row in self.cost["materiality"]["matrix"]:
            if row["statistical"] == statistical and row["materiality"] == materiality:
                return row["action"]
        # Conservative default: anything unmatched with CRITICAL materiality escalates.
        return "ESCALATE" if materiality == "CRITICAL" else "LOG_ONLY"

    # -- abstention --------------------------------------------------------

    @property
    def abstention_gates(self) -> list[dict]:
        return self.clinical["abstention_policy"]["gates"]

    @property
    def abstention_on_abstain(self) -> dict:
        return self.clinical["abstention_policy"]["on_abstain"]

    # -- entitlements ------------------------------------------------------

    def role_spec(self, role: str) -> dict:
        r = self.entitlements["roles"].get(role)
        if r is None:
            raise ContractError(f"Role {role!r} is not declared. default_decision is DENY.")
        return r

    def resource_rule(self, role: str, resource: str) -> dict:
        spec = self.role_spec(role)["resources"]
        return spec.get(resource, {"access": self.entitlements["default_decision"].lower(), "scope": "none"})

    def field_class(self, class_name: str) -> dict:
        return self.entitlements["field_classes"][class_name]

    def field_class_of(self, field_name: str) -> str | None:
        for cname, spec in self.entitlements["field_classes"].items():
            if field_name in spec.get("fields", []):
                return cname
        return None

    @property
    def retention(self) -> dict:
        return self.entitlements["retention_policy"]

    # -- grammar -----------------------------------------------------------

    def epistemic_spec(self, state: str) -> dict:
        return self.grammar["epistemic_states"][state]

    @property
    def universally_forbidden(self) -> dict:
        return self.grammar["universally_forbidden"]

    @property
    def validator_checks(self) -> list[dict]:
        return self.grammar["validator_checks"]

    @property
    def attendant_vocabulary(self) -> dict:
        return self.grammar["attendant_allowed_vocabulary"]

    @property
    def fallback_policy(self) -> dict:
        return self.grammar["fallback_policy"]

    # -- deployment profiles ----------------------------------------------

    def profile(self, tier: str) -> dict:
        p = self.profiles["profiles"].get(tier)
        if p is None:
            raise ContractError(f"Deployment tier {tier!r} is not declared.")
        return p

    @property
    def tiers(self) -> list[str]:
        return list(self.profiles["profiles"].keys())

    @property
    def guaranteed_everywhere(self) -> list[str]:
        return self.profiles["guaranteed_in_every_tier"]

    # -- schemes -----------------------------------------------------------

    def scheme(self, scheme_id: str) -> dict:
        return self.schemes["schemes"][scheme_id]

    def mappings(self) -> list[dict]:
        return self.schemes["mappings"]


# ---------------------------------------------------------------------------
# Threshold evaluation - the age-band engine
# ---------------------------------------------------------------------------


def _in_range(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def evaluate_band(
    contract: Contract, concept_id: str, value: Any, age_band: str
) -> tuple[ThresholdBand, str | None]:
    """
    Classify a value against the contract's age-band thresholds.

    Returns (band, note). The note carries the clinical caveat for that band -
    for example the geriatric blunted-response warning - and is surfaced
    verbatim in the nurse UI so the reason for a band-specific decision is
    visible at the point of care.
    """
    spec = contract.thresholds_for(concept_id, age_band)
    if not spec:
        return ThresholdBand.UNKNOWN, None
    note = spec.get("note")

    crit = spec.get("critical") or {}
    if isinstance(crit, dict) and crit:
        if _matches_critical(value, crit):
            return ThresholdBand.CRITICAL, note

    conc = spec.get("concerning") or []
    for rng in conc:
        if _matches_concerning(value, rng):
            return ThresholdBand.CONCERNING, note

    norm = spec.get("normal")
    if norm is not None and _matches_normal(value, norm):
        return ThresholdBand.NORMAL, note

    # Declared thresholds exist but the value matched none of them. That is a
    # contract gap, not a normal value. Refuse to guess.
    return ThresholdBand.UNKNOWN, note


def _matches_critical(value: Any, crit: dict) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if "in" in crit and value in crit["in"]:
            return True
        if "eq" in crit and value == crit["eq"]:
            return True
        return False
    for op, ref in crit.items():
        if op == "gt" and value > ref:
            return True
        if op == "gte" and value >= ref:
            return True
        if op == "lt" and value < ref:
            return True
        if op == "lte" and value <= ref:
            return True
        if op == "eq" and value == ref:
            return True
        if op == "in" and value in ref:
            return True
    return False


def _matches_concerning(value: Any, rng: Any) -> bool:
    if isinstance(rng, (list, tuple)):
        if len(rng) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in rng):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return _in_range(value, rng[0], rng[1])
            return False
        return value in list(rng)
    return value == rng


def _matches_normal(value: Any, norm: Any) -> bool:
    if isinstance(norm, (list, tuple)):
        if len(norm) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in norm):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return _in_range(value, norm[0], norm[1])
            return False
        return value in list(norm)
    return value == norm


def band_severity(band: ThresholdBand) -> int:
    return {
        ThresholdBand.NORMAL: 0,
        ThresholdBand.UNKNOWN: 1,
        ThresholdBand.CONCERNING: 2,
        ThresholdBand.CRITICAL: 3,
    }[band]


def cross_band_explainer(contract: Contract, concept_id: str, value: float, bands: list[str]) -> str:
    """
    Produce the 'HR 148 - normal for age 2, critical for age 75' string that
    makes the age-stratification argument visible in one line.
    """
    parts = []
    for b in bands:
        band, _ = evaluate_band(contract, concept_id, value, b)
        parts.append(f"{contract.age_band_label(b)}: {band.value.lower()}")
    unit = contract.unit(concept_id) or ""
    return f"{concept_id} {value}{unit} -> " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_CONTRACT: Contract | None = None


def get_contract() -> Contract:
    global _CONTRACT
    if _CONTRACT is None:
        _CONTRACT = Contract()
    return _CONTRACT
