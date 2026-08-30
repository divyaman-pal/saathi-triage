"""
Cross-scheme acuity conversion.

The only way to turn an ESI value into an MTS, CTAS or local three-colour value.
Every conversion consults the versioned mapping table, records what is lost, and
REFUSES the edges registered as UNSAFE rather than returning a best guess.

This is what stops "ESI 2", "MTS Orange", "CTAS 2", "urgent" and the local
hospital's "Red" from being silently treated as the same label.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contract_loader import Contract
from .models import Acuity, SchemeMappingRefused


@dataclass
class Conversion:
    source: Acuity
    target: Acuity
    fidelity: str
    lost_in_translation: str
    mapping_version: str

    def display(self) -> str:
        return (
            f"{self.source} -> {self.target}  [{self.fidelity}]  "
            f"(mapping v{self.mapping_version})"
        )


def convert(contract: Contract, value: Acuity, to_scheme_id: str, to_scheme_version: str | None = None) -> Conversion:
    """Convert an acuity between schemes, or refuse."""
    if to_scheme_version is None:
        to_scheme_version = contract.scheme(to_scheme_id)["scheme_version"]

    if (value.scheme_id, value.scheme_version) == (to_scheme_id, to_scheme_version):
        return Conversion(value, value, "EXACT", "", "n/a")

    for m in contract.mappings():
        src, dst = m["from"], m["to"]
        if (
            src["scheme_id"] == value.scheme_id
            and str(src["scheme_version"]) == str(value.scheme_version)
            and dst["scheme_id"] == to_scheme_id
            and str(dst["scheme_version"]) == str(to_scheme_version)
        ):
            if m.get("refuse"):
                raise SchemeMappingRefused(
                    f"Conversion {value.scheme_id}/{value.scheme_version} -> "
                    f"{to_scheme_id}/{to_scheme_version} is registered UNSAFE and is refused.\n"
                    f"Reason: {m['refusal_reason'].strip()}"
                )
            table = m["map"]
            if value.level not in table:
                raise SchemeMappingRefused(
                    f"Level {value.level} has no declared mapping into {to_scheme_id}."
                )
            return Conversion(
                source=value,
                target=Acuity(
                    scheme_id=to_scheme_id, scheme_version=str(to_scheme_version), level=table[value.level]
                ),
                fidelity=m["fidelity"],
                lost_in_translation=m.get("lost_in_translation", "").strip(),
                mapping_version=m.get("version", "?"),
            )

    raise SchemeMappingRefused(
        f"No declared mapping from {value.scheme_id}/{value.scheme_version} to "
        f"{to_scheme_id}/{to_scheme_version}. Unregistered conversions are refused, "
        "not improvised."
    )


def describe_scheme(contract: Contract, scheme_id: str) -> str:
    s = contract.scheme(scheme_id)
    return f"{s['name']} ({s['scheme_id']}/{s['scheme_version']}, {s.get('origin','-')})"


def monotonic_check(previous: Acuity, current: Acuity, *, human_override: bool = False) -> None:
    """
    INVARIANT 1: SAATHI raises urgency and never lowers it.

    Raises on violation. Called at every point where an acuity is updated, and
    additionally enforced structurally by SafetyContract's model validator.
    """
    if previous.scheme_id != current.scheme_id or previous.scheme_version != current.scheme_version:
        raise SchemeMappingRefused(
            "Monotonicity cannot be evaluated across schemes. Convert explicitly first."
        )
    if human_override:
        return
    if current.level > previous.level:
        raise ValueError(
            f"MONOTONIC ESCALATION VIOLATION: {previous} -> {current}. "
            "Only a human, with a recorded reason, may reduce urgency."
        )
