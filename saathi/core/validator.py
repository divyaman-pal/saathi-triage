"""
The Claim Validator.

Re-checks rendered prose against the Safety Contract and the claim grammar, and
rejects anything outside it. Every rejection is logged with the offending text
retained for governance review.

A validator that has never fired is a validator nobody believes, so
tests/test_adversarial.py deliberately feeds it violating text of every class and
asserts each one is caught, and the demo shows a live catch.

TWO SEVERITIES
    STANDARD  -> reject and ask the renderer to try again (bounded retries)
    CRITICAL  -> reject hard, no retry, fall straight to the deterministic
                 template. Used for PII egress, abstention purity and the
                 attendant surface, where "try again" is not an acceptable
                 response to a breach.
"""

from __future__ import annotations

import re

from .contract_loader import Contract
from .models import EpistemicState, Role, SafetyContract, ValidationRejection

NUMBER_RE = re.compile(r"(?<![\w./-])(\d+(?:\.\d+)?)(?![\w/-])")
QUOTE_RE = re.compile(r"[\"'‘’“”]([^\"'‘’“”]{2,80})[\"'‘’“”]")

# Numbers that are always permitted: percentages of quality metrics, clock
# times, and small ordinals used structurally ("the 2 findings below").
STRUCTURAL_NUMBERS = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0, 100.0}


class Validator:
    def __init__(self, contract: Contract):
        self.c = contract
        self.forbidden = contract.universally_forbidden
        self.attendant_vocab = contract.attendant_vocabulary

    # -- helpers -----------------------------------------------------------

    def _quoted_spans(self, text: str) -> list[str]:
        return [m.group(1).lower() for m in QUOTE_RE.finditer(text)]

    def _lexicon_hits(self, text: str, category: str) -> list[str]:
        spec = self.forbidden.get(category, {})
        terms = spec.get("terms", [])
        low = text.lower()
        quoted = " | ".join(self._quoted_spans(text))
        hits = []
        for t in terms:
            if re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", low):
                # A diagnostic noun quoted verbatim from the patient's own words
                # is permitted inside an explicit quotation frame.
                if category == "diagnostic_nouns" and t in quoted:
                    continue
                hits.append(t)
        return hits

    # -- checks ------------------------------------------------------------

    def validate(self, text: str, sc: SafetyContract) -> list[ValidationRejection]:
        rejections: list[ValidationRejection] = []
        low = text.lower()

        # VC_PII_EGRESS (CRITICAL)
        from .rbac import PII_PATTERNS
        for pattern, _ in PII_PATTERNS:
            m = pattern.search(text)
            if m:
                rejections.append(ValidationRejection(
                    check_id="VC_PII_EGRESS", severity="CRITICAL",
                    detail="Identifier-shaped token in rendered output.",
                    offending_text=m.group(0)))

        # VC_PERSONA_SCOPE (CRITICAL) - allow-list, not deny-list
        if sc.persona is Role.ATTENDANT:
            banned = ["acuity", "level 1", "level 2", "level 3", "score", "risk",
                      "critical", "urgent", "serious", "danger", "deteriorat",
                      "worse", "queue", "position", "priority", "esi", "%"]
            for b in banned:
                if b in low:
                    rejections.append(ValidationRejection(
                        check_id="VC_PERSONA_SCOPE", severity="CRITICAL",
                        detail=(f"Attendant-facing text contains {b!r}. The attendant surface is an "
                                "allow-list: no acuity, no score, no risk language, no queue position."),
                        offending_text=b))
            if NUMBER_RE.search(re.sub(r"\b\d{1,2}[:.]\d{2}\b", "", text)):
                rejections.append(ValidationRejection(
                    check_id="VC_PERSONA_SCOPE", severity="CRITICAL",
                    detail="Attendant-facing text contains a number describing the patient's condition.",
                    offending_text=text[:120]))

        # VC_ABSTENTION_PURITY (CRITICAL)
        if sc.epistemic_status is EpistemicState.INSUFFICIENT_SIGNAL:
            for t in self.forbidden["risk_words_for_abstention"]["terms"]:
                if re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", low):
                    rejections.append(ValidationRejection(
                        check_id="VC_ABSTENTION_PURITY", severity="CRITICAL",
                        detail=("An abstaining assessment may not carry directional risk language. "
                                "Saying 'probably low risk, but the signal is poor' converts an "
                                "admission of blindness into an assessment."),
                        offending_text=t))

        # VC_FORBIDDEN_LEXICON (STANDARD, except reassurance which is CRITICAL)
        for category in ("diagnostic_nouns", "causal_verbs", "certainty_verbs",
                         "prognostic_phrases", "reassurance_phrases", "treatment_verbs"):
            for hit in self._lexicon_hits(text, category):
                rejections.append(ValidationRejection(
                    check_id="VC_FORBIDDEN_LEXICON",
                    severity="CRITICAL" if category == "reassurance_phrases" else "STANDARD",
                    detail=(f"Forbidden {category.replace('_', ' ')}: {hit!r}. "
                            + self.forbidden[category].get("rationale", "").strip()[:220]),
                    offending_text=hit))

        # VC_NUMBER_WHITELIST
        allowed = set(sc.allowed_numbers) | STRUCTURAL_NUMBERS
        stripped = re.sub(r"\b\d{1,2}[:.]\d{2}\b", " ", text)          # clock times
        stripped = re.sub(r"\bv?\d+\.\d+(\.\d+)?\b", " ", stripped)    # version strings
        for m in NUMBER_RE.finditer(stripped):
            val = float(m.group(1))
            if val not in allowed and round(val) not in allowed:
                rejections.append(ValidationRejection(
                    check_id="VC_NUMBER_WHITELIST", severity="STANDARD",
                    detail=(f"Number {m.group(1)} is not in the Safety Contract's allowed_numbers. "
                            "Every figure shown to a clinician must trace to an evidence value."),
                    offending_text=m.group(0)))

        # VC_LENGTH
        words = len(text.split())
        if words > sc.max_words:
            rejections.append(ValidationRejection(
                check_id="VC_LENGTH", severity="STANDARD",
                detail=(f"{words} words against a {sc.max_words} ceiling for state "
                        f"{sc.epistemic_status.value}. A nurse view that needs scrolling has failed."),
                offending_text=f"{words} words"))

        # VC_DISCLOSURE_PRESENT
        keywords = {
            "prior_record_missing": ["prior record", "no record", "first presentation"],
            "observation_only_model": ["observation-only", "observation only", "reduced feature"],
            "insufficient_signal": ["insufficient", "cannot"],
            "escalated_on_materiality_not_model": ["clinical significance", "model did not",
                                                   "not on model", "materiality"],
            "uncalibrated_subgroup": ["calibrat"],
            "stale_data_present": ["stale", "min old", "validity", "past its"],
        }
        for d in sc.required_disclosures:
            keys = keywords.get(d)
            if keys and not any(k in low for k in keys):
                rejections.append(ValidationRejection(
                    check_id="VC_DISCLOSURE_PRESENT", severity="STANDARD",
                    detail=f"Required disclosure {d!r} is absent from the rendered text.",
                    offending_text=d))

        # VC_CLAIM_TYPE - attribution language where the state forbids it
        if "attribution" in sc.forbidden_claim_types:
            for phrase in ("contributed", "drove the", "largest contributor", "accounted for",
                           "was the main factor"):
                if phrase in low:
                    rejections.append(ValidationRejection(
                        check_id="VC_CLAIM_TYPE", severity="STANDARD",
                        detail=(f"Attribution language {phrase!r} is forbidden in state "
                                f"{sc.epistemic_status.value}. Presenting one channel's "
                                "contribution as the explanation hides the disagreement."),
                        offending_text=phrase))
        if "probability_statement" in sc.forbidden_claim_types and "%" in text:
            rejections.append(ValidationRejection(
                check_id="VC_CLAIM_TYPE", severity="STANDARD",
                detail=f"Probability language is forbidden in state {sc.epistemic_status.value}.",
                offending_text="%"))

        return rejections

    def is_critical(self, rejections: list[ValidationRejection]) -> bool:
        return any(r.severity == "CRITICAL" for r in rejections)


def summarise(rejections: list[ValidationRejection]) -> str:
    if not rejections:
        return "PASSED - no claim outside the Safety Contract."
    return " | ".join(f"{r.check_id}({r.severity}): {r.offending_text}" for r in rejections)
