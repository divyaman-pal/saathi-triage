"""
Decision Cost Engine.

The operating point is DERIVED FROM COST, not tuned for accuracy.

    escalate  iff  p * C_under  >  (1 - p) * C_over
              iff  p  >  C_over / (C_under + C_over)

At the contract's 12:1 ratio that is p > 0.0769, not p > 0.5. The difference
between those two numbers is the difference between a triage assistant and a
liability, and it is a POLICY choice owned by a named clinician - not a library
default hidden inside `predict()`.

Everything here reads from contract/cost_policy.yaml. Changing the ratio changes
every threshold in the system, is versioned, and is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract_loader import Contract
from .models import CostDecision


class CostEngine:
    def __init__(self, contract: Contract):
        self.c = contract
        self.cost_under = contract.cost_under
        self.cost_over = contract.cost_over
        self.ratio = contract.cost_ratio
        self.policy_version = contract.policy_version

    # -- thresholds --------------------------------------------------------

    @property
    def derived_threshold(self) -> float:
        """The cost-minimising threshold. Pure arithmetic on the policy."""
        return self.cost_over / (self.cost_under + self.cost_over)

    def threshold_for_level(self, level: int) -> float:
        return self.c.level_threshold(level)

    def acuity_from_probability(self, p: float) -> int:
        """
        Map a calibrated probability to an ESI level using cost-derived
        thresholds. Higher acuity has a LOWER threshold - it is easier to
        escalate INTO a more urgent level, which is the asymmetry made concrete.
        """
        if p >= self.c.level_threshold(1):
            return 1
        if p >= self.c.level_threshold(2):
            return 2
        if p >= self.c.level_threshold(3):
            return 3
        if p >= self.c.level_threshold(4):
            return 4
        return 5

    def acuity_symmetric(self, p: float) -> int:
        """
        The same model at a symmetric 0.5 threshold, for side-by-side display.

        This comparison is the single most persuasive artefact in the system:
        it shows, in cases rather than percentages, exactly which patients the
        accuracy-optimised version leaves in the waiting room.
        """
        if p >= 0.85:
            return 1
        if p >= 0.50:
            return 2
        if p >= 0.25:
            return 3
        if p >= 0.10:
            return 4
        return 5

    # -- decision ----------------------------------------------------------

    def decide(self, probability: float, current_level: int, *, level: int = 2) -> CostDecision:
        """
        Escalate or hold, at the cost-derived threshold - and, alongside it,
        what an accuracy-optimised 0.5 threshold would have decided with the
        very same probability.

        Both are computed on every patient so the comparison is a property of
        the running system rather than a chart made once for a slide.
        """
        thr = self.threshold_for_level(level)
        p = float(probability)
        esc_a = int(p >= thr)
        esc_s = int(p >= 0.5)
        from .triage_rules import escalation_target
        return CostDecision(
            probability=round(p, 4),
            threshold_used=thr,
            threshold_symmetric=0.5,
            cost_under=self.cost_under,
            cost_over=self.cost_over,
            ratio=self.ratio,
            decision_asymmetric=esc_a,
            decision_symmetric=esc_s,
            target_level_asymmetric=escalation_target(current_level) if esc_a else current_level,
            target_level_symmetric=escalation_target(current_level) if esc_s else current_level,
            expected_cost_escalate=round((1 - p) * self.cost_over, 4),
            expected_cost_hold=round(p * self.cost_under, 4),
            policy_version=self.policy_version,
        )

    def explain(self, d: CostDecision) -> str:
        return (
            f"P(needs escalation) = {d.probability:.3f}. "
            f"Escalating a patient who did not need it costs {d.cost_over:g}; "
            f"holding a patient who did costs {d.cost_under:g}. "
            f"Expected cost of escalating = (1 - {d.probability:.3f}) x {d.cost_over:g} = {d.expected_cost_escalate:.3f}; "
            f"expected cost of holding = {d.probability:.3f} x {d.cost_under:g} = {d.expected_cost_hold:.3f}. "
            f"Threshold {d.threshold_used:.4f} (cost-derived, policy v{d.policy_version}), "
            f"not 0.5 (accuracy-optimised). "
            + ("The two thresholds DISAGREE on this patient: the cost-derived one escalates, "
               "the accuracy-optimised one would hold."
               if d.disagrees_with_symmetric else
               "Both thresholds agree on this patient.")
        )

    # -- analysis ----------------------------------------------------------

    def net_benefit(self, y_true: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        Decision-curve analysis. Net benefit at threshold p_t:

            NB = TP/n - (FP/n) * (p_t / (1 - p_t))

        The odds term is the exchange rate between a false positive and a true
        positive that the threshold implies. Reporting it makes the operating
        point arguable by a clinician rather than asserted by an engineer.
        """
        n = len(y_true)
        out = []
        for t in thresholds:
            pred = p >= t
            tp = float(np.sum(pred & (y_true == 1)))
            fp = float(np.sum(pred & (y_true == 0)))
            w = t / max(1e-9, (1 - t))
            out.append(tp / n - (fp / n) * w)
        return np.array(out)

    def expected_cost(self, y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
        pred = p >= threshold
        fn = float(np.sum(~pred & (y_true == 1)))     # under-triage
        fp = float(np.sum(pred & (y_true == 0)))      # over-triage
        tp = float(np.sum(pred & (y_true == 1)))
        tn = float(np.sum(~pred & (y_true == 0)))
        n = max(1, len(y_true))
        return {
            "threshold": float(threshold),
            "under_triage_n": fn,
            "over_triage_n": fp,
            "under_triage_rate": fn / max(1.0, tp + fn),
            "over_triage_rate": fp / max(1.0, fp + tn),
            "sensitivity": tp / max(1.0, tp + fn),
            "specificity": tn / max(1.0, tn + fp),
            "total_cost": fn * self.cost_under + fp * self.cost_over,
            "cost_per_100_patients": (fn * self.cost_under + fp * self.cost_over) / n * 100,
        }

    def sensitivity_analysis(self, y_true: np.ndarray, p: np.ndarray) -> list[dict[str, float]]:
        """
        The same comparison at ratios 6, 12 and 24, as the policy file requires.

        If a conclusion only holds at exactly 12, it is an artefact of the
        parameter and must be reported as one.
        """
        lo, hi = self.c.cost["asymmetric_cost"]["sensitivity_range"]
        rows = []
        for ratio in (lo, self.ratio, hi):
            thr = 1.0 / (1.0 + ratio)
            r = self.expected_cost(y_true, p, thr)
            r["ratio"] = float(ratio)
            rows.append(r)
        return rows

    def comparison_table(self, y_true: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float]]:
        """Symmetric vs asymmetric, side by side. The headline artefact."""
        return {
            "symmetric_0.5": self.expected_cost(y_true, p, 0.5),
            f"asymmetric_{self.derived_threshold:.4f}": self.expected_cost(y_true, p, self.derived_threshold),
        }


@dataclass
class MissedCase:
    patient_id: str
    probability: float
    truth: int
    asymmetric_level: int
    symmetric_level: int


def missed_by_symmetric(engine: CostEngine, rows: list[tuple[str, float, int]]) -> list[MissedCase]:
    """
    Which specific patients does the accuracy-optimised threshold leave behind?

    Returned as a list of named cases rather than a rate, because "the symmetric
    model misses these four patients, and here they are" persuades where "3.1%
    under-triage" does not.
    """
    out = []
    for pid, prob, truth in rows:
        a = engine.acuity_from_probability(prob)
        s = engine.acuity_symmetric(prob)
        if truth == 1 and a <= 2 < s:
            out.append(MissedCase(pid, round(prob, 4), truth, a, s))
    return out
