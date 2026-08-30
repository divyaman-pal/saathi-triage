"""
Tabular risk estimation.

TWO ROUTES, NOT ONE MODEL WITH IMPUTED HISTORY
----------------------------------------------
  FULL_v1              - observation features + linked-record features
  OBSERVATION_ONLY_v1  - observation features only

Roughly half of Indian ED arrivals have no retrievable record. The wrong
response is to run the full model with an imputed comorbidity count and present
the output as equivalent. The right response is a genuinely separate model
trained only on what is actually available, labelled as such at every point it
is displayed, with a shortened re-check interval to compensate for the reduced
information. That is what this module implements.

CALIBRATION IS NOT DISCRIMINATION
---------------------------------
AUC is close to useless as a headline metric for a decision that will be made at
a cost-derived operating point. What matters is whether P(escalation) = 0.08
actually means eight in a hundred, IN THIS SUBGROUP. So each route carries a
per-age-group isotonic calibrator, and each reports its own calibration status.
A subgroup with too few development cases is reported as UNCALIBRATED_SUBGROUP
and does not silently borrow the adult calibration curve.

ATTRIBUTION IS NOT CAUSATION
----------------------------
Contributions come from exact TreeSHAP (xgboost's pred_contribs). A SHAP value
states how much a feature moved THIS MODEL'S OUTPUT relative to the dataset
baseline. It is not a physiological mechanism and the claim grammar forbids it
from being narrated as one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from .features import FULL_FEATURES, OBSERVATION_FEATURES, FeatureVector, feature_names

ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"

MODEL_VERSIONS = {
    "FULL_v1": "xgb_full_v3.2",
    "OBSERVATION_ONLY_v1": "xgb_obs_v3.2",
}

# Age groups used for calibration. Deliberately coarser than the contract's
# seven clinical bands: isotonic regression needs enough events per group to
# mean anything, and pretending otherwise would be its own miscalibration.
CALIBRATION_GROUPS = [
    ("paediatric", 0.0, 18.0),
    ("adult", 18.0, 65.0),
    ("geriatric", 65.0, 130.0),
]
MIN_CALIBRATION_N = 150


def calibration_group(age_years: float) -> str:
    for name, lo, hi in CALIBRATION_GROUPS:
        if lo <= age_years < hi:
            return name
    return "adult"


@dataclass
class Prediction:
    probability: float                     # calibrated
    probability_raw: float                 # pre-calibration
    route: str
    model_version: str
    calibration_status: str
    contributions: dict[str, float] = field(default_factory=dict)
    baseline: float = 0.0

    def top_contributions(self, n: int = 6) -> list[tuple[str, float]]:
        items = [(k, v) for k, v in self.contributions.items() if abs(v) > 1e-4]
        items.sort(key=lambda kv: -abs(kv[1]))
        return items[:n]


class RiskRoute:
    """One trained route: booster + per-group calibrators."""

    def __init__(self, route: str):
        self.route = route
        self.names = feature_names(route)
        self.version = MODEL_VERSIONS[route]
        self.booster: xgb.Booster | None = None
        self.calibrators: dict[str, IsotonicRegression] = {}
        self.calibration_n: dict[str, int] = {}
        self.calibrated_on: str = ""
        self.importances: dict[str, float] = {}
        self.base_rate: float = 0.0

    # -- training ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, ages: np.ndarray, seed: int = 11) -> None:
        n = len(y)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = int(n * 0.70)
        tr, cal = idx[:cut], idx[cut:]

        # Class imbalance is real: critical events are rare. We weight the
        # positive class up, but we do NOT let that stand in for the cost
        # asymmetry - that is applied later, at the decision threshold, where it
        # is visible and auditable. Conflating the two hides the policy inside
        # the model.
        pos = max(1, int(y[tr].sum()))
        neg = max(1, len(tr) - pos)
        dtrain = xgb.DMatrix(X[tr], label=y[tr], feature_names=self.names,
                             missing=float("nan"))
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "eta": 0.06,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 8,
            "reg_lambda": 2.0,
            "scale_pos_weight": min(4.0, neg / pos),
            "seed": seed,
            "nthread": 4,
        }
        self.booster = xgb.train(params, dtrain, num_boost_round=220)
        self.base_rate = float(y.mean())

        gains = self.booster.get_score(importance_type="gain")
        total = sum(gains.values()) or 1.0
        self.importances = {k: v / total for k, v in gains.items()}

        # Held-out calibration, fitted separately per age group.
        praw = self._raw(X[cal])
        self.calibrated_on = date.today().isoformat()
        for name, lo, hi in CALIBRATION_GROUPS:
            m = (ages[cal] >= lo) & (ages[cal] < hi)
            k = int(m.sum())
            self.calibration_n[name] = k
            if k >= MIN_CALIBRATION_N and 0 < y[cal][m].sum() < k:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(praw[m], y[cal][m])
                self.calibrators[name] = iso

    def _raw(self, X: np.ndarray) -> np.ndarray:
        assert self.booster is not None
        d = xgb.DMatrix(X, feature_names=self.names, missing=float("nan"))
        return self.booster.predict(d)

    # -- inference ---------------------------------------------------------

    def predict(self, fv: FeatureVector, age_years: float, *, with_attribution: bool = True) -> Prediction:
        assert self.booster is not None, f"Route {self.route} is not trained."
        row = np.array([fv.as_row(self.names)], dtype=float)
        raw = float(self._raw(row)[0])

        grp = calibration_group(age_years)
        iso = self.calibrators.get(grp)
        if iso is not None:
            p = float(iso.predict([raw])[0])
            status = f"CALIBRATED_{grp}_{self.calibrated_on}"
        else:
            # No borrowing. If we have not calibrated this subgroup, the number
            # is the uncalibrated model output and it says so.
            p = raw
            status = f"UNCALIBRATED_SUBGROUP_{grp}_n={self.calibration_n.get(grp, 0)}"

        contribs: dict[str, float] = {}
        baseline = 0.0
        if with_attribution:
            d = xgb.DMatrix(row, feature_names=self.names, missing=float("nan"))
            sv = self.booster.predict(d, pred_contribs=True)[0]
            baseline = float(sv[-1])
            contribs = {n: float(v) for n, v in zip(self.names, sv[:-1])}

        return Prediction(
            probability=max(0.0, min(1.0, p)),
            probability_raw=raw,
            route=self.route,
            model_version=self.version,
            calibration_status=status,
            contributions=contribs,
            baseline=baseline,
        )

    # -- persistence -------------------------------------------------------

    def save(self, directory: Path) -> None:
        assert self.booster is not None
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / f"{self.route}.ubj"))
        meta = {
            "route": self.route,
            "version": self.version,
            "names": self.names,
            "importances": self.importances,
            "base_rate": self.base_rate,
            "calibrated_on": self.calibrated_on,
            "calibration_n": self.calibration_n,
            "calibrators": {
                g: {"x": list(map(float, iso.X_thresholds_)), "y": list(map(float, iso.y_thresholds_))}
                for g, iso in self.calibrators.items()
            },
        }
        (directory / f"{self.route}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, route: str, directory: Path) -> "RiskRoute":
        obj = cls(route)
        b = xgb.Booster()
        b.load_model(str(directory / f"{route}.ubj"))
        obj.booster = b
        meta = json.loads((directory / f"{route}.meta.json").read_text(encoding="utf-8"))
        obj.names = meta["names"]
        obj.version = meta["version"]
        obj.importances = meta["importances"]
        obj.base_rate = meta["base_rate"]
        obj.calibrated_on = meta["calibrated_on"]
        obj.calibration_n = meta["calibration_n"]
        for g, pts in meta["calibrators"].items():
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(np.array(pts["x"]), np.array(pts["y"]))
            obj.calibrators[g] = iso
        return obj


class RiskModel:
    """Both routes behind one object, plus the kill switch used by the tests."""

    def __init__(self) -> None:
        self.routes: dict[str, RiskRoute] = {}
        self.enabled = True

    def predict(self, fv: FeatureVector, age_years: float, **kw) -> Prediction | None:
        """
        Returns None when models are disabled - the degraded-mode path.

        A None here must move the patient TOWARD human attention. Callers are
        forbidden from treating it as 'low risk'; the pipeline routes it to the
        rules-only + abstention path and the invariant suite tests that.
        """
        if not self.enabled:
            return None
        route = self.routes.get(fv.route)
        if route is None:
            route = self.routes["OBSERVATION_ONLY_v1"]
        return route.predict(fv, age_years, **kw)

    def importances(self, route: str) -> dict[str, float]:
        r = self.routes.get(route)
        return r.importances if r else {}

    def versions(self) -> dict[str, str]:
        return {k: v.version for k, v in self.routes.items()}

    def save(self, directory: Path = ARTIFACT_DIR) -> None:
        for r in self.routes.values():
            r.save(directory)

    @classmethod
    def load(cls, directory: Path = ARTIFACT_DIR) -> "RiskModel":
        m = cls()
        for route in ("FULL_v1", "OBSERVATION_ONLY_v1"):
            m.routes[route] = RiskRoute.load(route, directory)
        return m

    @classmethod
    def is_built(cls, directory: Path = ARTIFACT_DIR) -> bool:
        return all((directory / f"{r}.ubj").exists() for r in ("FULL_v1", "OBSERVATION_ONLY_v1"))


# ---------------------------------------------------------------------------
# Rules-only baseline
# ---------------------------------------------------------------------------


def rules_only_score(fv: FeatureVector) -> float:
    """
    THE HONEST FIRST-MONTH BASELINE.

    Cold start runs rules-only while data is collected prospectively. This is
    what a Tier C site gets on day one, and what every site gets before any
    model is trusted. It is a monotone function of the CONTRACT's age-band
    severities plus the deterministic clock - no learned parameters at all.

    We report its performance alongside the model in eval/, because a model that
    does not beat this is not worth the deployment risk.
    """
    def g(k: str, default: float = 0.0) -> float:
        v = fv.values.get(k, float("nan"))
        return default if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

    severity = (
        g("band_hr") + g("band_rr") + g("band_spo2") + g("band_sbp")
        + g("band_temp") + g("band_cap")
    )
    score = 0.05 + 0.055 * severity
    score += 0.10 * g("avpu_ord")
    score += 0.06 * g("work_of_breathing")
    score += 0.05 * max(0.0, g("arrival_mode_ord") - 1.0)
    score += 0.04 * max(0.0, g("posture_ord") - 1.0)
    score += 0.09 * g("att_confusion")
    score += 0.05 * g("att_sleepy")
    score += 0.10 * (1.0 - g("att_responds", 1.0))
    score += 0.015 * max(0.0, g("d_rr_20"))
    score += 0.004 * max(0.0, g("d_hr_20"))
    score += 0.02 * max(0.0, g("stillness_min") - 8.0) / 6.0
    return float(max(0.0, min(1.0, score)))
