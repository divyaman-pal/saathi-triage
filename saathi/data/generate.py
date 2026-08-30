"""
Builds the training set and fits both risk routes.

    python -m saathi.data.generate

THE PREDICTION TASK
-------------------
Each training row is a SNAPSHOT taken at a random point t inside the patient's
wait, not at the end of it. The label is forward-looking and deliberately NARROW:

    y = deterioration onset occurs within t + 30 minutes

It is NOT "is this patient sick". A patient who arrives critically unwell and
stays exactly that unwell for an hour is a NEGATIVE for this model, and that is
correct - the arrival acuity already caught them, deterministically, in
core/triage_rules.py. Asking one model to answer both "how sick now" and "about
to get worse" is how a triage model ends up looking accurate while quietly
re-deriving the nurse's own snapshot and adding nothing.

This split is the reason the evaluation reports two separate numbers:
  - arrival acuity vs ground-truth arrival acuity   -> the RULES layer
  - deterioration prediction vs ground-truth onset  -> the MODEL layer
A composite label would have hidden which of the two was doing the work.

WHAT IS EXCLUDED
----------------
The 20 designed cases (P-001..P-020) are NOT in the training set. They are drawn
from a different seed stream entirely (surge_fill start_index=100000). A model
must never be demonstrated on cases written to make it look good.

LEAKAGE AUDIT
-------------
Features recorded BECAUSE a clinician was already worried are leakage. In this
feature set the candidates are:
  - since_human_min      : shorter when someone already went to look. RETAINED,
                           because in simulation nobody re-checks reactively, but
                           flagged as the first feature to drop on real data.
  - n_channels_live      : a patient being watched by more channels may be a
                           patient someone was already worried about. RETAINED
                           for the same reason, with the same caveat.
  - nurse re-check vitals: a SECOND set of vitals taken early is the classic
                           leak. In the training snapshots only the arrival
                           nurse reading exists, so the leak is not present here
                           - and would be, on real data.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..core.contract_loader import get_contract
from ..core.features import assemble, feature_names
from ..core.gating import gate
from ..core.models import Channel, reset_ids
from ..core.risk_model import ARTIFACT_DIR, RiskModel, RiskRoute
from .cohort import PatientProfile, training_population
from .simulate import Simulator

HORIZON_MINUTES = 30.0


def label_at(p: PatientProfile, t: float) -> int:
    """Does this patient deteriorate within the next HORIZON_MINUTES?"""
    onset = p.vitals.onset_min
    return int(onset is not None and t <= onset <= t + HORIZON_MINUTES)


def build_rows(profiles: list[PatientProfile], seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns (X_full, y, ages, has_record_mask, feature_names_full)."""
    c = get_contract()
    rng = random.Random(seed)
    now = datetime(2026, 4, 12, 11, 0, 0)
    names = feature_names("FULL_v1")

    rows: list[list[float]] = []
    ys: list[int] = []
    ages: list[float] = []
    has_rec: list[bool] = []

    for p in profiles:
        t = round(rng.uniform(5.0, max(6.0, p.arrival_minutes_ago)), 1)
        sim = Simulator(c, now, "TIER_A", camera_stride_minutes=4.0, camera_lookback_minutes=24.0)
        ev = sim.evidence_for(p, elapsed_minutes=t)
        usable, rejected = gate(c, ev, now)
        live = len({e.source_channel for e in usable} - {Channel.SYSTEM})
        fv = assemble(c, usable, rejected, age_years=p.age_years, sex=p.sex,
                      has_prior_record=p.has_prior_record, live_channels=live)
        rows.append(fv.as_row(names))
        ys.append(label_at(p, t))
        ages.append(p.age_years)
        has_rec.append(p.has_prior_record)

    return (np.array(rows, dtype=float), np.array(ys, dtype=int),
            np.array(ages, dtype=float), np.array(has_rec, dtype=bool), names)


def main(n: int = 5200, seed: int = 7) -> None:
    t0 = time.time()
    print(f"Generating {n} training encounters from the stated generative process ...")
    profiles = training_population(n=n, seed=seed)
    reset_ids()
    X_full, y, ages, has_rec, names_full = build_rows(profiles)
    print(f"  built {X_full.shape[0]} rows x {X_full.shape[1]} features in {time.time()-t0:.1f}s")
    print(f"  positive rate: {y.mean():.3f}   with-record rate: {has_rec.mean():.3f}")
    print(f"  NaN fraction (missing, NOT imputed): {np.isnan(X_full).mean():.3f}")

    model = RiskModel()

    # FULL_v1 trains only on encounters that actually have a linked record.
    print("Fitting FULL_v1 (observation + linked record) ...")
    full = RiskRoute("FULL_v1")
    full.fit(X_full[has_rec], y[has_rec], ages[has_rec])
    model.routes["FULL_v1"] = full

    # OBSERVATION_ONLY_v1 trains on EVERYONE, using only observation columns.
    # It is a separate model, not the full model with history blanked out.
    print("Fitting OBSERVATION_ONLY_v1 (observation features only) ...")
    obs_names = feature_names("OBSERVATION_ONLY_v1")
    cols = [names_full.index(nm) for nm in obs_names]
    obs = RiskRoute("OBSERVATION_ONLY_v1")
    obs.fit(X_full[:, cols], y, ages)
    model.routes["OBSERVATION_ONLY_v1"] = obs

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACT_DIR)

    print("\nCalibration coverage (isotonic, fitted per age group on held-out data):")
    for route in model.routes.values():
        for grp, k in route.calibration_n.items():
            state = "CALIBRATED" if grp in route.calibrators else "UNCALIBRATED - reported as such at inference"
            print(f"  {route.route:22s} {grp:11s} n={k:5d}  {state}")

    print("\nTop gain-importance features, FULL_v1:")
    for k, v in sorted(full.importances.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:24s} {v:.4f}")

    print(f"\nArtifacts written to {ARTIFACT_DIR}")
    print(f"Total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
