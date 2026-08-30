"""
SAATHI evaluation.

    python -m saathi.eval.evaluate

WHAT THESE NUMBERS ARE
    Properties of saathi/data/cohort.py. A model trained on data drawn from a
    process we wrote will recover the process we wrote. Everything below
    demonstrates that the machinery behaves as specified; NOTHING below is
    evidence about clinical performance in an emergency department.

WHAT IS DELIBERATELY NOT THE HEADLINE
    AUC. It summarises ranking across every possible threshold, and we do not
    operate at every possible threshold - we operate at one, derived from a cost
    ratio. A model with excellent discrimination and poor calibration in the
    elderly is a safety hazard wearing a good number.

THE HEADLINE NUMBERS
    1. Under-triage rate at the operating point, with a confidence interval.
    2. Calibration, per age band, at that operating point.
    3. The symmetric-vs-asymmetric comparison, in named cases.
    4. What the rules-only cold start actually delivers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ..core.contract_loader import get_contract
from ..core.cost_engine import CostEngine, missed_by_symmetric
from ..core.features import assemble, feature_names
from ..core.gating import gate
from ..core.models import Channel, reset_ids
from ..core.risk_model import RiskModel, calibration_group, rules_only_score
from ..core.triage_rules import arrival_acuity_with_complaint
from ..core import red_flags as rf
from ..data.cohort import PatientProfile, surge_fill
from ..data.generate import HORIZON_MINUTES, label_at
from ..data.simulate import Simulator

ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


# ---------------------------------------------------------------------------
# Interval estimation
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """
    Wilson score interval for a proportion.

    Used rather than the normal approximation because the interesting rates here
    are small and the normal interval misbehaves badly near zero - which is
    exactly where an under-triage rate lives, and exactly where an
    over-optimistic interval would be most misleading.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(p, 4), round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def fmt_ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{p:6.3f}  [{lo:.3f}, {hi:.3f}]  (n={n})"


# ---------------------------------------------------------------------------
# Evaluation cohort
# ---------------------------------------------------------------------------


@dataclass
class Row:
    patient_id: str
    age: float
    age_band: str
    calib_group: str
    sex: str
    skin_tone: str
    language: str
    attendant_present: bool
    has_record: bool
    truth_acuity: int
    truth_deteriorates: int
    rules_arrival_acuity: int
    p_model: float
    p_rules_only: float
    red_flag_fired: bool


def build_eval_rows(n: int = 1600, seed: int = 424242) -> list[Row]:
    c = get_contract()
    model = RiskModel.load()
    reset_ids()
    profiles: list[PatientProfile] = surge_fill(n, seed=seed, start_index=900000, occupancy=1.0)
    from datetime import datetime
    now = datetime(2026, 4, 12, 11, 0, 0)
    rows: list[Row] = []
    import random
    rng = random.Random(seed + 1)

    for p in profiles:
        t = round(rng.uniform(5.0, max(6.0, p.arrival_minutes_ago)), 1)
        sim = Simulator(c, now, "TIER_A", camera_stride_minutes=4.0, camera_lookback_minutes=24.0)
        ev = sim.evidence_for(p, elapsed_minutes=t)
        usable, rejected = gate(c, ev, now)
        live = len({e.source_channel for e in usable} - {Channel.SYSTEM})
        fv = assemble(c, usable, rejected, age_years=p.age_years, sex=p.sex,
                      has_prior_record=p.has_prior_record, live_channels=live)
        band = c.age_band(p.age_years)
        ctx = rf.RuleContext(age_years=p.age_years, age_band=band,
                             complaint_symptoms=list(p.complaint_symptoms))
        hits = rf.evaluate(c, usable + rejected, ctx)
        level, _ = arrival_acuity_with_complaint(fv, hits, list(p.complaint_symptoms))
        pred = model.predict(fv, p.age_years, with_attribution=False)

        rows.append(Row(
            patient_id=p.patient_id, age=p.age_years, age_band=band,
            calib_group=calibration_group(p.age_years), sex=p.sex,
            skin_tone=p.skin_tone_band, language=p.language,
            attendant_present=p.attendant.present, has_record=p.has_prior_record,
            truth_acuity=p.truth_arrival_acuity,
            truth_deteriorates=label_at(p, t),
            rules_arrival_acuity=level,
            p_model=pred.probability, p_rules_only=rules_only_score(fv),
            red_flag_fired=bool(hits),
        ))
    return rows


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def report_arrival_rules(rows: list[Row]) -> dict:
    """
    THE RULES LAYER - what the first month of deployment actually delivers.

    Under-triage here means the deterministic layer assigned a LESS urgent level
    than the ground truth. That is the number that matters, and it is reported
    before any model number, because at cold start the model is not running.
    """
    print("\n" + "=" * 78)
    print("1. ARRIVAL ACUITY - deterministic rules layer (the cold-start system)")
    print("=" * 78)
    print("This is what runs on day one at any site, before a single prospective")
    print("case has been collected. No learned parameters. Contract age bands and")
    print("clinician-authored red flags only.\n")

    n = len(rows)
    under = [r for r in rows if r.rules_arrival_acuity > r.truth_acuity]
    over = [r for r in rows if r.rules_arrival_acuity < r.truth_acuity]
    exact = n - len(under) - len(over)
    severe_under = [r for r in under if r.truth_acuity <= 2 and r.rules_arrival_acuity >= 3]

    print(f"  exact agreement        {fmt_ci(exact, n)}")
    print(f"  UNDER-triage (any)     {fmt_ci(len(under), n)}")
    print(f"  UNDER-triage (L1/L2 missed to L3+)  {fmt_ci(len(severe_under), n)}   <-- the number that matters")
    print(f"  OVER-triage (any)      {fmt_ci(len(over), n)}")
    print("\n  Confusion matrix (rows = truth, cols = assigned):")
    print("        " + "".join(f"  L{c:<5d}" for c in range(1, 6)))
    for t in range(1, 6):
        line = f"    L{t}  "
        for a in range(1, 6):
            line += f"{sum(1 for r in rows if r.truth_acuity == t and r.rules_arrival_acuity == a):6d} "
        print(line)

    print("\n  Under-triage by age band (the age-stratification check):")
    for band in sorted({r.age_band for r in rows}):
        sub = [r for r in rows if r.age_band == band]
        su = [r for r in sub if r.truth_acuity <= 2 and r.rules_arrival_acuity >= 3]
        if len(sub) >= 20:
            print(f"    {band:14s} {fmt_ci(len(su), len(sub))}")
    return {"n": n, "exact": exact, "under": len(under), "over": len(over),
            "severe_under": len(severe_under)}


def report_model(rows: list[Row], engine: CostEngine) -> dict:
    """THE MODEL LAYER - predicting deterioration during the wait."""
    print("\n" + "=" * 78)
    print("2. DETERIORATION MODEL - discrimination, then the thing that matters")
    print("=" * 78)
    y = np.array([r.truth_deteriorates for r in rows])
    p = np.array([r.p_model for r in rows])
    pr = np.array([r.p_rules_only for r in rows])

    print(f"  Prediction task : deterioration onset within {HORIZON_MINUTES:.0f} minutes")
    print(f"  Event rate      : {y.mean():.3f}  ({int(y.sum())} of {len(y)})")
    if 0 < y.sum() < len(y):
        print(f"  AUC (model)     : {roc_auc_score(y, p):.3f}   <-- NOT the headline. See below.")
        print(f"  AUC (rules-only): {roc_auc_score(y, pr):.3f}")
    print("\n  AUC is reported for completeness and then set aside. We do not operate")
    print("  at every threshold; we operate at one, and it is derived from a cost ratio.")

    thr = engine.derived_threshold
    print(f"\n  Operating point: p >= {thr:.4f}  (cost-derived, ratio {engine.ratio:g}:1)")
    m = engine.expected_cost(y, p, thr)
    tp = int(np.sum((p >= thr) & (y == 1)))
    fn = int(np.sum((p < thr) & (y == 1)))
    fp = int(np.sum((p >= thr) & (y == 0)))
    tn = int(np.sum((p < thr) & (y == 0)))
    print(f"    sensitivity          {fmt_ci(tp, tp + fn)}   <-- HEADLINE")
    print(f"    under-triage rate    {fmt_ci(fn, tp + fn)}   <-- HEADLINE")
    print(f"    specificity          {fmt_ci(tn, tn + fp)}")
    print(f"    over-triage rate     {fmt_ci(fp, fp + tn)}")
    print(f"    escalation rate      {fmt_ci(tp + fp, len(y))}   <-- must fit the alert budget")

    print("\n  Sensitivity per ground-truth arrival acuity:")
    for lvl in (1, 2, 3, 4, 5):
        sub = [r for r in rows if r.truth_acuity == lvl and r.truth_deteriorates == 1]
        if len(sub) >= 5:
            caught = sum(1 for r in sub if r.p_model >= thr)
            print(f"    truth L{lvl}  {fmt_ci(caught, len(sub))}")
        elif sub:
            print(f"    truth L{lvl}  n={len(sub)} - too few events to estimate; NOT reported")
    return m


def report_calibration(rows: list[Row]) -> dict:
    """
    CALIBRATION, PER AGE BAND. The section a careful reader turns to first.

    A model with 0.85 AUC that is miscalibrated in the elderly is a safety
    hazard, not an achievement.
    """
    print("\n" + "=" * 78)
    print("3. CALIBRATION BY AGE GROUP - does 0.08 mean eight in a hundred, HERE?")
    print("=" * 78)
    out = {}
    for grp in ("paediatric", "adult", "geriatric"):
        sub = [r for r in rows if r.calib_group == grp]
        if len(sub) < 40:
            print(f"\n  {grp}: n={len(sub)} - too few to assess. NOT reported, not borrowed "
                  f"from another group.")
            continue
        y = np.array([r.truth_deteriorates for r in sub])
        p = np.array([r.p_model for r in sub])
        cil = p.mean() - y.mean()
        print(f"\n  {grp}  (n={len(sub)}, observed event rate {y.mean():.3f})")
        print(f"    calibration-in-the-large: mean predicted {p.mean():.4f} "
              f"vs observed {y.mean():.4f}   bias {cil:+.4f}")
        edges = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
        print("      bin              n   mean pred   observed")
        for lo, hi in zip(edges, edges[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() >= 10:
                print(f"      [{lo:.2f},{hi:.2f})  {int(m.sum()):5d}     {p[m].mean():.4f}     {y[m].mean():.4f}")
        out[grp] = {"n": len(sub), "bias": round(float(cil), 4),
                    "mean_pred": round(float(p.mean()), 4),
                    "observed": round(float(y.mean()), 4)}
    print("\n  Reading this table: a positive bias means the model is over-predicting")
    print("  deterioration in that group - it will over-triage them. A negative bias")
    print("  means it under-predicts, and under-prediction in the elderly is the")
    print("  specific failure this whole architecture exists to prevent.")
    return out


def report_threshold_comparison(rows: list[Row], engine: CostEngine) -> dict:
    """
    THE HEADLINE ARTEFACT.

    The same model, at the accuracy-optimised threshold and at the cost-derived
    one, side by side - and then the specific patients the accuracy-optimised
    version leaves in the waiting room.
    """
    print("\n" + "=" * 78)
    print("4. SYMMETRIC vs ASYMMETRIC THRESHOLD - the same model, two policies")
    print("=" * 78)
    y = np.array([r.truth_deteriorates for r in rows])
    p = np.array([r.p_model for r in rows])
    table = engine.comparison_table(y, p)

    print(f"\n  {'':38s} {'symmetric 0.500':>17s} {'cost-derived ' + f'{engine.derived_threshold:.4f}':>17s}")
    keys = [("sensitivity", "sensitivity"), ("under_triage_rate", "UNDER-triage rate"),
            ("over_triage_rate", "over-triage rate"), ("specificity", "specificity"),
            ("under_triage_n", "patients under-triaged"),
            ("over_triage_n", "patients over-triaged"),
            ("cost_per_100_patients", "policy cost per 100 patients")]
    sym = table["symmetric_0.5"]
    asym = table[f"asymmetric_{engine.derived_threshold:.4f}"]
    for k, label in keys:
        fs = f"{sym[k]:17.3f}" if isinstance(sym[k], float) else f"{sym[k]:17}"
        fa = f"{asym[k]:17.3f}" if isinstance(asym[k], float) else f"{asym[k]:17}"
        print(f"  {label:38s} {fs} {fa}")

    missed = missed_by_symmetric(engine, [(r.patient_id, r.p_model, r.truth_deteriorates)
                                          for r in rows])
    print(f"\n  Patients the accuracy-optimised threshold LEAVES IN THE WAITING ROOM: {len(missed)}")
    for m in missed[:10]:
        print(f"    {m.patient_id}  p={m.probability:.4f}  deteriorated=YES  "
              f"cost-derived -> L{m.asymmetric_level}, accuracy-optimised -> L{m.symmetric_level}")
    if len(missed) > 10:
        print(f"    ... and {len(missed) - 10} more")
    if sym["sensitivity"] == 0.0:
        print("\n  The symmetric column reads 0.000 sensitivity. That is not a broken")
        print("  comparison - it is the point. With a 4% event rate, a WELL-CALIBRATED")
        print("  model should almost never emit p > 0.5, because more than half of such")
        print("  patients really would have to deteriorate. An accuracy-optimised")
        print("  threshold on a rare event is not conservative; it is inert.")

    print("\n  This is the argument, in patients rather than percentages. Choosing")
    print("  0.5 is not choosing 'no policy'. It is silently choosing a 1:1 cost")
    print("  ratio, and asserting that an unattended deterioration and a wasted")
    print("  nurse walk are equally bad.")

    print("\n  SENSITIVITY ANALYSIS - required by cost_policy.yaml:")
    print(f"  {'ratio':>7s} {'threshold':>10s} {'sens':>8s} {'under':>8s} {'over':>8s} {'escalated':>10s}")
    for r in engine.sensitivity_analysis(y, p):
        esc = (r["under_triage_n"] * 0 + np.sum(p >= r["threshold"])) / len(p)
        print(f"  {r['ratio']:7.0f} {r['threshold']:10.4f} {r['sensitivity']:8.3f} "
              f"{r['under_triage_rate']:8.3f} {r['over_triage_rate']:8.3f} {esc:10.3f}")
    print("\n  If a conclusion holds only at exactly 12:1, it is an artefact of the")
    print("  parameter rather than a property of the system, and must be reported so.")

    print("\n  DECISION CURVE (net benefit; higher is better at that threshold):")
    ts = np.array([0.02, 0.05, engine.derived_threshold, 0.10, 0.20, 0.35, 0.50])
    nb = engine.net_benefit(y, p, ts)
    nb_all = np.array([y.mean() - (1 - y.mean()) * (t / (1 - t)) for t in ts])
    print(f"  {'threshold':>10s} {'model':>9s} {'escalate-all':>14s} {'escalate-none':>15s}")
    for t, a, b in zip(ts, nb, nb_all):
        mark = "  <-- operating point" if abs(t - engine.derived_threshold) < 1e-9 else ""
        print(f"  {t:10.4f} {a:9.4f} {b:14.4f} {0.0:15.4f}{mark}")
    return {"symmetric": sym, "asymmetric": asym, "n_missed_by_symmetric": len(missed)}


def report_baselines(rows: list[Row], engine: CostEngine) -> dict:
    """
    What does the model add over what you already had?

    Three baselines, because a model that does not beat them is not worth the
    deployment risk:
      RULES-ONLY   the cold-start system, contract thresholds and red flags
      NURSE-ALONE  a triage snapshot with no re-assessment at all - which is
                   what the department does today
      ESCALATE-ALL the trivial policy that never misses anyone
    """
    print("\n" + "=" * 78)
    print("5. BASELINES - what does the learned model actually add?")
    print("=" * 78)
    y = np.array([r.truth_deteriorates for r in rows])
    thr = engine.derived_threshold
    out = {}

    p_model = np.array([r.p_model for r in rows])
    m_model = engine.expected_cost(y, p_model, thr)
    out["model"] = m_model
    print(f"\n  model at the cost-derived threshold p >= {thr:.4f}")
    print(f"    sensitivity {m_model['sensitivity']:.3f}   under-triage {m_model['under_triage_rate']:.3f}   "
          f"over-triage {m_model['over_triage_rate']:.3f}   cost/100 {m_model['cost_per_100_patients']:.2f}")

    # The rules-only score is a bounded heuristic, NOT a calibrated probability.
    # Applying the model's threshold to it compares two different scales and
    # flatters whichever happens to be more generous. The fair comparison holds
    # the OPERATIONAL COST constant - the same number of escalations, and
    # therefore the same load on the same nurses - and asks who catches more.
    n_escalated = int((p_model >= thr).sum())
    p_rules = np.array([r.p_rules_only for r in rows])
    rule_thr = float(np.sort(p_rules)[::-1][min(n_escalated, len(p_rules) - 1)])
    m_rules = engine.expected_cost(y, p_rules, rule_thr)
    out["rules_only_matched_rate"] = m_rules
    print(f"\n  rules-only cold start, MATCHED to the same escalation rate")
    print(f"    (score >= {rule_thr:.3f}, giving {int((p_rules >= rule_thr).sum())} escalations "
          f"against the model's {n_escalated})")
    print(f"    sensitivity {m_rules['sensitivity']:.3f}   under-triage {m_rules['under_triage_rate']:.3f}   "
          f"over-triage {m_rules['over_triage_rate']:.3f}   cost/100 {m_rules['cost_per_100_patients']:.2f}")
    delta = m_model["sensitivity"] - m_rules["sensitivity"]
    print(f"\n    The learned model is worth {delta:+.3f} sensitivity at identical nurse load.")
    if delta <= 0.02:
        print("    That is not a meaningful gain. On this cohort the model does not")
        print("    earn its deployment risk over the rules alone, and saying so is")
        print("    more useful than a chart that hides it.")

    # Nurse-alone: the arrival snapshot, never revisited. Catches only those who
    # were already sick enough at arrival to be an L1/L2.
    caught = sum(1 for r in rows if r.truth_deteriorates == 1 and r.rules_arrival_acuity <= 2)
    events = int(y.sum())
    print("\n  nurse-alone (triage once, never re-assess) - today's practice")
    print(f"    of the {events} patients who deteriorated during the wait, the arrival")
    print(f"    snapshot had already placed {caught} at L1/L2. The remaining "
          f"{events - caught} were correctly triaged on arrival and then not looked at again.")
    print(f"    sensitivity to in-wait deterioration  {fmt_ci(caught, events)}")
    print("\n    THIS IS THE GAP SAATHI EXISTS TO CLOSE. It is not a claim that the")
    print("    nurse was wrong - the arrival decision was right. It is a claim that")
    print("    the arrival decision was the only one anybody made.")

    print("\n  escalate-everyone (the trivial policy)")
    print(f"    sensitivity 1.000   over-triage 1.000   "
          f"cost/100 {(len(y) - events) * engine.cost_over / len(y) * 100:.2f}")
    print("    Never misses anyone, and would be ignored by the second shift.")
    out["nurse_alone_sensitivity"] = round(caught / max(1, events), 4)
    return out


def report_system_level(n: int = 260, seed: int = 555) -> dict:
    """
    THE NUMBER THAT ACTUALLY MATTERS.

    Everything above measures the tabular model in isolation, which is the
    fairest way to judge the model and the least useful way to judge the system.
    A patient in SAATHI is protected by SEVEN independent escalation paths:

        deterministic red flags        materiality (proxy reports)
        trajectory materiality         multi-factor accumulation
        channel discordance            the cost-threshold model
        wait-time SLA re-check

    A model sensitivity of 0.36 does not mean 64% of deteriorating patients are
    missed; it means the model alone catches 36% and six other mechanisms are
    also running. This section runs the REAL pipeline - the same code the demo
    and the API use - over a held-out cohort and asks the only question a
    clinician cares about: was this patient's urgency raised, or a human sent to
    look, before they got worse?
    """
    from ..runtime import Runtime
    from ..data.cohort import surge_fill

    print("\n" + "=" * 78)
    print("6. WHOLE-SYSTEM ESCALATION - every path, running the real pipeline")
    print("=" * 78)

    rt = Runtime(surge_n=0).build(include_surge=False, reset_store=False)
    rt.profiles = {p.patient_id: p for p in
                   surge_fill(n, seed=seed, start_index=700000, occupancy=1.0)}
    rt.assess_all()

    det = [a for a in rt.assessments.values()
           if rt.profiles[a.patient_id].truth_deteriorates]
    well = [a for a in rt.assessments.values()
            if not rt.profiles[a.patient_id].truth_deteriorates]

    def acted(a) -> bool:
        """Urgency raised, OR a human was explicitly sent to look."""
        return (a.acuity_current.level < a.acuity_arrival.level
                or bool(a.red_flags)
                or a.abstention.abstained
                or any(m.action in ("ESCALATE", "RECHECK") for m in a.materiality)
                or a.sla.breached)

    def raised(a) -> bool:
        return a.acuity_current.level < a.acuity_arrival.level or bool(a.red_flags)

    print(f"  Held-out cohort: {len(rt.assessments)} encounters, "
          f"{len(det)} of whom deteriorated during the wait.\n")
    if det:
        print(f"    urgency actually RAISED for a deteriorating patient   "
              f"{fmt_ci(sum(raised(a) for a in det), len(det))}")
        print(f"    ANY action taken (raised, flagged, re-check forced)   "
              f"{fmt_ci(sum(acted(a) for a in det), len(det))}   <-- the safety net")
    if well:
        print(f"\n    urgency raised for a patient who did NOT deteriorate  "
              f"{fmt_ci(sum(raised(a) for a in well), len(well))}   <-- the price paid")
        print(f"    any action for a patient who did not deteriorate      "
              f"{fmt_ci(sum(acted(a) for a in well), len(well))}")

    paths: dict[str, int] = {}
    for a in det:
        if not raised(a):
            continue
        key = a.acuity_change_reason.split(":")[0]
        paths[key] = paths.get(key, 0) + 1
    if paths:
        print("\n  Which path caught the deteriorating patients whose urgency rose:")
        for k, v in sorted(paths.items(), key=lambda kv: -kv[1]):
            print(f"    {k:38s} {v:4d}")
        print("\n  Read that list carefully. If one path dominates, the other six are")
        print("  decoration. If the model appears far down it, the model is not what")
        print("  is protecting these patients - the contract and the rules are.")
    return {"n": len(rt.assessments), "deteriorated": len(det),
            "raised": sum(raised(a) for a in det), "acted": sum(acted(a) for a in det),
            "false_raise": sum(raised(a) for a in well), "paths": paths}


def report_subgroups(rows: list[Row], engine: CostEngine) -> dict:
    """
    FAIRNESS. Reported because a system whose accuracy depends on whether a
    family member came along has an equity problem that must be quantified, not
    hidden.
    """
    print("\n" + "=" * 78)
    print("7. SUBGROUP PERFORMANCE - stratified, with the uncomfortable ones included")
    print("=" * 78)
    thr = engine.derived_threshold
    out: dict = {}

    def block(title: str, key, note: str = "") -> None:
        print(f"\n  by {title}:")
        if note:
            print(f"    {note}")
        print(f"    {'group':22s} {'n':>5s} {'events':>7s} {'sensitivity (95% CI)':>28s} {'escalation rate':>16s}")
        for g in sorted({key(r) for r in rows}, key=str):
            sub = [r for r in rows if key(r) == g]
            ev = [r for r in sub if r.truth_deteriorates == 1]
            esc = sum(1 for r in sub if r.p_model >= thr) / max(1, len(sub))
            if len(ev) >= 8:
                caught = sum(1 for r in ev if r.p_model >= thr)
                print(f"    {str(g):22s} {len(sub):5d} {len(ev):7d} {fmt_ci(caught, len(ev)):>28s} {esc:16.3f}")
                out.setdefault(title, {})[str(g)] = wilson(caught, len(ev))
            else:
                print(f"    {str(g):22s} {len(sub):5d} {len(ev):7d} {'too few events - NOT reported':>28s} {esc:16.3f}")

    block("age band", lambda r: r.age_band)
    block("sex", lambda r: r.sex)
    block("Fitzpatrick skin-tone band", lambda r: r.skin_tone,
          note="rPPG SNR degrades at higher Fitzpatrick types under standard pipelines. "
               "This row is a MONITORED METRIC, not a footnote.")
    block("attendant present", lambda r: "present" if r.attendant_present else "ABSENT",
          note="If performance depends on whether a family member came along, that is an "
               "equity problem and it belongs on this page.")
    block("prior record", lambda r: "linked" if r.has_record else "none")
    block("language", lambda r: r.language)
    return out


def report_alert_budget(rows: list[Row], engine: CostEngine) -> None:
    """Does the escalation rate fit inside what the floor can absorb?"""
    c = get_contract()
    print("\n" + "=" * 78)
    print("8. ALERT BUDGET - can the department actually absorb this?")
    print("=" * 78)
    thr = engine.derived_threshold
    rate = sum(1 for r in rows if r.p_model >= thr) / len(rows)
    for tier in ("TIER_A", "TIER_B", "TIER_C"):
        prof = c.profile(tier)
        arrivals = prof["baseline_arrivals_per_hour"]
        nurses = prof["triage_capable_nurses"]
        absorbable = nurses * c.surge["honest_failure_point"]["absorbable_escalations_per_hour_per_nurse"]
        generated = rate * arrivals
        budget = c.alert_budget["normal"]["alerts_per_hour"]
        verdict = "WITHIN BUDGET" if generated <= min(absorbable, budget) else "EXCEEDS BUDGET"
        print(f"\n  {tier}: {arrivals} arrivals/hour, {nurses} triage nurse(s)")
        print(f"    escalation rate {rate:.3f} -> {generated:.1f} escalations/hour")
        print(f"    floor can absorb {absorbable:.0f}/hour; policy budget {budget}/hour -> {verdict}")
        if generated > min(absorbable, budget):
            print("    Over budget: the lowest-ranked alerts are DEFERRED to the batched")
            print("    queue view, never discarded, and red flags and abstentions stay exempt.")


def report_limitations() -> None:
    print("\n" + "=" * 78)
    print("9. WHAT THESE NUMBERS DO NOT SHOW")
    print("=" * 78)
    for line in [
        "The cohort is SIMULATED from a process we wrote (saathi/data/cohort.py).",
        "  A model fitted to it recovers that process. None of the above is evidence",
        "  about clinical performance in any emergency department.",
        "",
        "No transfer claim is made from MIMIC-IV-ED or NHAMCS. Both are US datasets;",
        "  neither describes an Indian district ED. Pre-training would be FROZEN and",
        "  treated as a prior, with prospective validation through BODH/ABDM.",
        "",
        "rPPG values here are a replayed quality trace, not a trained model. The",
        "  reliability weight of 0.35 for respiratory rate from video, and the SNR",
        "  floor that rejects it outright, encode our actual confidence in it.",
        "",
        "The skin-tone subgroup row above reflects a penalty WE INJECTED into the",
        "  simulator to make the monitoring visible. It is not a measurement of",
        "  real-world rPPG bias - it is a demonstration that the metric is watched.",
        "",
        "Automation bias is NOT measured here and cannot be. Whether showing a nurse",
        "  a recommendation improves or anchors their judgement is answerable only by",
        "  a prospective study with a shadow-mode arm, which is why the deployment",
        "  plan starts with thirty days of exactly that.",
        "",
        "The feedback loop is real and unhandled: if SAATHI escalates a patient who is",
        "  then seen sooner and does well, the label is contaminated. Any retraining",
        "  must account for it; none of the above does.",
    ]:
        print("  " + line)


# ---------------------------------------------------------------------------


def main(n: int = 1600) -> None:
    c = get_contract()
    engine = CostEngine(c)
    print("=" * 78)
    print("SAATHI EVALUATION")
    print("=" * 78)
    print(f"contract v{c.version} | rules v{c.ruleset_version} | cost policy v{c.policy_version}")
    print(f"cost ratio {engine.ratio:g}:1 -> operating threshold {engine.derived_threshold:.4f}")
    print(f"Building {n} held-out encounters (seed disjoint from training) ...")
    rows = build_eval_rows(n)

    results = {
        "arrival_rules": report_arrival_rules(rows),
        "model": report_model(rows, engine),
        "calibration": report_calibration(rows),
        "threshold_comparison": report_threshold_comparison(rows, engine),
        "baselines": report_baselines(rows, engine),
        "system_level": report_system_level(),
        "subgroups": report_subgroups(rows, engine),
    }
    report_alert_budget(rows, engine)
    report_limitations()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "evaluation.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nMachine-readable results written to {out}")


if __name__ == "__main__":
    main()
