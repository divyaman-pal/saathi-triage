"""
Multi-channel fusion.

FUSION, NOT VOTING. Three channels that agree tell you one thing. Three channels
that disagree tell you something else, and the something else is often more
useful. A majority vote throws that away; this module models it.

THE ASYMMETRY THAT MATTERS
--------------------------
Discordance is signed, and the two directions are NOT treated alike:

  ESCALATING discordance - the family reports a change that the sensors and the
      vitals do not show. This RAISES concern. It is the calm-but-confused
      patient, and it is the case a vital-sign early warning score cannot see.

  DE-ESCALATING discordance - the sensors look alarming and the family, who know
      the baseline, say nothing has changed. This can only DAMPEN a pending
      escalation. It can never lower an acuity, because monotonicity forbids
      that, and it can never fully cancel an objective critical value.

That asymmetry is the whole reason the attendant channel is worth having, and it
is why "just average the channels" would be actively unsafe.

REPORTER CALIBRATION
--------------------
Attendant reports are weighted per reporter, learned across the visit by
comparing each proxy report against contemporaneous objective observations. An
attendant whose reports are repeatedly contradicted converges toward the floor
weight; an attendant whose reports are corroborated converges toward the ceiling.

This is what makes the escalation path non-gameable in the SCORING LAYER rather
than on a slide - but note carefully what it does NOT do: a low-weighted
reporter still triggers nurse re-checks. We down-weight their influence on the
SCORE. We never stop listening to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contract_loader import Contract, band_severity
from .models import Channel, Evidence, SupportDirection, ThresholdBand

# Concepts that carry a channel's "concern" reading. Free text, identifiers and
# clock values are excluded - they are not observations of the patient's state.
CONCERN_CONCEPTS = {
    "HEART_RATE", "RESP_RATE", "SPO2", "SBP", "TEMP", "CAP_REFILL", "SHOCK_INDEX",
    "AVPU", "ARRIVAL_MODE", "POSTURE", "WORK_OF_BREATHING", "STILLNESS_MINUTES",
    "SKIN_COLOR_CHANGE", "CONFUSION_NEW", "SLEEPINESS_INCREASE", "RESPONDS_TO_NAME",
    "PAIN_SELF_REPORT",
}

REPORTER_FLOOR = 0.30
REPORTER_CEILING = 0.85
REPORTER_BASE = 0.70


@dataclass
class ChannelSubScore:
    channel: Channel
    concern: float                # 0..1
    n_items: int
    mean_reliability: float
    evidence_ids: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class ReporterCalibration:
    weight: float = REPORTER_BASE
    corroborated: int = 0
    contradicted: int = 0
    n_reports: int = 0
    detail: str = ""


@dataclass
class Discordance:
    magnitude: float = 0.0            # 0..1, unsigned - how much do channels disagree
    escalating: float = 0.0           # attendant sees what the sensors do not
    de_escalating: float = 0.0        # sensors alarm, the people who know the baseline do not
    agreement: float = 1.0            # 1 - normalised spread; feeds channel_agreement confidence
    narrative: str = ""
    pairs: list[tuple[str, str, float]] = field(default_factory=list)


@dataclass
class Trajectory:
    concept_id: str
    delta: float
    minutes: float
    material: bool
    materiality_threshold: float          # the CLINICAL bar, from the contract
    direction: str
    evidence_ids: list[str] = field(default_factory=list)
    acquisition_method: str = ""
    reliability_weight: float = 1.0
    effective_threshold: float = 0.0      # the bar THIS INSTRUMENT must clear
    n_points: int = 0
    start_value: float = 0.0
    end_value: float = 0.0

    def describe(self, unit: str | None) -> str:
        u = f" {unit}" if unit else ""
        sign = "+" if self.delta > 0 else ""
        return (f"{self.concept_id} {sign}{self.delta:g}{u} over {self.minutes:.0f} min "
                f"({self.start_value:g} to {self.end_value:g}, {self.acquisition_method})")

    def explain_threshold(self) -> str:
        return (
            f"Clinical materiality threshold {self.materiality_threshold:g}; "
            f"evidence bar for this instrument {self.effective_threshold:.1f} "
            f"({self.acquisition_method}, reliability {self.reliability_weight:.2f}). "
            f"Observed change {abs(self.delta):g} over {self.n_points} readings.")


# ---------------------------------------------------------------------------
# Per-channel concern
# ---------------------------------------------------------------------------


def channel_subscores(contract: Contract, usable: list[Evidence]) -> dict[Channel, ChannelSubScore]:
    """
    Reduce each channel to a single 0..1 concern reading, using the CONTRACT's
    age-band severity for every observation, weighted by that acquisition
    method's reliability.

    Deliberately coarse. This is not a second risk model - it exists only so the
    channels can be compared with each other.
    """
    buckets: dict[Channel, list[tuple[float, float, str]]] = {}
    for ev in usable:
        if ev.concept_id not in CONCERN_CONCEPTS:
            continue
        if ev.threshold_band is ThresholdBand.UNKNOWN:
            continue
        sev = band_severity(ev.threshold_band) / 3.0
        buckets.setdefault(ev.source_channel, []).append((sev, ev.reliability_weight, ev.evidence_id))

    out: dict[Channel, ChannelSubScore] = {}
    for ch, items in buckets.items():
        # Concern is driven by the WORST observation the channel produced, not
        # the average. Averaging a critical value against five normal ones is
        # how a channel's alarm gets diluted into silence.
        worst = max(s for s, _, _ in items)
        mean = sum(s * w for s, w, _ in items) / max(1e-6, sum(w for _, w, _ in items))
        concern = 0.70 * worst + 0.30 * mean
        out[ch] = ChannelSubScore(
            channel=ch,
            concern=round(min(1.0, concern), 3),
            n_items=len(items),
            mean_reliability=round(sum(w for _, w, _ in items) / len(items), 3),
            evidence_ids=[e for _, _, e in items],
        )
    return out


# ---------------------------------------------------------------------------
# Reporter calibration
# ---------------------------------------------------------------------------


def calibrate_reporter(usable: list[Evidence], all_evidence: list[Evidence]) -> ReporterCalibration:
    """
    Move a reporter's weight by comparing their claims against a HUMAN check.

    Three rules make this fair, and each exists because the naive version is
    unfair in a specific, harmful way:

      ONLY A NURSE ADJUDICATES. The camera cannot see confusion. Counting a
      normal-looking camera window as evidence against a family's report of new
      confusion would penalise reporters for detecting exactly the thing the
      camera is blind to - which is the entire reason the attendant channel
      exists. Only a contemporaneous nurse observation can contradict a proxy
      report.

      ONLY CLAIMS OF CHANGE COUNT. Answering "no, nothing has changed" when
      nothing has changed is trivially correct and says nothing about a
      reporter's quality. Counting it would let a silent family accumulate
      credibility they never demonstrated.

      A SINGLE CONTRADICTION MOVES ALMOST NOTHING. The weight shift is shrunk by
      min(1, n/3), so one report contradicted by one nurse check barely moves
      the dial. Repeated contradiction moves it to the floor. This is what
      separates the calm-but-confused patient, whose family is right and the
      sensors are wrong, from the anxious relative pressing the button every
      four minutes.

    And note what a low weight does NOT do: it never stops a red flag firing,
    and it never stops the nurse re-check. It reduces the reporter's influence
    on the SCORE. We keep listening.
    """
    proxies = [e for e in usable
               if e.source_channel is Channel.ATTENDANT
               and e.concept_id in ("CONFUSION_NEW", "SLEEPINESS_INCREASE", "RESPONDS_TO_NAME")]
    if not proxies:
        return ReporterCalibration(detail="No proxy reports yet - reporter at base weight.")

    # Human adjudicators only.
    adjudicators = [e for e in all_evidence
                    if e.source_channel is Channel.NURSE
                    and e.concept_id in ("AVPU", "HEART_RATE", "RESP_RATE", "SPO2", "SBP",
                                         "CAP_REFILL", "TEMP")
                    and e.signal_quality.passed_floor]

    corr = contra = 0
    for pr in proxies:
        claims_change = (bool(pr.value) if pr.concept_id != "RESPONDS_TO_NAME" else not bool(pr.value))
        if not claims_change:
            continue
        near = [o for o in adjudicators
                if abs((o.observed_at - pr.observed_at).total_seconds()) <= 8 * 60]
        if not near:
            continue        # nobody looked; the claim is neither confirmed nor refuted
        if any(o.threshold_band in (ThresholdBand.CONCERNING, ThresholdBand.CRITICAL) for o in near):
            corr += 1
        else:
            contra += 1

    n = corr + contra
    if n == 0:
        return ReporterCalibration(
            n_reports=len(proxies),
            detail=("No nurse observation close enough in time to adjudicate any of this "
                    f"reporter's {len(proxies)} report(s). Weight held at base "
                    f"{REPORTER_BASE:.2f} - an unadjudicated report is not a wrong one."))

    net = (corr - contra) / n
    shrink = min(1.0, n / 3.0)          # one contradiction is not a pattern
    weight = REPORTER_BASE + 0.40 * net * shrink
    weight = round(max(REPORTER_FLOOR, min(REPORTER_CEILING, weight)), 3)
    return ReporterCalibration(
        weight=weight, corroborated=corr, contradicted=contra, n_reports=len(proxies),
        detail=(f"{corr} of {n} change-reports corroborated by a contemporaneous nurse check "
                f"({contra} contradicted). Reporter weight {weight:.2f} "
                f"(base {REPORTER_BASE:.2f}, floor {REPORTER_FLOOR:.2f}, ceiling {REPORTER_CEILING:.2f}). "
                f"Escalation path is unaffected: every report still buys a nurse re-check."),
    )


# ---------------------------------------------------------------------------
# Discordance
# ---------------------------------------------------------------------------


def discordance(
    subscores: dict[Channel, ChannelSubScore], reporter: ReporterCalibration
) -> Discordance:
    nurse = subscores.get(Channel.NURSE)
    camera = subscores.get(Channel.CAMERA)
    attendant = subscores.get(Channel.ATTENDANT)

    objective_parts = [s.concern for s in (nurse, camera) if s is not None]
    if not objective_parts and attendant is None:
        return Discordance(narrative="No channel produced comparable observations.")

    # WORST, not mean. Averaging the objective channels lets a channel that
    # happened to produce one trivially normal fresh item - a pain score of 1
    # after every vital sign has gone stale - halve the concern of a channel
    # that is actually seeing something. That is not fusion, it is dilution,
    # and it would let the attendant channel appear to disagree with sensors
    # that in fact agree with it.
    objective = max(objective_parts) if objective_parts else 0.0
    d = Discordance()
    pairs: list[tuple[str, str, float]] = []

    if nurse is not None and camera is not None:
        gap = abs(nurse.concern - camera.concern)
        pairs.append(("nurse", "camera", round(gap, 3)))

    if attendant is not None and objective_parts:
        gap = attendant.concern - objective
        pairs.append(("attendant", "objective", round(gap, 3)))
        if gap > 0:
            # The family sees something the instruments do not. Scaled by the
            # reporter's learned calibration.
            d.escalating = round(gap * reporter.weight, 3)
        else:
            # The instruments alarm and the family says this is baseline. This
            # DAMPENS only, and is capped so it can never cancel a critical
            # objective finding outright.
            d.de_escalating = round(min(0.35, -gap * 0.6), 3)

    spread = max((p[2] for p in pairs), default=0.0)
    d.magnitude = round(min(1.0, spread), 3)
    d.agreement = round(max(0.0, 1.0 - spread), 3)
    d.pairs = pairs

    bits = []
    if d.escalating > 0.05:
        bits.append(
            f"attendant channel reports a change the nurse and camera channels do not show "
            f"(gap {pairs[-1][2]:+.2f}, reporter weight {reporter.weight:.2f})")
    if d.de_escalating > 0.05:
        bits.append(
            f"objective channels read higher than the attendant, who reports the patient is at "
            f"baseline (gap {pairs[-1][2]:+.2f}) - this dampens escalation and cannot lower acuity")
    if nurse is not None and camera is not None and abs(nurse.concern - camera.concern) > 0.25:
        bits.append(f"nurse concern {nurse.concern:.2f} vs camera concern {camera.concern:.2f}")
    d.narrative = "; ".join(bits) if bits else "Channels broadly agree."
    return d


# ---------------------------------------------------------------------------
# Trajectory - level vs rate of change, with materiality gating
# ---------------------------------------------------------------------------


def trajectories(
    contract: Contract, evidence: list[Evidence], window_minutes: float = 30.0
) -> list[Trajectory]:
    """
    Rate of change, computed so that it means something.

    Three decisions here, each because the naive version produces false alarms
    that would destroy the system's credibility inside a week:

      1. STALE-AS-A-VALUE IS NOT STALE-AS-A-TREND. A respiratory rate from
         twenty minutes ago is too old to report as "the patient's rate now",
         and is exactly what you need to know what it WAS twenty minutes ago.
         Trends are therefore computed over everything that passed its QUALITY
         floor, whether or not it has since passed its staleness horizon. What
         is excluded is anything the sensor could not measure properly - which
         is a different thing entirely.

      2. NEVER MIX ACQUISITION METHODS IN ONE DELTA. A nurse's manual count at
         arrival minus a camera window now is not a trend, it is the difference
         between two instruments. Each (concept, method) pair gets its own
         trajectory and the most reliable one with enough readings wins.

      3. THE EVIDENCE BAR SCALES WITH INSTRUMENT NOISE. The contract's
         materiality threshold is a CLINICAL constant: a 4-breath change matters
         in an adult, full stop. But the confidence with which you may assert
         that a 4-breath change occurred depends on what measured it. rPPG
         respiratory rate carries reliability 0.35 and a per-window error of
         several breaths; two noisy windows will differ by 5 or 6 for no reason
         at all. So the bar this instrument must clear is

             effective = clinical_threshold / reliability ** 0.35

         which leaves a nurse count at 4 and raises rPPG to roughly 5.8. Values
         are additionally median-smoothed at each end of the window. Without
         both, this function reports respiratory-rate "deterioration" on well
         patients, which is precisely the alarm-fatigue failure mode.
    """
    trackable = ("HEART_RATE", "RESP_RATE", "SPO2", "SBP", "STILLNESS_MINUTES", "TEMP")
    groups: dict[tuple[str, str], list[Evidence]] = {}
    for e in evidence:
        if e.concept_id not in trackable:
            continue
        if not e.signal_quality.passed_floor:
            continue
        groups.setdefault((e.concept_id, e.acquisition_method.value), []).append(e)

    best: dict[str, Trajectory] = {}
    for (concept, method), items in groups.items():
        if len(items) < 4:
            continue
        items.sort(key=lambda e: e.observed_at)
        latest = items[-1]
        cutoff = latest.observed_at.timestamp() - window_minutes * 60
        items = [e for e in items if e.observed_at.timestamp() >= cutoff]
        if len(items) < 4:
            continue

        vals: list[tuple[float, Evidence]] = []
        for e in items:
            try:
                vals.append((float(e.value), e))
            except (TypeError, ValueError):
                continue
        if len(vals) < 4:
            continue

        minutes = (vals[-1][1].observed_at - vals[0][1].observed_at).total_seconds() / 60.0
        if minutes < 2.0:
            continue

        # THEIL-SEN slope: the median of all pairwise slopes separated by at
        # least three minutes. Chosen over a median-of-endpoints difference
        # because that estimator LAGS on a ramp - it compares the middle of the
        # first few readings with the middle of the last few, and so
        # systematically under-reads a quantity that is still climbing, which
        # is the only situation we care about. Theil-Sen is unbiased on a ramp
        # and still ignores the outlier windows that a noisy rPPG estimate
        # produces several times an hour.
        slopes: list[float] = []
        t0 = vals[0][1].observed_at.timestamp()
        pts = [(v, (e.observed_at.timestamp() - t0) / 60.0) for v, e in vals]
        for i in range(len(pts)):
            vi, ti = pts[i]
            for j in range(i + 1, len(pts)):
                vj, tj = pts[j]
                if tj - ti >= 3.0:
                    slopes.append((vj - vi) / (tj - ti))
        if not slopes:
            continue
        slopes.sort()
        slope = slopes[len(slopes) // 2]
        delta = slope * minutes

        k = max(2, min(5, len(vals) // 3))
        head = sorted(v for v, _ in vals[:k])
        start_v = head[len(head) // 2]
        end_v = start_v + delta

        clinical = contract.materiality_threshold(concept) or 0.0
        rel = contract.reliability_weight(method)
        effective = clinical / max(0.05, rel) ** 0.35

        t = Trajectory(
            concept_id=concept,
            delta=round(delta, 1),
            minutes=round(minutes, 1),
            material=abs(delta) >= effective,
            materiality_threshold=clinical,
            direction="rising" if delta > 0 else ("falling" if delta < 0 else "flat"),
            evidence_ids=[vals[0][1].evidence_id, vals[-1][1].evidence_id],
            acquisition_method=method,
            reliability_weight=rel,
            effective_threshold=round(effective, 2),
            n_points=len(vals),
            start_value=round(start_v, 1),
            end_value=round(end_v, 1),
        )
        # Most reliable instrument wins for a given concept.
        cur = best.get(concept)
        if cur is None or rel > cur.reliability_weight:
            best[concept] = t

    out = list(best.values())
    out.sort(key=lambda t: -abs(t.delta / max(1e-6, t.effective_threshold or 1.0)))
    return out


def has_material_worsening(trajs: list[Trajectory]) -> bool:
    """
    Is any measured quantity moving in the wrong direction by more than the
    evidence bar for the instrument that measured it?

    This gates the de-escalating discordance path. A family who say their
    relative is at baseline may prevent an escalation founded only on a
    borderline model probability. They may NOT talk down an observed, material,
    worsening trend - and this is the function that enforces the difference.
    """
    for t in trajs:
        if not t.material:
            continue
        if ((t.concept_id in ("HEART_RATE", "RESP_RATE", "TEMP", "STILLNESS_MINUTES") and t.delta > 0)
                or (t.concept_id in ("SPO2", "SBP") and t.delta < 0)):
            return True
    return False


def trajectory_boost(trajs: list[Trajectory]) -> float:
    """Scalar trajectory contribution to EWER. Only MATERIAL changes count."""
    boost = 0.0
    for t in trajs:
        if not t.material:
            continue
        ratio = abs(t.delta) / max(1e-6, t.materiality_threshold or 1.0)
        worsening = (
            (t.concept_id in ("HEART_RATE", "RESP_RATE", "TEMP", "STILLNESS_MINUTES") and t.delta > 0)
            or (t.concept_id in ("SPO2", "SBP") and t.delta < 0)
        )
        if worsening:
            boost += min(0.25, 0.10 * ratio)
    return round(min(0.5, boost), 3)


# ---------------------------------------------------------------------------
# Contradictory evidence
# ---------------------------------------------------------------------------


def split_supporting_contradicting(
    usable: list[Evidence], escalating: bool
) -> tuple[list[Evidence], list[Evidence]]:
    """
    Retrieve evidence FOR and evidence AGAINST the leading assessment.

    The nurse card has a populated 'what argues against this' region on every
    patient. Surfacing only confirmatory evidence is how a decision-support tool
    turns into an anchoring device.
    """
    supporting: list[Evidence] = []
    contradicting: list[Evidence] = []
    for ev in _latest_per_source(usable):
        if ev.concept_id not in CONCERN_CONCEPTS:
            continue
        sev = band_severity(ev.threshold_band)
        if escalating:
            if sev >= 2:
                supporting.append(ev.model_copy(update={"supports_or_contradicts": SupportDirection.SUPPORTS}))
            elif ev.threshold_band is ThresholdBand.NORMAL:
                contradicting.append(ev.model_copy(update={"supports_or_contradicts": SupportDirection.CONTRADICTS}))
        else:
            if ev.threshold_band is ThresholdBand.NORMAL:
                supporting.append(ev.model_copy(update={"supports_or_contradicts": SupportDirection.SUPPORTS}))
            elif sev >= 2:
                contradicting.append(ev.model_copy(update={"supports_or_contradicts": SupportDirection.CONTRADICTS}))

    # Most severe, most reliable, and MOST RECENT first. Sorting ascending on
    # observed_at here would make the per-concept dedupe below keep the oldest
    # reading of each concept, which is the opposite of what a nurse needs.
    supporting.sort(key=lambda e: (-band_severity(e.threshold_band),
                                   -e.reliability_weight, -e.observed_at.timestamp()))
    contradicting.sort(key=lambda e: (-e.reliability_weight, -e.observed_at.timestamp()))
    return _dedupe(supporting), _dedupe(contradicting)


def _latest_per_source(items: list[Evidence]) -> list[Evidence]:
    """
    Keep only the most recent observation of each concept FROM EACH SOURCE.

    A superseded reading is history, not counter-evidence. Without this, a
    family who answered "no, not more sleepy" at T+10 and "yes, more sleepy" at
    T+30 has their stale answer retrieved into the 'what argues against this'
    panel, where it reads to a nurse as a live rebuttal of the escalation it
    actually caused.

    Note that this is per (concept, acquisition_method), NOT per concept. Two
    DIFFERENT sources disagreeing about the same concept right now is real
    discordance and must still reach the card - that disagreement is the signal.
    What is suppressed is only a single source contradicting its own older self.
    """
    best: dict[tuple[str, object], Evidence] = {}
    for e in items:
        k = (e.concept_id, e.acquisition_method)
        if k not in best or e.observed_at > best[k].observed_at:
            best[k] = e
    return list(best.values())


def _dedupe(items: list[Evidence]) -> list[Evidence]:
    """One row per concept, keeping the highest-priority source after sorting."""
    seen: set[str] = set()
    out: list[Evidence] = []
    for e in items:
        if e.concept_id in seen:
            continue
        seen.add(e.concept_id)
        out.append(e)
    return out
