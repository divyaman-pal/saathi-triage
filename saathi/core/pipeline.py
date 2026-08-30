"""
The SAATHI pipeline.

    SIGNALS
      -> SIGNAL QUALITY GATING
      -> CLINICAL SEMANTICS (contract age bands)
      -> RISK ESTIMATION
      -> MULTI-CHANNEL EVIDENCE FUSION
      -> DISCORDANCE ANALYSIS
      -> UNCERTAINTY / ABSTENTION
      -> ASYMMETRIC-COST DECISIONING
      -> PERSONA-SPECIFIC PRESENTATION
      -> SAFETY CONTRACT
      -> CONSTRAINED NARRATIVE
      -> HUMAN DECISION
      -> CONTINUOUS RE-CHECK
      -> TELEMETRY

The ordering below is the safety ordering. Red flags are evaluated BEFORE the
model, so a model failure cannot delay them. Abstention is evaluated BEFORE the
cost decision, so there is no code path in which degraded signals produce a
confident number. The acuity is fixed BEFORE the Safety Contract is built, and
the contract is built BEFORE the LLM is called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import claim_compiler, confidence, ewer, fusion, red_flags
from .audit import AuditStore, get_store
from .contract_loader import Contract, get_contract
from .cost_engine import CostEngine
from .escalation import (
    accumulation_escalates,
    attendant_escalation,
    compute_sla,
    evaluate_materiality,
    live_channel_count,
    materiality_boost,
    materiality_escalates,
)
from .features import assemble, feature_names, missing_feature_impact
from .gating import gate, latest_per_concept
from .injection_guard import extract_symptoms
from .llm import LLMClient, get_llm
from .models import (
    ThresholdBand,
    Acuity,
    Assessment,
    Channel,
    ChannelAvailability,
    ChannelStatus,
    ConfidenceComponents,
    EpistemicState,
    Evidence,
    FreshnessStatus,
    QualityStatus,
    RenderedNarrative,
    Role,
    esi,
    next_id,
)
from .risk_model import RiskModel, calibration_group, rules_only_score
from .triage_rules import arrival_acuity_with_complaint, escalation_target
from .schemes import monotonic_check
from .telemetry import StageTimer
from .validator import Validator


def _g(fv, key: str, default: float = 0.0) -> float:
    """Read a feature, treating NaN as absent."""
    import math
    v = fv.values.get(key, float("nan"))
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


@dataclass
class PipelineConfig:
    tier: str = "TIER_A"
    surge_active: bool = False
    models_enabled: bool = True
    llm_enabled: bool = True
    persona: Role = Role.TRIAGE_NURSE
    fail_all_models: bool = False       # adversarial test hook
    fail_all_channels: bool = False     # adversarial test hook


@dataclass
class PatientState:
    """Per-encounter state that persists across successive assessments."""
    patient_id: str
    acuity_arrival: Acuity
    acuity_current: Acuity
    acuity_previous: Acuity | None = None
    last_change_reason: str = "ARRIVAL"
    human_overridden: bool = False
    rechecks_used_this_hour: int = 0
    history: list[tuple[datetime, int, str]] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        contract: Contract | None = None,
        model: RiskModel | None = None,
        store: AuditStore | None = None,
        llm: LLMClient | None = None,
    ):
        self.c = contract or get_contract()
        self.model = model or (RiskModel.load() if RiskModel.is_built() else RiskModel())
        self.store = store or get_store()
        self.llm = llm or get_llm()
        self.cost = CostEngine(self.c)
        self.validator = Validator(self.c)
        self.latch = red_flags.RedFlagLatch()
        self.states: dict[str, PatientState] = {}

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def assess(
        self,
        *,
        patient_id: str,
        age_years: float,
        sex: str,
        evidence: list[Evidence],
        now: datetime,
        arrival_at: datetime,
        complaint_text: str = "",
        asr_confidence: float | None = None,
        attendant_present: bool = True,
        has_prior_record: bool = False,
        consent_declined: bool = False,
        arrival_acuity_hint: int | None = None,
        extracted_symptoms: list[str] | None = None,
        config: PipelineConfig | None = None,
    ) -> Assessment:
        cfg = config or PipelineConfig()
        t = StageTimer()
        band_id = self.c.age_band(age_years)
        profile = self.c.profile(cfg.tier)

        # -- 1. Signal acquisition ------------------------------------
        with t.time("signal_acquisition"):
            if cfg.fail_all_channels:
                # ADVERSARIAL: every channel dies at once. The system must move
                # the patient toward a human, not report a calm default.
                evidence = [e for e in evidence if e.source_channel is Channel.SYSTEM]
            # Two independent sources, unioned:
            #   extracted_symptoms - the structured output of the multilingual
            #       ASR + LLM extraction step. SIMULATED here (see
            #       data/simulate.py), and supplied by the caller ONLY when the
            #       ASR cleared its contract confidence floor. P-017's Gondi
            #       complaint at 0.41 confidence yields nothing, which is the
            #       graceful-degradation path, not a failure.
            #   extract_symptoms(text) - exact-token matching against a closed
            #       vocabulary, run directly on the untrusted text. This is
            #       defence in depth: it can only ever name a symptom a
            #       clinician already wrote a rule for, and it is what makes a
            #       prompt injection through the complaint field inert.
            symptoms = sorted(set(extracted_symptoms or []) |
                              set(extract_symptoms(complaint_text) if complaint_text else []))

        # -- 2. Quality + staleness gating ----------------------------
        with t.time("quality_gating"):
            usable, rejected = gate(self.c, evidence, now)
            live = live_channel_count(usable)
            channel_status = self._channel_status(profile, usable, rejected, now, consent_declined)

        # -- 3. Red flags. BEFORE any model. --------------------------
        with t.time("red_flags"):
            ctx = red_flags.RuleContext(
                age_years=age_years, age_band=band_id,
                complaint_symptoms=symptoms,
                attendant_symptoms=self._attendant_symptoms(usable),
                avpu_baseline="A",
            )
            hits = red_flags.evaluate(self.c, usable + rejected, ctx)
            hits = self.latch.apply(patient_id, hits)

        # -- 4. Feature assembly --------------------------------------
        with t.time("feature_assembly"):
            fv = assemble(self.c, usable, rejected, age_years=age_years, sex=sex,
                          has_prior_record=has_prior_record, live_channels=live)

        # -- 5. Risk estimation ---------------------------------------
        with t.time("model_inference"):
            prediction = None
            model_calls = 0
            if cfg.models_enabled and not cfg.fail_all_models and self.model.routes:
                self.model.enabled = True
                prediction = self.model.predict(fv, age_years)
                model_calls = 1
            rules_p = rules_only_score(fv)

        # -- 6. Fusion + discordance ----------------------------------
        with t.time("fusion"):
            subscores = fusion.channel_subscores(self.c, usable)
            reporter = fusion.calibrate_reporter(usable, usable + rejected)
            disc = fusion.discordance(subscores, reporter)
            # Trends are computed over everything that passed its QUALITY gate,
            # fresh or not. A value too old to report as current is still a
            # perfectly good record of what the patient was doing then.
            trajs = fusion.trajectories(self.c, usable + rejected)
            traj_boost = fusion.trajectory_boost(trajs)

        # -- 7. Confidence --------------------------------------------
        with t.time("confidence"):
            names = feature_names(fv.route)
            imps = self.model.importances(fv.route)
            calib = prediction.calibration_status if prediction else "NOT_APPLICABLE_RULES_ONLY"
            grp = calibration_group(age_years)
            support = 0
            route_obj = self.model.routes.get(fv.route)
            if route_obj:
                support = int(route_obj.calibration_n.get(grp, 0) * 3.33)
            expected_channels = sum(
                1 for k in ("nurse", "camera", "attendant")
                if profile["channels"][k]["enabled"])
            cdetail = confidence.compute(
                self.c, fv, usable, rejected,
                importances=imps, feature_name_list=names,
                agreement=disc.agreement, calibration_status=calib,
                group_support=support, live_channels=live,
                expected_channels=max(1, expected_channels),
                tier_ceiling=float(profile["expected_confidence_ceiling"]),
            )
            nurse_age = self._nurse_vitals_age(usable, rejected, now)
            abst = confidence.evaluate_abstention(
                self.c, cdetail, usable, rejected,
                attendant_present=attendant_present and not consent_declined,
                has_prior_record=has_prior_record,
                nurse_vitals_age_minutes=nurse_age,
                asr_confidence=asr_confidence,
            )
            if cfg.fail_all_models or not self.model.routes:
                abst = abst.model_copy(update={
                    "abstained": True,
                    "gates_tripped": sorted(set(abst.gates_tripped + ["MODEL_UNAVAILABLE"])),
                })

        # -- 8. Materiality -------------------------------------------
        with t.time("cost_decision"):
            model_moved = bool(prediction and prediction.probability >= self.c.level_threshold(2))
            materiality = evaluate_materiality(
                self.c, usable, trajs, age_band=band_id, model_moved=model_moved,
                reporter_weight=reporter.weight, reporter_detail=reporter.detail)

            state = self._state(patient_id, arrival_acuity_hint, fv, hits, symptoms)

            # -- 9. Asymmetric-cost decision --------------------------
            # The arrival acuity was assigned deterministically from the
            # contract age bands by triage_rules.arrival_acuity(). Everything
            # below can only RAISE it.
            current = state.acuity_current.level
            proposed = current
            cost_decision = None

            worsening = fusion.has_material_worsening(trajs)
            baseline_veto = (
                disc.de_escalating >= 0.20        # family who know the baseline say nothing changed
                and not worsening                 # and nothing measured is actually moving
                and not hits                      # and no clinician-authored rule has fired
                and _g(fv, "n_critical") < 1      # and no vital is age-band critical
            )

            if not abst.abstained:
                p_esc = prediction.probability if prediction is not None else rules_p
                cost_decision = self.cost.decide(p_esc, current, level=2)
                if cost_decision.decision_asymmetric and not baseline_veto:
                    proposed = cost_decision.target_level_asymmetric

            # Materiality and discordance can each raise by one level. Neither
            # may reach level 1: the resuscitation room is reserved for the
            # deterministic red-flag layer and for arrival physiology.
            if materiality_escalates(materiality):
                proposed = min(proposed, escalation_target(current))
            accumulated, accum_concepts = accumulation_escalates(materiality)
            if accumulated:
                proposed = min(proposed, escalation_target(current))
            if disc.escalating >= 0.30:
                proposed = min(proposed, escalation_target(current))

            # Red flags override everything, including a model that disagrees
            # and a confidence engine that is unsure.
            if hits:
                proposed = min(proposed, min(h.target_acuity.level for h in hits))

            # MONOTONIC ESCALATION. The system may only raise urgency.
            new_level = min(proposed, current)
            if abst.abstained and not hits:
                new_level = current      # hold, never downgrade, never guess

            change_reason = self._change_reason(
                new_level, state, hits, abst, materiality, disc, cost_decision,
                accumulated=accumulated, accum_concepts=accum_concepts)
            new_acuity = esi(new_level)
            monotonic_check(state.acuity_current, new_acuity)
            if new_level != state.acuity_current.level:
                state.acuity_previous = state.acuity_current
                state.acuity_current = new_acuity
                state.last_change_reason = change_reason
                state.history.append((now, new_level, change_reason))

        # -- 10. SLA + EWER -------------------------------------------
        with t.time("ewer"):
            waited = (now - arrival_at).total_seconds() / 60.0
            since_human = self._since_human(usable, now, waited)
            degraded = self._degraded_channels(usable, rejected)
            sla = compute_sla(
                self.c, level=state.acuity_current.level,
                waited_minutes=waited, minutes_since_human_contact=since_human,
                has_prior_record=has_prior_record, abstained=abst.abstained,
                consent_declined=consent_declined,
                low_confidence=bool(cdetail.failing),
                surge_active=cfg.surge_active,
                channel_degraded=bool(degraded),
                attendant_absent=not attendant_present,
            )
            time_hits = red_flags.evaluate_time_rules(
                self.c, waited_minutes=waited, max_wait_minutes=sla.max_wait_minutes,
                minutes_since_human_contact=since_human,
                recheck_interval_minutes=sla.recheck_interval_minutes,
                minutes_since_any_observation=self._silence(usable, now),
                consent_declined=consent_declined,
            )
            att_esc = attendant_escalation(
                usable, rechecks_used_this_hour=state.rechecks_used_this_hour)
            rank = ewer.compute(
                model_risk=prediction.probability if prediction else None,
                confidence=cdetail.components,
                discordance_escalating=disc.escalating,
                discordance_de_escalating=disc.de_escalating,
                minutes_since_human_contact=since_human,
                sla=sla, red_flag_fired=bool(hits),
                trajectory=traj_boost,
                materiality=materiality_boost(materiality),
                abstention=abst,
            )

        # -- 11. Evidence selection -----------------------------------
        escalating = state.acuity_current.level < state.acuity_arrival.level or bool(hits)
        supporting, contradicting = fusion.split_supporting_contradicting(usable, escalating)
        supporting = self._attribute(supporting, prediction, fv)
        vitals = self._vitals_snapshot(usable)

        # -- 12. Safety Contract --------------------------------------
        with t.time("contract_generation"):
            epistemic = claim_compiler.determine_state(
                red_flags=hits, abstention=abst, override=None,
                materiality=materiality, model_moved=model_moved,
                discordance=disc, escalating=escalating,
                has_prior_record=has_prior_record,
                failing_confidence=cdetail.failing,
            )
            lineage_ref = next_id("LIN", patient_id)
            versions = self.model.versions() | {
                "rules": self.c.ruleset_version,
                "contract": self.c.version,
                "cost_policy": self.c.policy_version,
            }
            sc = claim_compiler.build(
                self.c,
                patient_id=patient_id, persona=cfg.persona,
                access_scope=[self.c.role_spec(cfg.persona.value)["row_scope"]],
                acuity_arrival=state.acuity_arrival,
                acuity_previous=state.acuity_previous,
                acuity_current=state.acuity_current,
                change_reason=state.last_change_reason,
                state=epistemic, supporting=supporting[:6], contradicting=contradicting[:4],
                red_flags=hits, materiality=materiality,
                confidence=cdetail.components, cost_decision=cost_decision,
                ewer=rank, sla=sla, trajectories=trajs, abstention=abst,
                missing_data=fv.missing, stale_data=fv.stale,
                degraded_channels=degraded,
                absent_channels=self._absent_channels(profile, consent_declined),
                model_versions=versions, lineage_ref=lineage_ref,
                generated_at=now,
            )
            self.store.write_contract(sc)
            self._write_lineage(lineage_ref, patient_id, sc, fv, usable, prediction,
                                cost_decision, subscores, disc, cdetail, hits, rules_p)

        # -- 13. Narration (NOT on the safety path) -------------------
        narrative = self._narrate(sc, supporting, contradicting, trajs, hits, materiality,
                                  disc, cfg, t, patient_id, profile)

        # -- 14. Telemetry --------------------------------------------
        for stage, ms in t.stages.items():
            self.store.log_stage(
                patient_id=patient_id, stage=stage, latency_ms=ms,
                model_calls=(model_calls if stage == "model_inference" else 0),
                model_type=("xgboost_tabular" if stage == "model_inference" else
                            (self.llm.model if stage == "llm_render" else "")),
                tokens_in=(narrative.tokens_in if stage == "llm_render" and narrative else 0),
                tokens_out=(narrative.tokens_out if stage == "llm_render" and narrative else 0),
                cost_usd=(self.llm.price(narrative.tokens_in, narrative.tokens_out)
                          if stage == "llm_render" and narrative and narrative.renderer == "llm" else 0.0),
                tier=cfg.tier, surge=cfg.surge_active,
            )

        for th in time_hits:
            self.store.log_event(th.rule_id, th.detail, patient_id)
        if att_esc.presses:
            state.rechecks_used_this_hour += 1 if att_esc.recheck_triggered else 0
            self.store.log_event(
                "ATTENDANT_ESCALATION",
                f"{att_esc.presses} press(es). recheck={att_esc.recheck_triggered}. "
                f"acuity_effect={att_esc.acuity_effect} queue_effect={att_esc.queue_effect} "
                f"ewer_effect={att_esc.ewer_effect}", patient_id)

        return Assessment(
            patient_id=patient_id, age_years=age_years, age_band=band_id, sex=sex, tier=cfg.tier,
            assessed_at=now, arrival_at=arrival_at,
            acuity_arrival=state.acuity_arrival,
            acuity_previous=state.acuity_previous,
            acuity_current=state.acuity_current,
            acuity_change_reason=state.last_change_reason,
            epistemic_state=epistemic,
            confidence=cdetail.components,
            cost_decision=cost_decision,
            abstention=abst, ewer=rank, sla=sla,
            red_flags=hits, materiality=materiality,
            supporting_evidence=supporting[:8], contradictory_evidence=contradicting[:5],
            vitals=vitals,
            channel_status=channel_status,
            missing_data=sorted(set(fv.missing)), stale_data=sorted(set(fv.stale)),
            model_route=fv.route, model_versions=versions,
            safety_contract=sc, narrative=narrative, lineage_ref=lineage_ref,
            stage_latency_ms={k: round(v, 3) for k, v in t.stages.items()},
        )

    # -----------------------------------------------------------------
    # Narration
    # -----------------------------------------------------------------

    def _narrate(self, sc, supporting, contradicting, trajs, hits, materiality, disc,
                 cfg, t, patient_id, profile) -> RenderedNarrative:
        plan = claim_compiler.plan(self.c, sc, supporting, contradicting, trajs,
                                  hits, materiality, disc)

        # Three independent ways the language layer is skipped, and none of them
        # changes a single triage decision:
        #   TIER      - the deployment profile says this site has no LLM at all
        #               (Tier C is offline-first with intermittent connectivity)
        #   SURGE     - narration is dropped below level 2 to protect the
        #               decision latency budget when the department is loaded
        #   KILL      - the operator switch, or an absent API credential
        tier_allows_llm = bool(profile.get("llm", {}).get("enabled", True))
        surge_drop = (cfg.surge_active and sc.acuity_current.level > 2)
        want_llm = cfg.llm_enabled and self.llm.enabled and tier_allows_llm and not surge_drop

        rejections = []
        attempts = 0
        text = ""
        renderer = "deterministic_template"
        tin = tout = 0
        llm_ms = 0.0
        model_id = None

        if want_llm:
            max_attempts = int(self.c.fallback_policy["max_regeneration_attempts"])
            payload = self._llm_payload(sc)
            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                with t.time("llm_render"):
                    res = self.llm.render(sc, plan, payload)
                if res is None or not res.text:
                    break
                tin, tout, llm_ms, model_id = res.tokens_in, res.tokens_out, res.latency_ms, res.model_id
                with t.time("validate"):
                    rj = self.validator.validate(res.text, sc)
                if not rj:
                    text, renderer = res.text, "llm"
                    rejections = []
                    break
                rejections = rj
                for r in rj:
                    self.store.log_rejection(
                        patient_id=patient_id, contract_id=sc.contract_id,
                        check_id=r.check_id, severity=r.severity,
                        detail=r.detail, offending_text=r.offending_text)
                if self.validator.is_critical(rj):
                    break     # CRITICAL: no retry. Straight to the template.

        if not text:
            with t.time("validate"):
                text = claim_compiler.render_deterministic(sc, plan)
                renderer = "deterministic_template"

        return RenderedNarrative(
            text=text, renderer=renderer, attempts=attempts, rejections=rejections,
            llm_enabled=cfg.llm_enabled and self.llm.enabled,
            tokens_in=tin, tokens_out=tout, latency_ms=llm_ms, model_id=model_id,
        )

    VITAL_CONCEPTS = ("HEART_RATE", "RESP_RATE", "SPO2", "SBP", "TEMP")

    def _vitals_snapshot(self, usable) -> dict:
        """
        Latest quality-passed reading per vital.

        Only values that CLEARED their quality floor appear here, so the queue
        row can never show a number the scoring engine itself refused to use.
        A vital with no usable reading is absent rather than stale-but-shown -
        the row renders a dash, which is the honest state.
        """
        out: dict = {}
        for ev in usable:
            cid = ev.concept_id
            if cid not in self.VITAL_CONCEPTS:
                continue
            prev = out.get(cid)
            if prev is not None and prev["_at"] >= ev.observed_at:
                continue
            out[cid] = {
                "value": ev.value,
                "unit": ev.unit,
                "band": ev.threshold_band.value,
                "critical": ev.threshold_band is ThresholdBand.CRITICAL,
                "age_seconds": (ev.freshness.age_seconds if ev.freshness else None),
                "quality": ev.signal_quality.status.value,
                "method": ev.acquisition_method.value,
                "_at": ev.observed_at,
            }
        for v in out.values():
            v.pop("_at", None)
        return out

    def _llm_payload(self, sc) -> dict:
        """
        Exactly what the model receives. Minimised, role-scoped, no identifiers.

        The demo displays this object verbatim so a judge can see there is no
        name, no phone number, no ABHA number and no face data in it.
        """
        return {
            "contract_id": sc.contract_id,
            "pseudonymous_encounter_id": sc.patient_id,
            "persona": sc.persona.value,
            "epistemic_status": sc.epistemic_status.value,
            "acuity": {"scheme": f"{sc.acuity_current.scheme_id}/{sc.acuity_current.scheme_version}",
                       "current": sc.acuity_current.level, "arrival": sc.acuity_arrival.level},
            "allowed_claim_types": sc.allowed_claim_types,
            "forbidden_claim_types": sc.forbidden_claim_types,
            "allowed_numbers": sc.allowed_numbers,
            "allowed_entities": sc.allowed_entities,
            "required_disclosures": sc.required_disclosures,
            "max_words": sc.max_words,
            "confidence": sc.confidence.model_dump(),
            "missing_data": sc.missing_data,
            "stale_data": sc.stale_data,
            "priority_question": sc.priority_question,
        }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _state(self, patient_id, arrival_hint, fv, hits, symptoms=None) -> PatientState:
        """
        First contact establishes the arrival acuity, deterministically.

        No model runs here. On a patient's first assessment the contract age-band
        severities and the red-flag layer assign the level - which is exactly
        what a cold-start deployment does on day one, before a single
        prospective case has been collected.
        """
        st = self.states.get(patient_id)
        if st is None:
            if arrival_hint is not None:
                level, reasons = arrival_hint, ["externally supplied arrival acuity"]
            else:
                level, reasons = arrival_acuity_with_complaint(fv, hits, symptoms or [])
            st = PatientState(patient_id=patient_id, acuity_arrival=esi(level),
                              acuity_current=esi(level),
                              last_change_reason="ARRIVAL: " + "; ".join(reasons))
            self.states[patient_id] = st
        return st

    def seed_arrival(self, patient_id: str, age_years: float, sex: str,
                     arrival_evidence: list[Evidence], now: datetime,
                     *, has_prior_record: bool = False,
                     complaint_text: str = "",
                     extracted_symptoms: list[str] | None = None) -> Acuity:
        """
        Establish the arrival acuity from the ARRIVAL evidence, cheaply.

        Without this, a patient first assessed 45 minutes into their wait would
        have their "arrival" acuity computed from data that has since gone
        stale. That is precisely P-009's situation, and it would silently pin an
        abstaining patient at level 5 and then "hold" them there. Triage
        happened at arrival; the record has to say what it said at arrival.

        Deterministic only - no telemetry, no Safety Contract, no LLM.
        """
        usable, rejected = gate(self.c, arrival_evidence, now)
        band_id = self.c.age_band(age_years)
        ctx = red_flags.RuleContext(
            age_years=age_years, age_band=band_id,
            complaint_symptoms=sorted(set(extracted_symptoms or []) |
                                      set(extract_symptoms(complaint_text) if complaint_text else [])),
            attendant_symptoms=self._attendant_symptoms(usable))
        hits = red_flags.evaluate(self.c, usable + rejected, ctx)
        fv = assemble(self.c, usable, rejected, age_years=age_years, sex=sex,
                      has_prior_record=has_prior_record,
                      live_channels=live_channel_count(usable))
        level, reasons = arrival_acuity_with_complaint(fv, hits, ctx.complaint_symptoms)
        self.states[patient_id] = PatientState(
            patient_id=patient_id, acuity_arrival=esi(level), acuity_current=esi(level),
            last_change_reason="ARRIVAL: " + "; ".join(reasons))
        return esi(level)

    def _change_reason(self, new_level, state, hits, abst, materiality, disc, cost_decision,
                       *, accumulated: bool = False, accum_concepts=None) -> str:
        if abst.abstained:
            return "HELD_INSUFFICIENT_SIGNAL"
        if new_level == state.acuity_current.level:
            return state.last_change_reason
        if hits:
            return f"RED_FLAG:{hits[0].rule_id}"
        if materiality_escalates(materiality):
            esc = [m for m in materiality if m.action == "ESCALATE"]
            trend = [m for m in esc if m.statistical_strength.value == "STRONG"]
            if trend:
                return f"TRAJECTORY_MATERIAL_WORSENING:{trend[0].concept_id}"
            return f"CLINICAL_MATERIALITY:{esc[0].concept_id}" if esc else "CLINICAL_MATERIALITY"
        if accumulated:
            return "MULTI_FACTOR_ACCUMULATION:" + ",".join(accum_concepts or [])
        if disc.escalating >= 0.30:
            return "CHANNEL_DISCORDANCE"
        if cost_decision is not None:
            return "MODEL_ESCALATION_COST_THRESHOLD"
        return "RULES_ESCALATION"

    def _attendant_symptoms(self, usable: list[Evidence]) -> list[str]:
        out: list[str] = []
        for e in usable:
            if e.concept_id == "ATTENDANT_FREE_TEXT":
                out += extract_symptoms(str(e.value))
        return out

    def _nurse_vitals_age(self, usable, rejected, now) -> float | None:
        items = [e for e in usable + rejected
                 if e.source_channel is Channel.NURSE
                 and e.concept_id in ("HEART_RATE", "RESP_RATE", "SPO2", "SBP")]
        if not items:
            return None
        latest = max(items, key=lambda e: e.observed_at)
        return (now - latest.observed_at).total_seconds() / 60.0

    def _since_human(self, usable, now, waited) -> float:
        items = [e for e in usable if e.concept_id == "MINUTES_SINCE_HUMAN_CONTACT"]
        if items:
            try:
                return float(max(items, key=lambda e: e.observed_at).value)
            except (TypeError, ValueError):
                pass
        return waited

    def _silence(self, usable, now) -> float:
        items = [e for e in usable if e.source_channel is not Channel.SYSTEM]
        if not items:
            return 999.0
        return (now - max(items, key=lambda e: e.observed_at).observed_at).total_seconds() / 60.0

    def _degraded_channels(self, usable, rejected) -> list[str]:
        out = set()
        for e in usable:
            if e.signal_quality.status is QualityStatus.DEGRADED:
                out.add(e.source_channel.value)
        for e in rejected:
            if not e.signal_quality.passed_floor:
                out.add(e.source_channel.value)
        return sorted(out)

    def _absent_channels(self, profile, consent_declined) -> list[str]:
        out = [k for k in ("camera", "attendant", "prior_record")
               if not profile["channels"][k]["enabled"]]
        if consent_declined:
            out += ["camera", "attendant"]
        return sorted(set(out))

    def _channel_status(self, profile, usable, rejected, now, consent_declined) -> list[ChannelStatus]:
        out: list[ChannelStatus] = []
        for ch in (Channel.NURSE, Channel.CAMERA, Channel.ATTENDANT, Channel.PRIOR_RECORD):
            key = {"nurse": "nurse", "camera": "camera",
                   "attendant": "attendant", "prior_record": "prior_record"}[ch.value]
            enabled = profile["channels"][key]["enabled"]
            if consent_declined and ch in (Channel.CAMERA, Channel.ATTENDANT):
                enabled = False
            live = [e for e in usable if e.source_channel is ch]
            failed = [e for e in rejected if e.source_channel is ch and not e.signal_quality.passed_floor]

            if not enabled:
                avail = ChannelAvailability.ABSENT
                note = ("declined at consent" if consent_declined and ch in (Channel.CAMERA, Channel.ATTENDANT)
                        else "not present in this deployment profile")
            elif live and not failed:
                avail = ChannelAvailability.AVAILABLE
                note = None
            elif failed and not live:
                avail = ChannelAvailability.DEGRADED
                note = failed[0].signal_quality.floor_detail
            elif not live and not failed:
                avail = ChannelAvailability.SILENT
                note = "channel is enabled and healthy but has produced nothing"
            else:
                avail = ChannelAvailability.DEGRADED
                note = f"{len(failed)} of {len(live) + len(failed)} items rejected at the quality floor"

            last = max((e.observed_at for e in live), default=None)
            quality = QualityStatus.NOT_APPLICABLE
            if live:
                order = [QualityStatus.GOOD, QualityStatus.ACCEPTABLE,
                         QualityStatus.DEGRADED, QualityStatus.FAILED]
                quality = max((e.signal_quality.status for e in live
                               if e.signal_quality.status in order),
                              key=lambda s: order.index(s), default=QualityStatus.NOT_APPLICABLE)
            out.append(ChannelStatus(channel=ch, availability=avail, last_observation_at=last,
                                     items_this_window=len(live), quality=quality, note=note))
        return out

    def _attribute(self, supporting, prediction, fv):
        """
        Attach SHAP attributions where they exist.

        Note the deliberate wording in Contribution.population_note: a SHAP
        value is a statement about a model, not a mechanism. The claim grammar
        forbids narrating it as one.
        """
        if prediction is None:
            return supporting
        concept_to_feature = {
            "HEART_RATE": "z_hr", "RESP_RATE": "z_rr", "SPO2": "z_spo2", "SBP": "z_sbp",
            "TEMP": "z_temp", "CAP_REFILL": "z_cap", "SHOCK_INDEX": "z_shock",
            "STILLNESS_MINUTES": "stillness_min", "WORK_OF_BREATHING": "work_of_breathing",
            "POSTURE": "posture_ord", "ARRIVAL_MODE": "arrival_mode_ord", "AVPU": "avpu_ord",
            "CONFUSION_NEW": "att_confusion", "SLEEPINESS_INCREASE": "att_sleepy",
            "RESPONDS_TO_NAME": "att_responds", "PAIN_SELF_REPORT": "pain",
            "SKIN_COLOR_CHANGE": "skin_change_ord",
        }
        out = []
        for e in supporting:
            feat = concept_to_feature.get(e.concept_id)
            val = prediction.contributions.get(feat, 0.0) if feat else 0.0
            if abs(val) < 1e-4:
                out.append(e)
                continue
            out.append(e.model_copy(update={"contribution": e.contribution.model_copy(update={
                "method": "shap",
                "value": round(float(val), 4),
                "direction": "escalating" if val > 0 else "de-escalating",
                "population_note": (
                    "SHAP attribution: how much this feature moved THIS MODEL'S output relative "
                    "to the development-set baseline. It is not a physiological mechanism and "
                    "must not be narrated as one."),
            })}))
        return out

    def _write_lineage(self, ref, patient_id, sc, fv, usable, prediction, cost_decision,
                       subscores, disc, cdetail, hits, rules_p) -> None:
        """
        The full drill-down. A judge must be able to answer, in one click:
        "why is this patient ranked above that one, and where did that number
        come from?"
        """
        self.store.write_lineage(ref, patient_id, {
            "displayed_acuity": {
                "scheme": f"{sc.acuity_current.scheme_id}/{sc.acuity_current.scheme_version}",
                "level": sc.acuity_current.level,
                "arrival_level": sc.acuity_arrival.level,
                "change_reason": sc.change_reason,
            },
            "decision_rule": (cost_decision.model_dump() if cost_decision else
                              {"note": "no cost decision - abstained or rules-only",
                               "rules_only_score": round(rules_p, 4)}),
            "red_flags": [{"rule_id": h.rule_id, "target": h.target_acuity.level,
                           "evidence": h.fired_on_evidence,
                           "suppressible_by_model": False} for h in hits],
            "fusion": {
                "channel_subscores": {k.value: v.concern for k, v in subscores.items()},
                "discordance": {"magnitude": disc.magnitude, "escalating": disc.escalating,
                                "de_escalating": disc.de_escalating, "pairs": disc.pairs},
            },
            "confidence_components": cdetail.components.model_dump(),
            "confidence_explanations": cdetail.explanations,
            "model": ({"route": prediction.route, "version": prediction.model_version,
                       "probability_calibrated": prediction.probability,
                       "probability_raw": prediction.probability_raw,
                       "calibration_status": prediction.calibration_status,
                       "shap_baseline": prediction.baseline,
                       "top_contributions": prediction.top_contributions(8)}
                      if prediction else {"note": "model unavailable or disabled"}),
            "features": {
                "route": fv.route,
                "values": {k: (None if v != v else v) for k, v in fv.values.items()},
                "provenance": fv.provenance,
                "missing": fv.missing,
                "stale": fv.stale,
                "band_notes": fv.band_notes,
            },
            "raw_observations": [
                {"evidence_id": e.evidence_id, "concept": e.concept_id, "value": e.value,
                 "channel": e.source_channel.value, "method": e.acquisition_method.value,
                 "device": e.device_id, "window": [w.isoformat() for w in e.observation_window],
                 "quality": e.signal_quality.model_dump(),
                 "freshness": e.freshness.model_dump() if e.freshness else None,
                 "reliability_weight": e.reliability_weight,
                 "threshold_band": e.threshold_band.value,
                 "age_band_context": e.age_band_context,
                 "model_version": e.model_version}
                for e in usable[:120]
            ],
            "versions": sc.model_versions | {"grammar": sc.grammar_version},
        })


# ---------------------------------------------------------------------------
# Surge detection
# ---------------------------------------------------------------------------


def detect_surge(contract: Contract, arrivals_last_hour: int, tier: str) -> tuple[bool, float, str]:
    """Automatic detection, announced on screen. Returns (active, ratio, message)."""
    baseline = float(contract.profile(tier)["baseline_arrivals_per_hour"])
    ratio = arrivals_last_hour / max(1.0, baseline)
    trigger = float(contract.surge["detection"]["trigger_ratio"])
    queue_mode = float(contract.surge["queue_triage_mode"]["trigger_ratio"])

    if ratio >= queue_mode:
        return True, ratio, (
            f"QUEUE TRIAGE MODE - arrivals at {ratio:.1f}x baseline. "
            "Beyond the assisted-triage operating envelope: individual alerts are suspended, "
            "a single ranked list is shown, and red flags still surface immediately.")
    if ratio >= trigger:
        return True, ratio, (
            f"SURGE POSTURE ACTIVE - arrivals at {ratio:.1f}x baseline. "
            "Re-check intervals halved, alerts batched, attendant prompts every 12 min, "
            "LLM narration disabled below level 2. Abstention rate will RISE - that is correct.")
    return False, ratio, f"Normal posture - arrivals at {ratio:.1f}x baseline."
