"""
Turns a PatientProfile into the stream of Evidence objects the pipeline sees.

This is the SIMULATION BOUNDARY. Everything downstream of this module is real
code doing real work. Everything in this module is a stand-in for hardware we
have not built, and each stand-in is labelled:

  SIMULATED, REPLAYED TRACE : rPPG heart rate and respiratory rate, camera
                              occlusion and SNR traces, pose-derived stillness
                              and posture, skin-colour change
  SIMULATED, SCRIPTED       : ASR transcription and its confidence, attendant
                              prompt responses, prior-record lookups
  REAL                      : the observation windows, the reliability weights,
                              the quality floors, the staleness arithmetic and
                              the threshold bands - all read from the contract

We do not run a live camera, a live microphone or a trained rPPG model. Saying
so plainly is worth more than a demo that implies otherwise.
"""

from __future__ import annotations

import hashlib
import struct

from datetime import datetime, timedelta

from ..core.contract_loader import Contract, evaluate_band
from ..core.gating import evaluate_freshness, evaluate_quality
from ..core.models import (
    AcquisitionMethod,
    Channel,
    Evidence,
    QualityStatus,
    SignalQuality,
    next_id,
)
from .cohort import PatientProfile

CAMERA_WINDOW_SECONDS = 60.0
ARRIVAL_GESTALT_SECONDS = 8.0




def _stable_jitter(patient_id: str, minute: int) -> float:
    """
    Deterministic pseudo-random jitter in [-0.5, 0.5) for one patient-minute.

    Python's builtin hash() is randomised per process for str (PYTHONHASHSEED),
    so seeding simulated observation noise with it made every evaluation run
    produce different numbers - occasionally different enough to move an arrival
    acuity across a threshold. A generative process that cannot be reproduced
    cannot be audited, which defeats the point of stating it. blake2b is stable
    across processes, platforms and Python versions.
    """
    digest = hashlib.blake2b(f"{patient_id}|{minute}".encode(), digest_size=8).digest()
    return (struct.unpack("<Q", digest)[0] % 1000) / 1000.0 - 0.5

def _piecewise(trace: dict[float, float | str], t: float, default):
    """Value of a piecewise-constant trace at minute t."""
    val = default
    for k in sorted(trace):
        if t >= k:
            val = trace[k]
    return val


class Simulator:
    """Generates the evidence stream for one encounter up to `elapsed_minutes`."""

    def __init__(
        self,
        contract: Contract,
        now: datetime,
        tier: str = "TIER_A",
        *,
        camera_stride_minutes: float = 1.0,
        camera_lookback_minutes: float | None = None,
    ):
        """
        camera_stride_minutes / camera_lookback_minutes exist ONLY to make bulk
        training-set generation tractable. They coarsen how many rolling camera
        windows are materialised; they do not change any value, any quality
        figure or any gate. The demo and the API always run at stride 1.0 with
        no lookback limit, which is the full trace.
        """
        self.c = contract
        self.now = now
        self.tier = tier
        self.profile_spec = contract.profile(tier)
        self.camera_stride_minutes = camera_stride_minutes
        self.camera_lookback_minutes = camera_lookback_minutes

    # -- helpers -----------------------------------------------------------

    def _ts(self, arrival: datetime, t_min: float) -> datetime:
        return arrival + timedelta(minutes=t_min)

    def _mk(
        self,
        p: PatientProfile,
        concept_id: str,
        value,
        channel: Channel,
        method: AcquisitionMethod,
        end_t: float,
        arrival: datetime,
        window_s: float,
        *,
        device_id: str | None = None,
        quality: SignalQuality | None = None,
        grain: str | None = None,
        model_version: str | None = None,
    ) -> Evidence:
        end = self._ts(arrival, end_t)
        start = end - timedelta(seconds=window_s)
        band, note = evaluate_band(self.c, concept_id, value, self.c.age_band(p.age_years))
        return Evidence(
            evidence_id=next_id("EV", p.patient_id),
            patient_id=p.patient_id,
            source_channel=channel,
            acquisition_method=method,
            device_id=device_id,
            concept_id=concept_id,
            value=value,
            unit=self.c.unit(concept_id),
            observation_window=(start, end),
            grain=grain or self.c.concept(concept_id).get("grain", "patient x window"),
            signal_quality=quality or SignalQuality(status=QualityStatus.NOT_APPLICABLE),
            freshness=evaluate_freshness(self.c, concept_id, end, self.now),
            reliability_weight=self.c.reliability_weight(method),
            threshold_band=band,
            age_band_context=self._band_context(concept_id, p.age_years, note),
            access_classification=self.c.pii_class(concept_id) or "CLINICAL_DERIVED",
            model_version=model_version,
            contract_version=self.c.version,
        )

    def _band_context(self, concept_id: str, age: float, note: str | None) -> str:
        band_id = self.c.age_band(age)
        spec = self.c.thresholds_for(concept_id, band_id)
        if not spec:
            return f"{self.c.age_band_label(band_id)}: no declared thresholds"
        txt = f"{self.c.age_band_label(band_id)} thresholds: normal {spec.get('normal')}, critical {spec.get('critical')}"
        if note:
            txt += f" | {note}"
        return txt

    # -- channels ----------------------------------------------------------

    def _nurse(self, p: PatientProfile, arrival: datetime, elapsed: float) -> list[Evidence]:
        ev: list[Evidence] = []
        times = list(p.nurse_recheck_minutes)
        if p.nurse_vitals_offset_min > 0:
            times = [max(0.0, elapsed - p.nurse_vitals_offset_min)]
        cuff_available = "nibp_cuff" in (self.profile_spec["channels"]["nurse"].get("vitals_devices") or [])

        for t in times:
            if t > elapsed:
                continue
            v = p.vitals.at(t)
            ev.append(self._mk(p, "HEART_RATE", round(float(v["HEART_RATE"])), Channel.NURSE,
                               AcquisitionMethod.CUFF, t, arrival, 30, device_id="NURSE-MON-01"))
            ev.append(self._mk(p, "RESP_RATE", round(float(v["RESP_RATE"])), Channel.NURSE,
                               AcquisitionMethod.MANUAL_COUNT, t, arrival, 30))
            ev.append(self._mk(p, "SPO2", round(float(v["SPO2"])), Channel.NURSE,
                               AcquisitionMethod.CUFF, t, arrival, 30, device_id="NURSE-MON-01"))
            if cuff_available and v["SBP"] is not None:
                ev.append(self._mk(p, "SBP", round(float(v["SBP"])), Channel.NURSE,
                                   AcquisitionMethod.CUFF, t, arrival, 30, device_id="NURSE-MON-01"))
            ev.append(self._mk(p, "TEMP", round(float(v["TEMP"]), 1), Channel.NURSE,
                               AcquisitionMethod.CUFF, t, arrival, 30))
            ev.append(self._mk(p, "AVPU", v["AVPU"], Channel.NURSE,
                               AcquisitionMethod.CLINICAL_OBSERVATION, t, arrival, 10))
            ev.append(self._mk(p, "CAP_REFILL", float(v["CAP_REFILL"]), Channel.NURSE,
                               AcquisitionMethod.CLINICAL_OBSERVATION, t, arrival, 10))
            ev.append(self._mk(p, "PAIN_SELF_REPORT", int(v["PAIN_SELF_REPORT"]), Channel.NURSE,
                               AcquisitionMethod.SELF_REPORT, t, arrival, 10))

        # Spoken complaint, captured once at the desk.
        if self.profile_spec["channels"]["nurse"].get("asr") and p.complaint_text:
            q = evaluate_quality(self.c, AcquisitionMethod.ASR, asr_confidence=p.asr_confidence)
            ev.append(self._mk(p, "COMPLAINT_TEXT", p.complaint_text, Channel.NURSE,
                               AcquisitionMethod.ASR, 0.5, arrival, 25, quality=q,
                               model_version="asr_codemix_v1.3"))

        # Tier C has no camera, so the nurse taps arrival mode instead: the
        # gestalt signal survives even where the sensor does not.
        if not self.profile_spec["channels"]["camera"]["enabled"] and p.camera.arrival_mode:
            ev.append(self._mk(p, "ARRIVAL_MODE", p.camera.arrival_mode, Channel.NURSE,
                               AcquisitionMethod.CLINICAL_OBSERVATION, 0.2, arrival, 5))
        return ev

    def _camera(self, p: PatientProfile, arrival: datetime, elapsed: float) -> list[Evidence]:
        ev: list[Evidence] = []
        cam_cfg = self.profile_spec["channels"]["camera"]
        if not cam_cfg["enabled"] or not p.camera.enabled or not p.consent_given:
            return ev

        def q_at(t: float, method: AcquisitionMethod) -> SignalQuality:
            """
            Report only the metrics that actually bear on this acquisition method.

            SNR is meaningless for a pose estimator, and face-region occlusion
            does not directly describe an rPPG waveform's noise floor (it is
            already reflected in the SNR itself). Reporting every metric against
            every method would drag each item's status down to the worst number
            in the room and destroy the distinction the nurse needs: which
            specific sensor is struggling.
            """
            occl = float(_piecewise(p.camera.occlusion_override, t, 14.0))
            snr = float(_piecewise(p.camera.snr_override, t, 4.8))
            if method in (AcquisitionMethod.RPPG_HR, AcquisitionMethod.RPPG_RR):
                return evaluate_quality(self.c, method, snr_db=snr, motion_index=p.camera.motion_index)
            if method is AcquisitionMethod.POSE_ESTIMATION:
                return evaluate_quality(self.c, method, occlusion_pct=occl,
                                        motion_index=p.camera.motion_index)
            return evaluate_quality(self.c, method, occlusion_pct=occl)

        # Arrival gestalt: a single pass at the door.
        if cam_cfg.get("arrival_gestalt"):
            qg = q_at(0.0, AcquisitionMethod.GESTALT_ARRIVAL)
            ev.append(self._mk(p, "ARRIVAL_MODE", p.camera.arrival_mode, Channel.CAMERA,
                               AcquisitionMethod.GESTALT_ARRIVAL, 0.15, arrival, ARRIVAL_GESTALT_SECONDS,
                               device_id="DOOR-CAM-01", quality=qg, model_version="gestalt_v0.9_SIMULATED"))
            ev.append(self._mk(p, "WORK_OF_BREATHING", p.camera.work_of_breathing, Channel.CAMERA,
                               AcquisitionMethod.GESTALT_ARRIVAL, 0.15, arrival, ARRIVAL_GESTALT_SECONDS,
                               device_id="DOOR-CAM-01", quality=qg, model_version="gestalt_v0.9_SIMULATED"))

        if not cam_cfg.get("waiting_rppg"):
            return ev

        # Rolling waiting-area windows.
        stride = max(0.25, self.camera_stride_minutes)
        t = 1.0
        if self.camera_lookback_minutes is not None:
            first = max(1.0, elapsed - self.camera_lookback_minutes)
            t = first + ((elapsed - first) % stride)
        while t <= elapsed:
            if p.camera.reid_failure_at is not None and t >= p.camera.reid_failure_at:
                break  # face track lost - re-identification failure, channel goes SILENT
            v = p.vitals.at(t)
            qhr = q_at(t, AcquisitionMethod.RPPG_HR)
            qrr = q_at(t, AcquisitionMethod.RPPG_RR)

            # rPPG observation noise scales inversely with SNR.
            snr = qhr.snr_db or 4.0
            noise_hr = max(0.5, 9.0 / max(1.0, snr))
            noise_rr = max(1.0, 14.0 / max(1.0, snr))
            seed = _stable_jitter(p.patient_id, int(t))

            ev.append(self._mk(p, "HEART_RATE", round(float(v["HEART_RATE"]) + seed * 2 * noise_hr),
                               Channel.CAMERA, AcquisitionMethod.RPPG_HR, t, arrival, CAMERA_WINDOW_SECONDS,
                               device_id="WAIT-CAM-02", quality=qhr, model_version="rppg_v2.1_SIMULATED",
                               grain="patient x 60s_window"))
            ev.append(self._mk(p, "RESP_RATE", round(float(v["RESP_RATE"]) + seed * 2 * noise_rr),
                               Channel.CAMERA, AcquisitionMethod.RPPG_RR, t, arrival, CAMERA_WINDOW_SECONDS,
                               device_id="WAIT-CAM-02", quality=qrr, model_version="rppg_v2.1_SIMULATED",
                               grain="patient x 60s_window"))

            qp = q_at(t, AcquisitionMethod.POSE_ESTIMATION)
            still = 0.0
            if p.camera.stillness_from_min is not None and t >= p.camera.stillness_from_min:
                still = round(t - p.camera.stillness_from_min, 1)
            ev.append(self._mk(p, "STILLNESS_MINUTES", still, Channel.CAMERA,
                               AcquisitionMethod.POSE_ESTIMATION, t, arrival, CAMERA_WINDOW_SECONDS,
                               device_id="WAIT-CAM-02", quality=qp, model_version="pose_v1.4_SIMULATED"))
            ev.append(self._mk(p, "POSTURE", _piecewise(p.camera.posture, t, "upright"), Channel.CAMERA,
                               AcquisitionMethod.POSE_ESTIMATION, t, arrival, CAMERA_WINDOW_SECONDS,
                               device_id="WAIT-CAM-02", quality=qp, model_version="pose_v1.4_SIMULATED"))
            ev.append(self._mk(p, "WORK_OF_BREATHING", p.camera.work_of_breathing, Channel.CAMERA,
                               AcquisitionMethod.POSE_ESTIMATION, t, arrival, CAMERA_WINDOW_SECONDS,
                               device_id="WAIT-CAM-02", quality=qp, model_version="pose_v1.4_SIMULATED"))

            if int(t) % 2 == 0:
                ev.append(self._mk(p, "SKIN_COLOR_CHANGE",
                                   _piecewise(p.camera.skin_color_change, t, "unchanged"),
                                   Channel.CAMERA, AcquisitionMethod.RPPG_HR, t, arrival, 120,
                                   device_id="WAIT-CAM-02", quality=qhr,
                                   model_version="chroma_v0.7_SIMULATED"))
            t += stride
        return ev

    def _attendant(self, p: PatientProfile, arrival: datetime, elapsed: float) -> list[Evidence]:
        ev: list[Evidence] = []
        if not self.profile_spec["channels"]["attendant"]["enabled"]:
            return ev
        if not p.attendant.present or not p.consent_given:
            return ev
        a = p.attendant
        dev = f"ATTENDANT-{a.transport.upper()}"

        for t, val in a.responds_to_name.items():
            if t <= elapsed:
                ev.append(self._mk(p, "RESPONDS_TO_NAME", bool(val), Channel.ATTENDANT,
                                   AcquisitionMethod.PROXY_REPORT, t, arrival, 30, device_id=dev,
                                   grain="patient x prompt_response"))
        for t, val in a.confusion_new.items():
            if t <= elapsed:
                ev.append(self._mk(p, "CONFUSION_NEW", bool(val), Channel.ATTENDANT,
                                   AcquisitionMethod.PROXY_REPORT, t, arrival, 30, device_id=dev,
                                   grain="patient x prompt_response"))
        for t, val in a.sleepiness_increase.items():
            if t <= elapsed:
                ev.append(self._mk(p, "SLEEPINESS_INCREASE", bool(val), Channel.ATTENDANT,
                                   AcquisitionMethod.PROXY_REPORT, t, arrival, 30, device_id=dev,
                                   grain="patient x prompt_response"))
        for t, val in a.guided_rr.items():
            if t <= elapsed:
                ev.append(self._mk(p, "RESP_RATE", float(val), Channel.ATTENDANT,
                                   AcquisitionMethod.GUIDED_COUNT_15S, t, arrival, 15, device_id=dev,
                                   grain="patient x prompt_response"))
        for t in a.concern_presses:
            if t <= elapsed:
                ev.append(self._mk(p, "ATTENDANT_CONCERN", True, Channel.ATTENDANT,
                                   AcquisitionMethod.PROXY_REPORT, t, arrival, 5, device_id=dev,
                                   grain="patient x prompt_response"))
        for t, txt in a.free_text.items():
            if t <= elapsed:
                ev.append(self._mk(p, "ATTENDANT_FREE_TEXT", txt, Channel.ATTENDANT,
                                   AcquisitionMethod.PROXY_REPORT, t, arrival, 20, device_id=dev,
                                   grain="patient x prompt_response"))
        return ev

    def _prior(self, p: PatientProfile, arrival: datetime) -> list[Evidence]:
        if not self.profile_spec["channels"]["prior_record"]["enabled"]:
            return []
        if not p.prior.available:
            return []
        ev = []
        for concept, value in (
            ("COMORBIDITY_COUNT", p.prior.comorbidity_count),
            ("PRIOR_ED_VISITS_90D", p.prior.prior_ed_visits_90d),
            ("PRIOR_ICU_ADMISSION", p.prior.prior_icu_admission),
        ):
            e = self._mk(p, concept, value, Channel.PRIOR_RECORD, AcquisitionMethod.RECORD_LOOKUP,
                         0.3, arrival, 5, device_id="ABDM-HIECM", grain="patient x record_version")
            ev.append(e)
        return ev

    def _system(self, p: PatientProfile, arrival: datetime, elapsed: float) -> list[Evidence]:
        last_human = max([t for t in p.nurse_recheck_minutes if t <= elapsed] or [0.0])
        if p.nurse_vitals_offset_min > 0:
            last_human = max(0.0, elapsed - p.nurse_vitals_offset_min)
        return [
            self._mk(p, "WAIT_MINUTES", round(elapsed, 1), Channel.SYSTEM,
                     AcquisitionMethod.DETERMINISTIC, elapsed, arrival, 1),
            self._mk(p, "MINUTES_SINCE_HUMAN_CONTACT", round(elapsed - last_human, 1), Channel.SYSTEM,
                     AcquisitionMethod.DETERMINISTIC, elapsed, arrival, 1),
        ]

    def _derived(self, p: PatientProfile, evidence: list[Evidence], arrival: datetime, elapsed: float) -> list[Evidence]:
        """
        SHOCK_INDEX is computed ONLY when both inputs passed their quality and
        staleness gates. It is never computed from an imputed component.
        """
        hr = [e for e in evidence if e.concept_id == "HEART_RATE"
              and e.source_channel is Channel.NURSE and e.usable]
        sbp = [e for e in evidence if e.concept_id == "SBP" and e.usable]
        if not hr or not sbp:
            return []
        h, s = max(hr, key=lambda e: e.observed_at), max(sbp, key=lambda e: e.observed_at)
        if float(s.value) <= 0:
            return []
        t = (max(h.observed_at, s.observed_at) - arrival).total_seconds() / 60.0
        e = self._mk(p, "SHOCK_INDEX", round(float(h.value) / float(s.value), 2), Channel.NURSE,
                     AcquisitionMethod.DETERMINISTIC, t, arrival, 1)
        return [e.model_copy(update={"lineage_ref": f"derived_from:{h.evidence_id},{s.evidence_id}"})]

    # -- entry point -------------------------------------------------------

    def evidence_for(self, p: PatientProfile, elapsed_minutes: float | None = None) -> list[Evidence]:
        elapsed = p.arrival_minutes_ago if elapsed_minutes is None else elapsed_minutes
        arrival = self.now - timedelta(minutes=elapsed)
        ev: list[Evidence] = []
        ev += self._nurse(p, arrival, elapsed)
        ev += self._camera(p, arrival, elapsed)
        ev += self._attendant(p, arrival, elapsed)
        ev += self._prior(p, arrival)
        ev += self._system(p, arrival, elapsed)
        ev += self._derived(p, ev, arrival, elapsed)
        return sorted(ev, key=lambda e: e.observed_at)

    def arrival_time(self, p: PatientProfile, elapsed_minutes: float | None = None) -> datetime:
        elapsed = p.arrival_minutes_ago if elapsed_minutes is None else elapsed_minutes
        return self.now - timedelta(minutes=elapsed)
