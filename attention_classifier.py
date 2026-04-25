"""
attention_classifier.py  (v2 — Multi-Domain Weighted Voting)
=============================================================

Architecture:
    ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
    │  TIME DOMAIN    │  │ FREQUENCY DOMAIN │  │   NONLINEAR     │
    │  scorer         │  │  scorer          │  │   scorer        │
    │  w=0.40         │  │  w=0.30          │  │   w=0.30        │
    └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘
             └──────────────── FUSION ───────────────────┘
                                  │
                        weighted_score → State
                        + per-domain breakdown

States:  FOCUSED | PASSIVE | DISTRACTED | UNKNOWN
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Tuple, List, Dict, Optional
import time


# ── Constants ─────────────────────────────────────────────────────────────────

FOCUSED    = "FOCUSED"
PASSIVE    = "PASSIVE"
DISTRACTED = "DISTRACTED"
UNKNOWN    = "UNKNOWN"

STATE_COLORS = {
    FOCUSED:    (0, 220, 100),
    PASSIVE:    (0, 180, 255),
    DISTRACTED: (0, 60,  255),
    UNKNOWN:    (120, 120, 120),
}

STATE_EMOJIS = {
    FOCUSED:    "◉",
    PASSIVE:    "◎",
    DISTRACTED: "◯",
    UNKNOWN:    "?",
}

DOMAIN_WEIGHT = {
    "time":      0.40,
    "frequency": 0.30,
    "nonlinear": 0.30,
}


# ── Thresholds ────────────────────────────────────────────────────────────────

@dataclass
class Thresholds:
    """
    Literature-based starting points. Tune after pilot testing.
    References:
    - Rayner (1998): fixation 200-250ms reading, 150ms scanning
    - Itti & Koch (2001): saccade < 5deg during focus
    - Klimesch (1999): alpha power during relaxed attention
    - Richman & Moorman (2000): SampEn < 1.0 in focused states
    """
    # Time domain
    blink_rate_focused_max:       float = 15.0
    blink_rate_distracted_min:    float = 25.0
    fixation_dur_focused_min:     float = 250.0
    fixation_dur_distracted_max:  float = 150.0
    fixation_count_focused_max:   int   = 5
    fixation_count_distracted_min:int   = 8
    saccade_amp_focused_max:      float = 3.0
    saccade_amp_distracted_min:   float = 6.0
    saccade_count_focused_max:    int   = 10
    saccade_count_distracted_min: int   = 20
    ear_closed_threshold:         float = 0.20
    ear_open_threshold:           float = 0.30

    # Frequency domain
    very_low_focused_min:              float = 25.0   # % relative power
    very_low_distracted_max:           float = 10.0
    low_distracted_min:                float = 40.0
    low_focused_max:                   float = 25.0
    high_distracted_min:               float = 15.0
    spectral_entropy_focused_max:      float = 0.55
    spectral_entropy_distracted_min:   float = 0.75
    spectral_centroid_focused_max:     float = 1.5
    spectral_centroid_distracted_min:  float = 3.0
    dominant_freq_distracted_min:      float = 2.0

    # Nonlinear domain
    sample_entropy_focused_max:    float = 0.8
    sample_entropy_distracted_min: float = 1.4
    approx_entropy_focused_max:    float = 0.5
    approx_entropy_distracted_min: float = 0.9
    fractal_dim_focused_max:       float = 1.4
    fractal_dim_distracted_min:    float = 1.7
    dfa_focused_min:               float = 1.4
    dfa_distracted_max:            float = 1.0
    lyapunov_focused_max:          float = 0.5
    lyapunov_distracted_min:       float = 1.0
    rqa_rr_focused_min:            float = 0.20
    rqa_rr_distracted_max:         float = 0.08
    rqa_det_focused_min:           float = 0.70
    rqa_det_distracted_max:        float = 0.40
    rqa_adl_focused_min:           float = 3.0
    rqa_adl_distracted_max:        float = 2.0


# ── Domain score result ───────────────────────────────────────────────────────

@dataclass
class DomainScore:
    name:      str
    score:     float        # normalized -1.0 to +1.0
    weight:    float
    reasons:   List[str]
    n_signals: int

    @property
    def weighted(self) -> float:
        return self.score * self.weight

    @property
    def state(self) -> str:
        if self.score > 0.25:   return FOCUSED
        if self.score < -0.25:  return DISTRACTED
        return PASSIVE


@dataclass
class AttentionResult:
    state:         str
    confidence:    float
    score:         float
    domain_scores: Dict[str, DomainScore]
    reasons:       List[str]
    timestamp:     float = field(default_factory=time.time)

    @property
    def color_bgr(self):
        return STATE_COLORS.get(self.state, STATE_COLORS[UNKNOWN])

    @property
    def emoji(self):
        return STATE_EMOJIS.get(self.state, "?")

    def domain_summary(self) -> str:
        parts = []
        for name, ds in self.domain_scores.items():
            sym = "F" if ds.score > 0.25 else "D" if ds.score < -0.25 else "P"
            parts.append(f"{name[:4].upper()}={ds.score:+.2f}{sym}")
        return "  ".join(parts)


# ── Per-domain scorers ────────────────────────────────────────────────────────

class TimeDomainScorer:
    def __init__(self, t: Thresholds):
        self.t = t

    def score(self, f: dict) -> Tuple[float, List[str], int]:
        raw, reasons, n = 0.0, [], 0
        t = self.t
        g = f.get

        blink = g("blink_rate_per_min")
        if blink is not None:
            n += 1
            if blink <= t.blink_rate_focused_max:
                raw += 2.0; reasons.append(f"Blink {blink:.0f}/min +")
            elif blink >= t.blink_rate_distracted_min:
                raw -= 2.0; reasons.append(f"Blink {blink:.0f}/min -")
            else:
                reasons.append(f"Blink {blink:.0f}/min (normal)")

        fix_dur = g("fixation_mean_dur_ms")
        if fix_dur is not None and fix_dur > 0:
            n += 1
            if fix_dur >= t.fixation_dur_focused_min:
                raw += 2.5; reasons.append(f"Fix dur {fix_dur:.0f}ms +")
            elif fix_dur <= t.fixation_dur_distracted_max:
                raw -= 2.5; reasons.append(f"Fix dur {fix_dur:.0f}ms -")

        fix_cnt = g("fixation_count")
        if fix_cnt is not None:
            n += 1
            if fix_cnt <= t.fixation_count_focused_max:
                raw += 1.0
            elif fix_cnt >= t.fixation_count_distracted_min:
                raw -= 1.0; reasons.append(f"Fix cnt {fix_cnt:.0f} -")

        sacc_amp = g("saccade_mean_amp_deg")
        if sacc_amp is not None and sacc_amp > 0:
            n += 1
            if sacc_amp <= t.saccade_amp_focused_max:
                raw += 1.5; reasons.append(f"Sacc {sacc_amp:.1f}deg +")
            elif sacc_amp >= t.saccade_amp_distracted_min:
                raw -= 1.5; reasons.append(f"Sacc {sacc_amp:.1f}deg -")

        sacc_cnt = g("saccade_count")
        if sacc_cnt is not None:
            n += 1
            if sacc_cnt <= t.saccade_count_focused_max:
                raw += 1.0
            elif sacc_cnt >= t.saccade_count_distracted_min:
                raw -= 1.0

        ear = g("ear_mean")
        if ear is not None:
            n += 1
            if ear < t.ear_closed_threshold:
                raw -= 1.5; reasons.append(f"EAR {ear:.2f} - (droopy)")
            elif ear > t.ear_open_threshold:
                raw += 0.5

        return max(-1.0, min(1.0, raw / 9.5)), reasons, n


class FrequencyDomainScorer:
    def __init__(self, t: Thresholds):
        self.t = t

    def score(self, f: dict) -> Tuple[float, List[str], int]:
        rh, reh, nh = self._channel(f, "h")
        rv, rev, nv = self._channel(f, "v")
        if nh > 0 and nv > 0:
            raw = (rh + rv) / 2; n = nh + nv
        elif nh > 0:
            raw, n = rh, nh
        else:
            raw, n = rv, nv
        return max(-1.0, min(1.0, raw / 7.0)), reh[:3], n

    def _channel(self, f, ch):
        raw, reasons, n = 0.0, [], 0
        t = self.t
        p = f"freq_{ch}_"
        g = lambda k: f.get(p + k)

        vl = g("frequency_bands_very_low_relative_power")
        if vl is not None:
            n += 1
            if vl >= t.very_low_focused_min:
                raw += 2.0; reasons.append(f"VLow {vl:.0f}% +")
            elif vl <= t.very_low_distracted_max:
                raw -= 1.5; reasons.append(f"VLow {vl:.0f}% -")

        low = g("frequency_bands_low_relative_power")
        if low is not None:
            n += 1
            if low >= t.low_distracted_min:
                raw -= 2.0; reasons.append(f"Low band {low:.0f}% -")
            elif low <= t.low_focused_max:
                raw += 1.5

        high = g("frequency_bands_high_relative_power")
        if high is not None:
            n += 1
            if high >= t.high_distracted_min:
                raw -= 1.5; reasons.append(f"High band {high:.0f}% -")

        se = g("spectral_entropy")
        if se is not None:
            n += 1
            if se <= t.spectral_entropy_focused_max:
                raw += 1.5; reasons.append(f"SpEnt {se:.2f} +")
            elif se >= t.spectral_entropy_distracted_min:
                raw -= 1.5; reasons.append(f"SpEnt {se:.2f} -")

        sc = g("spectral_centroid")
        if sc is not None:
            n += 1
            if sc <= t.spectral_centroid_focused_max:
                raw += 1.0
            elif sc >= t.spectral_centroid_distracted_min:
                raw -= 1.0; reasons.append(f"Centroid {sc:.1f}Hz -")

        df = g("dominant_frequency")
        if df is not None:
            n += 1
            if df >= t.dominant_freq_distracted_min:
                raw -= 1.0

        return raw, reasons, n


class NonlinearDomainScorer:
    def __init__(self, t: Thresholds):
        self.t = t

    def score(self, f: dict) -> Tuple[float, List[str], int]:
        raw, reasons, n = 0.0, [], 0
        t = self.t

        def g(k):
            vh = f.get(f"nonlinear_h_{k}")
            vv = f.get(f"nonlinear_v_{k}")
            if vh is not None and vv is not None:
                return (vh + vv) / 2
            return vh if vh is not None else vv

        se = g("sample_entropy")
        if se is not None:
            n += 1
            if se <= t.sample_entropy_focused_max:
                raw += 2.0; reasons.append(f"SampEn {se:.2f} +")
            elif se >= t.sample_entropy_distracted_min:
                raw -= 2.0; reasons.append(f"SampEn {se:.2f} -")

        ae = g("approximate_entropy")
        if ae is not None:
            n += 1
            if ae <= t.approx_entropy_focused_max:
                raw += 1.5
            elif ae >= t.approx_entropy_distracted_min:
                raw -= 1.5; reasons.append(f"ApproxEn {ae:.2f} -")

        fd = g("fractal_dimension")
        if fd is not None:
            n += 1
            if fd <= t.fractal_dim_focused_max:
                raw += 1.5; reasons.append(f"FracDim {fd:.2f} +")
            elif fd >= t.fractal_dim_distracted_min:
                raw -= 1.5; reasons.append(f"FracDim {fd:.2f} -")

        dfa = g("dfa_exponent")
        if dfa is not None:
            n += 1
            if dfa >= t.dfa_focused_min:
                raw += 1.0; reasons.append(f"DFA {dfa:.2f} +")
            elif dfa <= t.dfa_distracted_max:
                raw -= 1.0

        lya = g("lyapunov_exponent")
        if lya is not None:
            n += 1
            if lya <= t.lyapunov_focused_max:
                raw += 1.0
            elif lya >= t.lyapunov_distracted_min:
                raw -= 1.0; reasons.append(f"Lyapunov {lya:.2f} -")

        rr = g("rqa_recurrence_rate")
        if rr is not None:
            n += 1
            if rr >= t.rqa_rr_focused_min:
                raw += 1.5; reasons.append(f"RQA-RR {rr:.2f} +")
            elif rr <= t.rqa_rr_distracted_max:
                raw -= 1.5; reasons.append(f"RQA-RR {rr:.2f} -")

        det = g("rqa_determinism")
        if det is not None:
            n += 1
            if det >= t.rqa_det_focused_min:
                raw += 2.0; reasons.append(f"RQA-Det {det:.2f} +")
            elif det <= t.rqa_det_distracted_max:
                raw -= 2.0; reasons.append(f"RQA-Det {det:.2f} -")

        adl = g("rqa_avg_diagonal_length")
        if adl is not None:
            n += 1
            if adl >= t.rqa_adl_focused_min:
                raw += 1.0
            elif adl <= t.rqa_adl_distracted_max:
                raw -= 1.0

        return max(-1.0, min(1.0, raw / 12.5)), reasons, n


# ── Main classifier ───────────────────────────────────────────────────────────

class AttentionClassifier:
    """
    Multi-domain weighted voting classifier.
    score = w_time*score_time + w_freq*score_freq + w_nonlin*score_nonlin
    Each domain score normalized to [-1, +1].
    Final score smoothed over last N windows.
    """

    def __init__(
        self,
        thresholds:        Optional[Thresholds] = None,
        smoothing_windows: int = 3,
        weights:           Optional[Dict[str, float]] = None,
    ):
        self.t = thresholds or Thresholds()
        self._time_scorer   = TimeDomainScorer(self.t)
        self._freq_scorer   = FrequencyDomainScorer(self.t)
        self._nonlin_scorer = NonlinearDomainScorer(self.t)
        self._weights = weights or DOMAIN_WEIGHT.copy()
        self._history: deque = deque(maxlen=smoothing_windows)
        self.last_result = AttentionResult(UNKNOWN, 0.0, 0.0, {}, ["Initializing..."])

    def classify(self, features: dict) -> AttentionResult:
        t_score, t_reasons, t_n = self._time_scorer.score(features)
        f_score, f_reasons, f_n = self._freq_scorer.score(features)
        n_score, n_reasons, n_n = self._nonlin_scorer.score(features)

        domain_scores = {
            "time":      DomainScore("time",      t_score, self._weights["time"],      t_reasons, t_n),
            "frequency": DomainScore("frequency", f_score, self._weights["frequency"], f_reasons, f_n),
            "nonlinear": DomainScore("nonlinear", n_score, self._weights["nonlinear"], n_reasons, n_n),
        }

        fused = sum(ds.weighted for ds in domain_scores.values())
        self._history.append(fused)
        smooth = sum(self._history) / len(self._history)
        state, confidence = self._score_to_state(smooth)

        all_reasons = []
        for ds in domain_scores.values():
            all_reasons.extend(ds.reasons[:2])

        result = AttentionResult(
            state=state, confidence=confidence, score=round(smooth, 3),
            domain_scores=domain_scores, reasons=all_reasons[:6],
        )
        self.last_result = result
        return result

    def reset(self):
        self._history.clear()
        self.last_result = AttentionResult(UNKNOWN, 0.0, 0.0, {}, ["Reset."])

    def set_weights(self, time: float, frequency: float, nonlinear: float):
        total = time + frequency + nonlinear
        self._weights = {k: v/total for k, v in
                         zip(["time","frequency","nonlinear"], [time, frequency, nonlinear])}

    def _score_to_state(self, score: float) -> Tuple[str, float]:
        F, D = 0.25, -0.25
        if score >= F:
            conf = 0.5 + min(0.5, (score - F) / (1.0 - F) * 0.5)
            return FOCUSED, round(conf, 2)
        elif score <= D:
            conf = 0.5 + min(0.5, (D - score) / (1.0 + D) * 0.5)
            return DISTRACTED, round(conf, 2)
        else:
            conf = 1.0 - abs(score) / F * 0.35
            return PASSIVE, round(conf, 2)


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    classifier = AttentionClassifier()

    scenarios = {
        "FOCUSED": {
            "blink_rate_per_min": 10.0, "fixation_mean_dur_ms": 320.0,
            "fixation_count": 4.0, "saccade_mean_amp_deg": 2.1,
            "saccade_count": 8.0, "ear_mean": 0.32,
            "freq_h_frequency_bands_very_low_relative_power": 30.0,
            "freq_h_frequency_bands_low_relative_power": 18.0,
            "freq_h_frequency_bands_high_relative_power": 5.0,
            "freq_h_spectral_entropy": 0.45, "freq_h_spectral_centroid": 1.1,
            "freq_h_dominant_frequency": 0.5,
            "freq_v_frequency_bands_very_low_relative_power": 28.0,
            "freq_v_frequency_bands_low_relative_power": 20.0,
            "freq_v_frequency_bands_high_relative_power": 6.0,
            "freq_v_spectral_entropy": 0.48, "freq_v_spectral_centroid": 1.2,
            "freq_v_dominant_frequency": 0.6,
            "nonlinear_h_sample_entropy": 0.6, "nonlinear_h_approximate_entropy": 0.4,
            "nonlinear_h_fractal_dimension": 1.3, "nonlinear_h_lyapunov_exponent": 0.35,
            "nonlinear_h_dfa_exponent": 1.6,
            "nonlinear_h_rqa_recurrence_rate": 0.28, "nonlinear_h_rqa_determinism": 0.82,
            "nonlinear_h_rqa_avg_diagonal_length": 4.2,
            "nonlinear_v_sample_entropy": 0.65, "nonlinear_v_approximate_entropy": 0.42,
            "nonlinear_v_fractal_dimension": 1.35, "nonlinear_v_lyapunov_exponent": 0.4,
            "nonlinear_v_dfa_exponent": 1.55,
            "nonlinear_v_rqa_recurrence_rate": 0.25, "nonlinear_v_rqa_determinism": 0.78,
            "nonlinear_v_rqa_avg_diagonal_length": 3.8,
        },
        "DISTRACTED": {
            "blink_rate_per_min": 30.0, "fixation_mean_dur_ms": 110.0,
            "fixation_count": 10.0, "saccade_mean_amp_deg": 8.5,
            "saccade_count": 25.0, "ear_mean": 0.18,
            "freq_h_frequency_bands_very_low_relative_power": 6.0,
            "freq_h_frequency_bands_low_relative_power": 48.0,
            "freq_h_frequency_bands_high_relative_power": 20.0,
            "freq_h_spectral_entropy": 0.82, "freq_h_spectral_centroid": 3.5,
            "freq_h_dominant_frequency": 2.5,
            "freq_v_frequency_bands_very_low_relative_power": 5.0,
            "freq_v_frequency_bands_low_relative_power": 50.0,
            "freq_v_frequency_bands_high_relative_power": 22.0,
            "freq_v_spectral_entropy": 0.85, "freq_v_spectral_centroid": 3.8,
            "freq_v_dominant_frequency": 2.8,
            "nonlinear_h_sample_entropy": 1.6, "nonlinear_h_approximate_entropy": 1.1,
            "nonlinear_h_fractal_dimension": 1.75, "nonlinear_h_lyapunov_exponent": 1.2,
            "nonlinear_h_dfa_exponent": 0.8,
            "nonlinear_h_rqa_recurrence_rate": 0.05, "nonlinear_h_rqa_determinism": 0.30,
            "nonlinear_h_rqa_avg_diagonal_length": 1.8,
            "nonlinear_v_sample_entropy": 1.7, "nonlinear_v_approximate_entropy": 1.2,
            "nonlinear_v_fractal_dimension": 1.8, "nonlinear_v_lyapunov_exponent": 1.3,
            "nonlinear_v_dfa_exponent": 0.75,
            "nonlinear_v_rqa_recurrence_rate": 0.04, "nonlinear_v_rqa_determinism": 0.28,
            "nonlinear_v_rqa_avg_diagonal_length": 1.6,
        },
        "PASSIVE": {
            "blink_rate_per_min": 18.0, "fixation_mean_dur_ms": 190.0,
            "fixation_count": 6.0, "saccade_mean_amp_deg": 4.2,
            "saccade_count": 14.0, "ear_mean": 0.27,
            "freq_h_frequency_bands_very_low_relative_power": 16.0,
            "freq_h_frequency_bands_low_relative_power": 32.0,
            "freq_h_frequency_bands_high_relative_power": 10.0,
            "freq_h_spectral_entropy": 0.65, "freq_h_spectral_centroid": 2.0,
            "freq_h_dominant_frequency": 1.0,
            "freq_v_frequency_bands_very_low_relative_power": 15.0,
            "freq_v_frequency_bands_low_relative_power": 33.0,
            "freq_v_frequency_bands_high_relative_power": 11.0,
            "freq_v_spectral_entropy": 0.67, "freq_v_spectral_centroid": 2.1,
            "freq_v_dominant_frequency": 1.1,
            "nonlinear_h_sample_entropy": 1.0, "nonlinear_h_approximate_entropy": 0.68,
            "nonlinear_h_fractal_dimension": 1.55, "nonlinear_h_lyapunov_exponent": 0.7,
            "nonlinear_h_dfa_exponent": 1.25,
            "nonlinear_h_rqa_recurrence_rate": 0.14, "nonlinear_h_rqa_determinism": 0.55,
            "nonlinear_h_rqa_avg_diagonal_length": 2.5,
            "nonlinear_v_sample_entropy": 1.05, "nonlinear_v_approximate_entropy": 0.70,
            "nonlinear_v_fractal_dimension": 1.58, "nonlinear_v_lyapunov_exponent": 0.72,
            "nonlinear_v_dfa_exponent": 1.22,
            "nonlinear_v_rqa_recurrence_rate": 0.13, "nonlinear_v_rqa_determinism": 0.52,
            "nonlinear_v_rqa_avg_diagonal_length": 2.4,
        },
    }

    print("=" * 62)
    print("  Multi-Domain Weighted Voting Classifier — Test")
    print(f"  Weights: TIME={DOMAIN_WEIGHT['time']:.0%}  "
          f"FREQ={DOMAIN_WEIGHT['frequency']:.0%}  "
          f"NONLIN={DOMAIN_WEIGHT['nonlinear']:.0%}")
    print("=" * 62)

    all_correct = True
    for expected, feat in scenarios.items():
        classifier.reset()
        for _ in range(3):
            result = classifier.classify(feat)

        ds = result.domain_scores
        match = "+" if result.state == expected else f"- (got {result.state})"
        if result.state != expected:
            all_correct = False

        print(f"\nExpected: {expected}  {match}")
        print(f"  Final:  {result.emoji} {result.state}  "
              f"conf={result.confidence:.0%}  score={result.score:+.3f}")
        print(f"  Domain breakdown:")
        for name, d in ds.items():
            filled = int((d.score + 1) * 5)
            bar = "█" * filled + "░" * (10 - filled)
            print(f"    {name:10s} [{bar}] {d.score:+.2f} × w={d.weight:.0%}"
                  f" → {d.weighted:+.3f}  ({d.n_signals} signals)")
        print(f"  Top reasons:")
        for r in result.reasons[:4]:
            print(f"    {r}")

    print(f"\n{'All tests PASSED +' if all_correct else 'Some tests FAILED -'}")