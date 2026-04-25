"""
attention_display.py
=====================
Renders attention state overlay onto the OpenCV frame.
Place at: EYE-TRACKER/attention_display.py

Usage:
    from attention_display import AttentionDisplay
    display = AttentionDisplay()
    frame = display.render(frame, result, features, buffer_fill_ratio)
"""

import cv2
import numpy as np
import time
from collections import deque
from attention_classifier import AttentionResult, FOCUSED, PASSIVE, DISTRACTED, UNKNOWN, STATE_COLORS


class AttentionDisplay:
    """
    Draws a clean, research-grade HUD overlay on the OpenCV frame.

    Layout:
    ┌─────────────────────────────────────────────┐
    │  [STATE BADGE]    [CONFIDENCE BAR]           │  ← top-left panel
    │  Reasons list                                │
    ├─────────────────────────────────────────────┤
    │  Score history sparkline                     │  ← bottom strip
    │  Key metrics row                             │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, history_len: int = 60):
        self._score_history: deque = deque(maxlen=history_len)
        self._state_history: deque = deque(maxlen=history_len)
        self._last_state: str = UNKNOWN
        self._state_start: float = time.time()
        self._state_duration: float = 0.0

    def render(
        self,
        frame: np.ndarray,
        result: AttentionResult,
        features: dict,
        buffer_fill: float = 1.0,
    ) -> np.ndarray:
        """
        Draw all overlay elements onto frame (in-place + return).
        """
        self._score_history.append(result.score)
        self._state_history.append(result.state)

        # Track how long current state has been active
        if result.state != self._last_state:
            self._state_start = time.time()
            self._last_state = result.state
        self._state_duration = time.time() - self._state_start

        h, w = frame.shape[:2]

        # Draw elements
        self._draw_main_panel(frame, result, w)
        self._draw_metrics_row(frame, features, h, w)
        self._draw_sparkline(frame, h, w)
        self._draw_buffer_bar(frame, buffer_fill, h, w)
        self._draw_state_timer(frame, result, h)

        return frame

    # ── Main panel (top-left) ─────────────────────────────────────────────────

    def _draw_main_panel(self, frame, result: AttentionResult, w: int):
        color = result.color_bgr
        state = result.state

        # Semi-transparent background — taller to fit domain breakdown
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (355, 175), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Colored left accent bar
        cv2.rectangle(frame, (10, 10), (16, 175), color, -1)

        # State label (large)
        label_map = {FOCUSED: "FOCUSED", PASSIVE: "PASSIVE", DISTRACTED: "DISTRACTED", UNKNOWN: "WAITING"}
        label = label_map.get(state, state)
        cv2.putText(frame, label,
                    (25, 44), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

        # Confidence bar
        bar_x, bar_y, bar_w, bar_h = 25, 52, 195, 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 55), -1)
        fill = int(bar_w * result.confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"{result.confidence:.0%}",
                    (bar_x + bar_w + 6, bar_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        # Fused score
        sc = result.score
        score_color = (80, 220, 80) if sc > 0 else (80, 80, 220) if sc < 0 else (150, 150, 150)
        cv2.putText(frame, f"Fused score: {sc:+.3f}",
                    (25, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.38, score_color, 1, cv2.LINE_AA)

        # ── Domain breakdown bars ────────────────────────────────────────────
        domain_colors = {
            "time":      (100, 200, 255),   # yellow
            "frequency": (255, 180,  80),   # cyan
            "nonlinear": (160, 255, 130),   # green
        }
        domain_labels = {"time": "TIME", "frequency": "FREQ", "nonlinear": "NONL"}

        cv2.putText(frame, "DOMAIN SCORES",
                    (25, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 110), 1, cv2.LINE_AA)

        bar_total_w = 160
        bar_h2 = 7
        for i, (dname, ds) in enumerate(result.domain_scores.items()):
            y = 100 + i * 22
            dc = domain_colors.get(dname, (150, 150, 150))

            # Label + weight
            cv2.putText(frame, f"{domain_labels[dname]} w={ds.weight:.0%}",
                        (25, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, dc, 1, cv2.LINE_AA)

            # Background bar (full width = ±1.0, center = 0)
            bx = 95
            center_x = bx + bar_total_w // 2
            cv2.rectangle(frame, (bx, y), (bx + bar_total_w, y + bar_h2), (40, 40, 45), -1)
            cv2.line(frame, (center_x, y - 1), (center_x, y + bar_h2 + 1), (70, 70, 75), 1)

            # Filled portion
            norm = ds.score  # already in [-1, +1]
            half = bar_total_w // 2
            if norm >= 0:
                x0 = center_x
                x1 = center_x + int(norm * half)
            else:
                x0 = center_x + int(norm * half)
                x1 = center_x
            x0, x1 = max(bx, min(x0, x1)), min(bx + bar_total_w, max(x0, x1))
            if x1 > x0:
                cv2.rectangle(frame, (x0, y), (x1, y + bar_h2), dc, -1)

            # Score text
            cv2.putText(frame, f"{ds.score:+.2f}",
                        (bx + bar_total_w + 5, y + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, dc, 1, cv2.LINE_AA)

            # Domain state indicator — ASCII only (OpenCV Hershey no Unicode)
            ds_sym = "F" if ds.score > 0.25 else "D" if ds.score < -0.25 else "P"
            cv2.putText(frame, ds_sym,
                        (bx + bar_total_w + 30, y + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, dc, 1, cv2.LINE_AA)

        # Top reason (1 line only, space is tight)
        if result.reasons:
            reason = result.reasons[0]
            if len(reason) > 40:
                reason = reason[:38] + ".."
            txt_color = (100, 220, 100) if "✓" in reason else (100, 100, 220) if "✗" in reason else (160, 160, 160)
            cv2.putText(frame, reason,
                        (25, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.34, txt_color, 1, cv2.LINE_AA)

    # ── Metrics row (bottom) ──────────────────────────────────────────────────

    def _draw_metrics_row(self, frame, features: dict, h: int, w: int):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 52), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        metrics = [
            ("BLINK",    features.get("blink_rate_per_min"),   "/min",  0,   30),
            ("FIX DUR",  features.get("fixation_mean_dur_ms"), "ms",    0,   500),
            ("SACC AMP", features.get("saccade_mean_amp_deg"), "",     0,   15),
            ("EAR",      features.get("ear_mean"),             "",      0.1, 0.5),
            ("ENTROPY",  features.get("nonlinear_h_sample_entropy"), "", 0,  2.0),
        ]

        n = len(metrics)
        col_w = w // n

        for i, (label, val, unit, vmin, vmax) in enumerate(metrics):
            x = i * col_w + col_w // 2
            base_y = h - 48

            # Label
            cv2.putText(frame, label,
                        (x - 28, base_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 130), 1, cv2.LINE_AA)

            if val is None:
                cv2.putText(frame, "--",
                            (x - 8, base_y + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 85), 1, cv2.LINE_AA)
                continue

            # Value text — cast to float first to handle numpy.float64
            try:
                txt = f"{float(val):.1f}{unit}"
                val = float(val)
            except (TypeError, ValueError):
                txt = "--"

            # Color: gradient based on normalized value
            norm = max(0.0, min(1.0, (val - vmin) / (vmax - vmin + 1e-9)))
            r = int(norm * 200)
            g = int((1 - norm) * 200)
            val_color = (0, g + 55, r + 55)  # BGR

            cv2.putText(frame, txt,
                        (x - len(txt) * 4, base_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, val_color, 1, cv2.LINE_AA)

            # Mini bar indicator
            bar_len = int(norm * (col_w - 20))
            bar_x0 = i * col_w + 10
            bar_x1 = bar_x0 + bar_len
            cv2.rectangle(frame, (bar_x0, h - 8), (i * col_w + col_w - 10, h - 4), (40, 40, 45), -1)
            cv2.rectangle(frame, (bar_x0, h - 8), (bar_x1, h - 4), val_color, -1)

    # ── Score sparkline ───────────────────────────────────────────────────────

    def _draw_sparkline(self, frame, h: int, w: int):
        if len(self._score_history) < 2:
            return

        spark_h = 40
        spark_y_base = h - 52   # just above metrics row
        spark_x0 = w - 210
        spark_x1 = w - 10

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (spark_x0 - 5, spark_y_base - spark_h - 5),
                      (spark_x1 + 5, spark_y_base + 5), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Zero line
        mid_y = spark_y_base - spark_h // 2
        cv2.line(frame, (spark_x0, mid_y), (spark_x1, mid_y), (50, 50, 55), 1)

        # Label
        cv2.putText(frame, "SCORE",
                    (spark_x0, spark_y_base - spark_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 110), 1, cv2.LINE_AA)

        # Sparkline
        hist = list(self._score_history)
        score_range = 12.0  # ±12 covers full range
        pts = []
        for i, s in enumerate(hist):
            x = spark_x0 + int(i / (len(hist) - 1) * (spark_x1 - spark_x0))
            norm = (s / score_range) * 0.5  # -0.5 to +0.5
            y = int(mid_y - norm * spark_h)
            y = max(spark_y_base - spark_h, min(spark_y_base, y))
            pts.append((x, y))

        for i in range(1, len(pts)):
            s = hist[i]
            if s > 2.5:
                color = (0, 200, 80)
            elif s < -2.5:
                color = (0, 60, 220)
            else:
                color = (0, 180, 220)
            cv2.line(frame, pts[i-1], pts[i], color, 2, cv2.LINE_AA)

        # Current score dot
        if pts:
            last_s = hist[-1]
            dot_color = (0, 220, 80) if last_s > 2.5 else (0, 60, 220) if last_s < -2.5 else (0, 200, 220)
            cv2.circle(frame, pts[-1], 4, dot_color, -1, cv2.LINE_AA)

    # ── Buffer fill bar ───────────────────────────────────────────────────────

    def _draw_buffer_bar(self, frame, fill: float, h: int, w: int):
        """Small bar showing how full the sliding window buffer is."""
        bx0, bx1 = 10, 110
        by = h - 58

        cv2.rectangle(frame, (bx0, by - 3), (bx1, by + 3), (40, 40, 45), -1)
        filled = int((bx1 - bx0) * min(1.0, fill))
        buf_color = (0, 200, 100) if fill >= 1.0 else (0, 140, 220)
        cv2.rectangle(frame, (bx0, by - 3), (bx0 + filled, by + 3), buf_color, -1)

        label = "BUFFER FULL" if fill >= 1.0 else f"BUFFERING {fill:.0%}"
        cv2.putText(frame, label,
                    (bx0, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 110), 1, cv2.LINE_AA)

    # ── State duration timer ──────────────────────────────────────────────────

    def _draw_state_timer(self, frame, result: AttentionResult, h: int):
        color = result.color_bgr
        dur = self._state_duration
        mins = int(dur // 60)
        secs = int(dur % 60)
        timer_str = f"{mins:02d}:{secs:02d}"

        cv2.putText(frame, timer_str,
                    (280, 44),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "in state",
                    (280, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 110), 1, cv2.LINE_AA)


# ── Standalone preview (no camera) ───────────────────────────────────────────

if __name__ == "__main__":
    from attention_classifier import AttentionClassifier, AttentionResult

    classifier = AttentionClassifier()
    display = AttentionDisplay()

    # Simulate features
    demo_features = {
        "blink_rate_per_min":     12.0,
        "fixation_mean_dur_ms":   280.0,
        "fixation_count":         5.0,
        "saccade_mean_amp_deg":   2.8,
        "saccade_count":          9.0,
        "nonlinear_h_sample_entropy": 0.7,
        "freq_h_dominant_frequency":  0.67,
        "ear_mean":               0.31,
    }

    result = classifier.classify(demo_features)

    # Create a blank frame to preview overlay
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (25, 25, 30)

    # Add some fake face landmarks to make it look realistic
    cv2.ellipse(frame, (320, 200), (120, 150), 0, 0, 360, (50, 50, 55), 2)

    # Run display 60x to fill sparkline history
    for i in range(60):
        import math
        feat = demo_features.copy()
        feat["blink_rate_per_min"] = 12 + 3 * math.sin(i * 0.2)
        r = classifier.classify(feat)
        frame_copy = frame.copy()
        display.render(frame_copy, r, feat, buffer_fill=min(1.0, i / 54))

    display.render(frame, result, demo_features, buffer_fill=1.0)

    cv2.imshow("Attention Display Preview", frame)
    print("Preview window open — press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()