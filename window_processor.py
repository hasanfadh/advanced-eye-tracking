"""
Window Processor for Real-Time Eye Tracking Analysis
=====================================================
Implements sliding window feature extraction that integrates with
the existing EyeTracker pipeline without modifying existing code.

Architecture:
    EyeTracker.process_frame()
        → add_to_history(data)
        → WindowProcessor.update(data)   ← NEW
            → if window ready: extract features from all 3 domains
            → return latest_features or None

Window Parameters (from research proposal):
    - Window size : 3 seconds = 90 frames @ 30 FPS
    - Step size   : 1 second  = 30 frames @ 30 FPS
    - Min samples : 45 frames (50% of window) to allow early analysis
"""

from collections import deque
import numpy as np
import time

from signal_processing.time_domain import TimeDomainAnalyzer
from signal_processing.frequency_domain import FrequencyAnalyzer
from signal_processing.nonlinear import NonlinearAnalyzer


class WindowProcessor:
    """
    Real-time sliding window processor for eye tracking signals.

    Usage:
        processor = WindowProcessor()

        # Inside main loop, after add_to_history():
        features = processor.update(data)
        if features:
            print(features)  # New feature dict every ~1 second
    """

    def __init__(
        self,
        window_sec: float = 3.0,
        step_sec: float = 1.0,
        sampling_rate: int = 30,
        min_fill_ratio: float = 0.5,
    ):
        """
        Args:
            window_sec      : Window length in seconds (proposal: 3.0)
            step_sec        : Step/stride in seconds (proposal: 1.0)
            sampling_rate   : Camera FPS (default: 30)
            min_fill_ratio  : Minimum fraction of window needed to run analysis.
                              0.5 = start after 1.5 seconds of data.
        """
        self.sampling_rate = sampling_rate
        self.window_size = int(window_sec * sampling_rate)   # 90 frames
        self.step_size   = int(step_sec  * sampling_rate)    # 30 frames
        self.min_samples = int(self.window_size * min_fill_ratio)  # 45 frames

        # Circular buffer — maxlen automatically drops oldest frames
        self._buffer: deque = deque(maxlen=self.window_size)

        # Track how many frames since last analysis
        self._frames_since_last_analysis: int = 0

        # Latest computed features (None until first window is ready)
        self.latest_features: dict | None = None

        # Timestamp of last successful analysis
        self.last_analysis_time: float = 0.0

        # Analyzers (instantiate once, reuse every window)
        self._time_analyzer  = TimeDomainAnalyzer(degrees_per_unit=40)
        self._freq_analyzer  = FrequencyAnalyzer(sampling_rate=sampling_rate)
        self._nonlin_analyzer = NonlinearAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, data: dict) -> dict | None:
        """
        Feed one frame of tracking data into the processor.

        Call this every frame, right after EyeTracker.add_to_history(data).
        Returns a feature dict when a new window is ready, otherwise None.

        Args:
            data: The dict returned by EyeTracker.process_frame()
                  Must contain 'face_detected', 'gaze_h_ratio', 'gaze_v_ratio',
                  'left_ear', 'right_ear', 'blink_detected'.

        Returns:
            dict with all extracted features, or None if window not ready yet.
        """
        # Only buffer frames where face is detected
        if not data.get('face_detected'):
            return None

        self._buffer.append(data)
        self._frames_since_last_analysis += 1

        # Not enough data yet
        if len(self._buffer) < self.min_samples:
            return None

        # Check if step interval has elapsed
        if self._frames_since_last_analysis < self.step_size:
            return None

        # --- Time to run analysis ---
        self._frames_since_last_analysis = 0
        self.last_analysis_time = time.time()

        features = self._extract_features()
        self.latest_features = features
        return features

    def reset(self):
        """Clear buffer and reset state (call on 'r' keypress)."""
        self._buffer.clear()
        self._frames_since_last_analysis = 0
        self.latest_features = None

    @property
    def buffer_fill_ratio(self) -> float:
        """How full the buffer is (0.0 – 1.0). Useful for UI progress bar."""
        return len(self._buffer) / self.window_size

    @property
    def is_ready(self) -> bool:
        """True once minimum samples are in buffer."""
        return len(self._buffer) >= self.min_samples

    # ------------------------------------------------------------------
    # Internal: signal extraction helpers
    # ------------------------------------------------------------------

    def _get_signal_arrays(self) -> dict:
        """
        Convert buffer of dicts → named numpy arrays.
        Fills NaN for missing values so analyzers don't crash.
        """
        frames = list(self._buffer)

        def _extract(key):
            vals = [f.get(key) for f in frames]
            arr  = np.array([v if v is not None else np.nan for v in vals], dtype=float)
            return arr

        gaze_h   = _extract('gaze_h_ratio')
        gaze_v   = _extract('gaze_v_ratio')
        left_ear = _extract('left_ear')
        right_ear= _extract('right_ear')
        avg_ear  = (left_ear + right_ear) / 2.0

        blinks   = np.array([1.0 if f.get('blink_detected') else 0.0 for f in frames])

        return {
            'gaze_h':   gaze_h,
            'gaze_v':   gaze_v,
            'avg_ear':  avg_ear,
            'blinks':   blinks,
        }

    # ------------------------------------------------------------------
    # Internal: main feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self) -> dict:
        """
        Run all three domain analyzers on the current window.

        Returns a flat feature dict tagged with window metadata.
        """
        t_start = time.time()

        signals = self._get_signal_arrays()
        gaze_h  = signals['gaze_h']
        gaze_v  = signals['gaze_v']
        avg_ear = signals['avg_ear']
        blinks  = signals['blinks']

        features = {
            'window_size_frames': len(self._buffer),
            'window_size_sec':    len(self._buffer) / self.sampling_rate,
        }

        # === 1. TIME DOMAIN ===
        features['time_h'] = self._time_analyzer.extract_features(gaze_h)
        features['time_v'] = self._time_analyzer.extract_features(gaze_v)

        # Fixations and saccades (need both axes)
        valid_mask = ~np.isnan(gaze_h) & ~np.isnan(gaze_v)
        if valid_mask.sum() > 10:
            gh_clean = gaze_h[valid_mask]
            gv_clean = gaze_v[valid_mask]

            fixations = self._time_analyzer.detect_fixations(
                gh_clean, gv_clean, sampling_rate=self.sampling_rate
            )
            saccades  = self._time_analyzer.detect_saccades(
                gh_clean, gv_clean, sampling_rate=self.sampling_rate
            )

            features['fixations']  = fixations
            features['saccades']   = saccades

            # Derived fixation / saccade summary stats
            features['fixation_count']       = len(fixations)
            features['fixation_mean_dur_ms'] = (
                float(np.mean([f['duration_ms'] for f in fixations]))
                if fixations else None
            )
            features['saccade_count']        = len(saccades)
            features['saccade_mean_amp_deg'] = (
                float(np.mean([s['amplitude'] for s in saccades]))
                if saccades else None
            )

        # Blink rate (blinks per minute)
        window_sec = len(self._buffer) / self.sampling_rate
        blink_count_window = int(np.sum(blinks))
        features['blink_count_window'] = blink_count_window
        features['blink_rate_per_min'] = (
            (blink_count_window / window_sec) * 60.0 if window_sec > 0 else None
        )

        # EAR stats (proxy for pupil openness / drowsiness)
        ear_clean = avg_ear[~np.isnan(avg_ear)]
        if len(ear_clean) > 0:
            features['ear_mean'] = float(np.mean(ear_clean))
            features['ear_std']  = float(np.std(ear_clean))

        # === 2. FREQUENCY DOMAIN ===
        # Use horizontal gaze signal as primary (more variance)
        gh_valid = gaze_h[~np.isnan(gaze_h)]
        gv_valid = gaze_v[~np.isnan(gaze_v)]

        if len(gh_valid) >= 16:
            self._freq_analyzer.clear_cache()
            features['freq_h'] = self._freq_analyzer.extract_spectral_features(gh_valid)

        if len(gv_valid) >= 16:
            self._freq_analyzer.clear_cache()
            features['freq_v'] = self._freq_analyzer.extract_spectral_features(gv_valid)

        # === 3. NONLINEAR ===
        # Use horizontal gaze (same as frequency)
        if len(gh_valid) >= 50:
            features['nonlinear_h'] = self._nonlin_analyzer.extract_nonlinear_features(gh_valid)

        if len(gv_valid) >= 50:
            features['nonlinear_v'] = self._nonlin_analyzer.extract_nonlinear_features(gv_valid)

        # === Timing metadata ===
        features['extraction_latency_ms'] = (time.time() - t_start) * 1000.0

        return features

    # ------------------------------------------------------------------
    # Convenience: flatten to a single-level dict (for CSV logging)
    # ------------------------------------------------------------------

    @staticmethod
    def flatten(features: dict, sep: str = '_') -> dict:
        """
        Flatten nested feature dict to single-level dict.

        Example:
            {'freq_h': {'spectral_entropy': 0.8}} 
            → {'freq_h_spectral_entropy': 0.8}

        Useful for writing one row to CSV / pandas DataFrame.
        """
        out = {}

        def _recurse(d, prefix):
            if isinstance(d, dict):
                for k, v in d.items():
                    _recurse(v, f"{prefix}{sep}{k}" if prefix else k)
            elif isinstance(d, list):
                # For fixation/saccade lists, store count and mean only
                pass
            else:
                if d is not None:
                    try:
                        out[prefix] = float(d)
                    except (TypeError, ValueError):
                        pass  # skip non-numeric

        _recurse(features, '')
        return out