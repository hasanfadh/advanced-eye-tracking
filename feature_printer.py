"""
Feature Printer — Real-Time Terminal Display
=============================================
Menampilkan semua fitur hasil WindowProcessor ke terminal
dengan format yang mudah dibaca per window.

Cara pakai di main.py:
    from feature_printer import print_features

    features = processor.update(data)
    if features:
        print_features(features, window_num=window_counter)
        window_counter += 1
"""


def print_features(features: dict, window_num: int = 0):
    """
    Print semua fitur dari WindowProcessor ke terminal.

    Args:
        features   : dict dari WindowProcessor.update()
        window_num : nomor window (untuk tracking urutan)
    """
    latency  = features.get('extraction_latency_ms', 0)
    win_sec  = features.get('window_size_sec', 0)
    win_fr   = features.get('window_size_frames', 0)

    _header(f"WINDOW #{window_num}  |  {win_sec:.1f}s ({win_fr} frames)  |  latency={latency:.0f}ms")

    _section("TIME DOMAIN — Horizontal Gaze")
    _print_time(features.get('time_h'))

    _section("TIME DOMAIN — Vertical Gaze")
    _print_time(features.get('time_v'))

    _section("TIME DOMAIN — Fixations & Saccades")
    _print_fixsacc(features)

    _section("TIME DOMAIN — Blink & EAR")
    _print_blink_ear(features)

    _section("FREQUENCY DOMAIN — Horizontal Gaze")
    _print_freq(features.get('freq_h'))

    _section("FREQUENCY DOMAIN — Vertical Gaze")
    _print_freq(features.get('freq_v'))

    _section("NONLINEAR — Horizontal Gaze")
    _print_nonlinear(features.get('nonlinear_h'))

    _section("NONLINEAR — Vertical Gaze")
    _print_nonlinear(features.get('nonlinear_v'))

    print()


# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────

def _header(text: str):
    line = "=" * 64
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def _section(title: str):
    print(f"\n  ── {title}")


def _val(label: str, value, unit: str = "", width: int = 30):
    """Print one labeled value, right-aligned."""
    if value is None:
        formatted = "N/A"
    elif isinstance(value, float):
        formatted = f"{value:.4f}"
    else:
        formatted = str(value)

    suffix = f" {unit}" if unit else ""
    print(f"     {label:<{width}} {formatted}{suffix}")


# ─────────────────────────────────────────────
# Domain-specific printers
# ─────────────────────────────────────────────

def _print_time(t: dict | None):
    if not t:
        print("     [no data]")
        return

    _val("mean",              t.get('mean'),             "norm")
    _val("std",               t.get('std'),              "norm")
    _val("variance",          t.get('variance'))
    _val("median",            t.get('median'),           "norm")
    _val("min",               t.get('min'),              "norm")
    _val("max",               t.get('max'),              "norm")
    _val("range",             t.get('range'),            "norm")
    _val("skewness",          t.get('skewness'))
    _val("kurtosis",          t.get('kurtosis'))
    _val("rms",               t.get('rms'))
    _val("peak_amplitude",    t.get('peak_amplitude'))
    _val("q25",               t.get('q25'))
    _val("q75",               t.get('q75'))
    _val("iqr",               t.get('iqr'))
    _val("zero_crossings",    t.get('zero_crossings'),   "count")
    _val("num_peaks",         t.get('num_peaks'),        "count")
    _val("mean_velocity",     t.get('mean_velocity'),    "deg/frame")
    _val("max_velocity",      t.get('max_velocity'),     "deg/frame")
    _val("mean_acceleration", t.get('mean_acceleration'),"deg/frame²")
    _val("max_acceleration",  t.get('max_acceleration'), "deg/frame²")


def _print_fixsacc(features: dict):
    # Fixations
    fix_count    = features.get('fixation_count')
    fix_mean_dur = features.get('fixation_mean_dur_ms')
    _val("fixation_count",        fix_count,    "count")
    _val("fixation_mean_dur",     fix_mean_dur, "ms")

    # Saccades
    sacc_count    = features.get('saccade_count')
    sacc_mean_amp = features.get('saccade_mean_amp_deg')
    _val("saccade_count",         sacc_count,    "count")
    _val("saccade_mean_amplitude",sacc_mean_amp, "deg")

    # Per-fixation detail (jika ada)
    fixations = features.get('fixations', [])
    if fixations:
        print(f"\n     Fixation details ({len(fixations)} fixations):")
        for i, f in enumerate(fixations):
            print(
                f"       [{i+1}] dur={f['duration_ms']:.0f}ms  "
                f"pos=({f['position_h']:+.3f}, {f['position_v']:+.3f})"
            )

    # Per-saccade detail (jika ada)
    saccades = features.get('saccades', [])
    if saccades:
        print(f"\n     Saccade details ({len(saccades)} saccades):")
        for i, s in enumerate(saccades):
            print(
                f"       [{i+1}] dur={s['duration_ms']:.0f}ms  "
                f"amp={s['amplitude']:.2f}deg  "
                f"peak_vel={s['peak_velocity']:.1f}deg/s"
            )


def _print_blink_ear(features: dict):
    _val("blink_count (window)",  features.get('blink_count_window'), "count")
    _val("blink_rate",            features.get('blink_rate_per_min'), "blinks/min")
    _val("ear_mean",              features.get('ear_mean'))
    _val("ear_std",               features.get('ear_std'))


def _print_freq(f: dict | None):
    if not f:
        print("     [no data]")
        return

    _val("dominant_frequency",  f.get('dominant_frequency'),  "Hz")
    _val("spectral_entropy",    f.get('spectral_entropy'))
    _val("spectral_centroid",   f.get('spectral_centroid'),   "Hz")
    _val("spectral_rolloff",    f.get('spectral_rolloff'),    "Hz")
    _val("total_power",         f.get('total_power'))

    bands = f.get('frequency_bands')
    if bands:
        print(f"\n     Frequency bands:")
        band_labels = {
            'very_low': '0–0.5 Hz  (drift/pursuit)',
            'low':      '0.5–2 Hz  (normal saccades)',
            'medium':   '2–5 Hz    (fast/microsaccades)',
            'high':     '5+ Hz     (noise/tremor)',
        }
        for band_key, label in band_labels.items():
            b = bands.get(band_key, {})
            abs_p = b.get('absolute_power')
            rel_p = b.get('relative_power')
            if abs_p is not None and rel_p is not None:
                print(f"       {label:<35} abs={abs_p:.6f}  rel={rel_p:.2f}%")


def _print_nonlinear(n: dict | None):
    if not n:
        print("     [no data]")
        return

    _val("sample_entropy",      n.get('sample_entropy'))
    _val("approximate_entropy", n.get('approximate_entropy'))
    _val("fractal_dimension",   n.get('fractal_dimension'))
    _val("lyapunov_exponent",   n.get('lyapunov_exponent'))
    _val("dfa_exponent",        n.get('dfa_exponent'))

    rqa = n.get('rqa')
    if rqa:
        print(f"\n     RQA (Recurrence Quantification):")
        _val("  recurrence_rate",      rqa.get('recurrence_rate'),      width=28)
        _val("  determinism",          rqa.get('determinism'),          width=28)
        _val("  avg_diagonal_length",  rqa.get('avg_diagonal_length'),  width=28)
        _val("  max_diagonal_length",  rqa.get('max_diagonal_length'),  width=28)