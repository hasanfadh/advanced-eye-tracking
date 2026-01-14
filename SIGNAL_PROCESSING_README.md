# Signal Processing Module Documentation

## Overview

This module provides advanced signal processing capabilities for eye tracking data analysis, perfectly aligned with NKUST's research topics:
- Signal acquisition analysis
- Advanced nonlinear biomedical signal processing
- Python coding: data analysis, signal analysis

---

## Installation

```bash
# Install additional dependencies (if needed)
pip install PyWavelets
```

All other dependencies are already included in requirements.txt.

---

## Quick Start

### Basic Usage

```python
from signal_processing import (
    TimeDomainAnalyzer,
    FrequencyAnalyzer,
    NonlinearAnalyzer,
    SignalFilter,
    QualityAssessor
)

# Your gaze data
gaze_h = [0.1, 0.15, 0.12, ...]  # Horizontal gaze ratios
gaze_v = [0.05, 0.08, 0.06, ...]  # Vertical gaze ratios

# Time domain analysis
td = TimeDomainAnalyzer()
features = td.extract_features(gaze_h)
print(f"Mean: {features['mean']}, Std: {features['std']}")

# Frequency analysis
fa = FrequencyAnalyzer(sampling_rate=30)
dominant_freq = fa.get_dominant_frequency(gaze_h)
print(f"Dominant frequency: {dominant_freq} Hz")

# Nonlinear analysis
nl = NonlinearAnalyzer()
entropy = nl.sample_entropy(gaze_h)
fractal = nl.fractal_dimension(gaze_h)
print(f"Entropy: {entropy}, Fractal: {fractal}")
```

---

## Module Details

### 1. Time Domain Analysis (`time_domain.py`)

**Purpose:** Extract temporal characteristics from signals

**Key Features:**
- Basic statistics (mean, std, variance, etc.)
- Peak detection
- Velocity and acceleration calculation
- **Fixation detection** (I-VT algorithm)
- **Saccade detection**

**Example:**
```python
td = TimeDomainAnalyzer()

# Extract comprehensive features
features = td.extract_features(gaze_signal)

# Detect fixations (periods of stable gaze)
fixations = td.detect_fixations(gaze_h, gaze_v, 
                                velocity_threshold=30,
                                duration_threshold=100)

for fix in fixations:
    print(f"Fixation at ({fix['position_h']:.2f}, {fix['position_v']:.2f})")
    print(f"Duration: {fix['duration_ms']:.0f} ms")

# Detect saccades (fast eye movements)
saccades = td.detect_saccades(gaze_h, gaze_v)
```

**Research Applications:**
- Reading pattern analysis
- Visual search behavior
- Attention measurement
- Cognitive load assessment

---

### 2. Frequency Domain Analysis (`frequency_domain.py`)

**Purpose:** Analyze frequency content of eye movements

**Key Features:**
- Fast Fourier Transform (FFT)
- Power Spectral Density (PSD)
- Dominant frequency detection
- Frequency band analysis
- Spectral entropy
- Spectrogram generation

**Example:**
```python
fa = FrequencyAnalyzer(sampling_rate=30)

# FFT analysis
fft_result = fa.compute_fft(gaze_signal)
plt.plot(fft_result['frequencies'], fft_result['magnitudes'])

# Power spectral density
psd_result = fa.compute_psd(gaze_signal, method='welch')

# Frequency bands analysis
bands = fa.analyze_frequency_bands(gaze_signal)
for band_name, data in bands.items():
    print(f"{band_name}: {data['relative_power']:.1f}%")
```

**Frequency Bands (Eye Movements):**
- **Very Low (0-0.5 Hz):** Drift, slow pursuit
- **Low (0.5-2 Hz):** Normal saccades
- **Medium (2-5 Hz):** Fast saccades, microsaccades
- **High (5+ Hz):** Noise, tremor

**Research Applications:**
- Movement disorder detection
- Fatigue monitoring (frequency changes)
- Attention lapses (low-frequency drift)

---

### 3. Nonlinear Analysis (`nonlinear.py`) **MOST IMPORTANT**

**Purpose:** Advanced complexity and chaos analysis

**Key Features:**
- **Sample Entropy** - Signal regularity measure
- **Approximate Entropy** - Complexity quantification
- **Fractal Dimension** - Self-similarity analysis
- **Lyapunov Exponent** - Chaos detection
- **DFA (Detrended Fluctuation Analysis)** - Long-range correlations
- **Recurrence Quantification Analysis**

**Example:**
```python
nl = NonlinearAnalyzer()

# Sample Entropy (0 = very regular, higher = more irregular)
entropy = nl.sample_entropy(gaze_signal, m=2, r=0.2)
print(f"Sample Entropy: {entropy:.4f}")

# Interpretation:
if entropy < 0.5:
    print("→ Regular, predictable pattern (e.g., focused reading)")
elif entropy < 1.0:
    print("→ Moderate complexity (normal behavior)")
else:
    print("→ Irregular, unpredictable (e.g., distracted, searching)")

# Fractal Dimension (1 = smooth, 2 = very rough)
fractal = nl.fractal_dimension(gaze_signal)
print(f"Fractal Dimension: {fractal:.4f}")

# Lyapunov Exponent (positive = chaotic)
lyapunov = nl.lyapunov_exponent(gaze_signal)
if lyapunov > 0:
    print("→ Chaotic dynamics detected")
else:
    print("→ Stable, predictable behavior")

# DFA Exponent (measures correlations)
dfa = nl.detrended_fluctuation_analysis(gaze_signal)
if dfa < 0.5:
    print("→ Anti-correlated (mean-reverting)")
elif dfa > 0.5:
    print("→ Correlated (persistent)")
```

**Research Applications:**
- **Cognitive state detection** (entropy changes with mental state)
- **Neurological disorder diagnosis** (abnormal complexity)
- **Fatigue detection** (dynamics change when tired)
- **Attention assessment** (chaos increases when distracted)

**This is PERFECT for NKUST's "Advanced nonlinear biomedical signal processing"!**

---

### 4. Signal Filtering (`filtering.py`)

**Purpose:** Noise reduction and signal enhancement

**Key Features:**
- Moving average
- Exponential moving average
- Median filter (removes spikes)
- **Butterworth filters** (low-pass, high-pass, band-pass)
- **Kalman filter** (optimal estimation)
- Savitzky-Golay filter
- Outlier removal
- Wavelet denoising

**Example:**
```python
sf = SignalFilter()

# Remove high-frequency noise
smooth_signal = sf.butterworth_lowpass(noisy_signal, cutoff=5, fs=30)

# Kalman filter for optimal estimation
filtered = sf.kalman_filter_batch(noisy_signal, 
                                  process_variance=0.01,
                                  measurement_variance=0.1)

# Remove outliers
clean_signal, outliers = sf.outlier_removal(signal, threshold=3)

# Real-time Kalman filtering
sf.kalman_filter_init(initial_value=0.0)
for measurement in stream:
    filtered_value = sf.kalman_filter_step(measurement)
```

**When to Use:**
- **Low-pass:** Remove camera noise, smooth trajectories
- **High-pass:** Remove drift, baseline wander
- **Kalman:** Real-time optimal estimation
- **Median:** Remove spikes and outliers

---

### 5. Quality Assessment (`quality_metrics.py`)

**Purpose:** Evaluate signal reliability and tracking quality

**Key Features:**
- Signal-to-Noise Ratio (SNR)
- Root Mean Square Error (RMSE)
- Tracking stability index
- Missing data detection
- Sampling consistency check
- Overall confidence score

**Example:**
```python
qa = QualityAssessor()

# Calculate SNR
snr = qa.calculate_snr(signal)
print(f"SNR: {snr:.2f} dB")

# Assess tracking stability
stability = qa.assess_tracking_stability(gaze_h, gaze_v)
print(f"Stability: {stability:.2f} (0-1)")

# Comprehensive quality assessment
quality_report = qa.assess_signal_quality(data_history)
print(f"Quality rating: {quality_report['quality_rating']}")
print(f"Confidence: {quality_report['confidence_score']:.2f}")
```

**Quality Ratings:**
- **Excellent (0.8-1.0):** Reliable for research
- **Good (0.6-0.8):** Acceptable quality
- **Fair (0.4-0.6):** Use with caution
- **Poor (0-0.4):** Not reliable

---

## Research Use Cases

### 1. Reading Pattern Analysis
```python
# Detect fixations during reading
fixations = td.detect_fixations(gaze_h, gaze_v)

# Calculate reading metrics
fixation_durations = [f['duration_ms'] for f in fixations]
avg_fixation = np.mean(fixation_durations)
reading_speed = len(fixations) / (total_time_seconds / 60)  # fixations per minute

print(f"Avg fixation: {avg_fixation:.0f} ms")
print(f"Reading speed: {reading_speed:.1f} fixations/min")
```

### 2. Cognitive Load Assessment
```python
# High cognitive load → higher entropy, lower frequencies
entropy = nl.sample_entropy(gaze_signal)
dominant_freq = fa.get_dominant_frequency(gaze_signal)

if entropy > 1.0 and dominant_freq < 1.0:
    print("High cognitive load detected")
```

### 3. Fatigue Detection
```python
# Fatigue causes:
# - Increased blink rate
# - Lower dominant frequency
# - Higher entropy (less stable)

# Analyze in time windows
window_size = 30 * 60  # 1 minute windows at 30 FPS
for i in range(0, len(data), window_size):
    window = data[i:i+window_size]
    entropy = nl.sample_entropy(window)
    freq = fa.get_dominant_frequency(window)
    
    # Track changes over time
    fatigue_index = entropy / freq  # Higher = more fatigue
```

### 4. Attention State Classification
```python
from sklearn.ensemble import RandomForestClassifier

# Extract features
features = []
for segment in data_segments:
    f = {
        'entropy': nl.sample_entropy(segment),
        'fractal': nl.fractal_dimension(segment),
        'dominant_freq': fa.get_dominant_frequency(segment),
        'mean_velocity': td.extract_features(segment)['mean_velocity']
    }
    features.append(f)

# Train classifier (focused vs distracted)
clf = RandomForestClassifier()
clf.fit(features, labels)
```

---

## Interpretation Guide

### Sample Entropy
- **< 0.5:** Very regular, repetitive (e.g., steady fixation, automated task)
- **0.5-1.0:** Moderate complexity (normal active viewing)
- **> 1.0:** High irregularity (search, distraction, cognitive load)

### Fractal Dimension
- **~1.0:** Smooth, straight lines (smooth pursuit)
- **~1.5:** Natural eye movement complexity
- **~2.0:** Very rough, erratic (neurological issues)

### Dominant Frequency
- **< 0.5 Hz:** Slow drift, sustained attention
- **0.5-2 Hz:** Normal saccadic frequency
- **> 3 Hz:** Fast movements, microsaccades, tremor

### DFA Exponent (α)
- **α < 0.5:** Anti-correlated (mean reverting)
- **α ≈ 0.5:** Uncorrelated (random walk)
- **α > 0.5:** Correlated (persistent behavior)
- **α ≈ 1.0:** Pink noise (1/f noise)

---

## Example: Complete Analysis Pipeline

```python
from signal_processing import *

# Load data
data = load_eye_tracking_data()
gaze_h = [d['gaze_h_ratio'] for d in data]
gaze_v = [d['gaze_v_ratio'] for d in data]

# 1. Filter signal
sf = SignalFilter()
gaze_h_filtered = sf.kalman_filter_batch(gaze_h)

# 2. Time domain analysis
td = TimeDomainAnalyzer()
fixations = td.detect_fixations(gaze_h_filtered, gaze_v)
velocity_features = td.extract_features(gaze_h_filtered)

# 3. Frequency analysis
fa = FrequencyAnalyzer(sampling_rate=30)
spectral_features = fa.extract_spectral_features(gaze_h_filtered)

# 4. Nonlinear analysis
nl = NonlinearAnalyzer()
complexity_features = nl.extract_nonlinear_features(gaze_h_filtered)

# 5. Quality assessment
qa = QualityAssessor()
quality = qa.assess_signal_quality(data)

# Create comprehensive report
report = {
    'fixations': len(fixations),
    'avg_fixation_duration': np.mean([f['duration_ms'] for f in fixations]),
    'mean_velocity': velocity_features['mean_velocity'],
    'dominant_frequency': spectral_features['dominant_frequency'],
    'sample_entropy': complexity_features['sample_entropy'],
    'fractal_dimension': complexity_features['fractal_dimension'],
    'signal_quality': quality['quality_rating']
}

print(json.dumps(report, indent=2))
```

---

## Integration with Main App

The signal processing modules are automatically integrated when you run the main eye tracker:

```bash
python main.py
```

After recording, you'll be asked:
```
🔬 Generate signal processing analysis? (y/n): y
```

This will generate a comprehensive analysis report with:
- Time domain features
- Frequency spectrum analysis
- Nonlinear dynamics metrics
- Signal quality assessment

---

## Research Paper Template

### Suggested Paper Structure

**Title:** "Nonlinear Signal Processing Analysis of Eye Movement Patterns for Cognitive State Assessment"

**Abstract:**
- Eye tracking as biomedical signal
- Signal processing techniques applied
- Key findings

**Methods:**
- Data acquisition (webcam, 30 FPS)
- Signal processing pipeline
- Feature extraction (time, frequency, nonlinear)
- Statistical analysis

**Results:**
- Fixation/saccade characteristics
- Frequency spectrum analysis
- Entropy and complexity measures
- Comparison between conditions

**Discussion:**
- Interpretation of nonlinear features
- Comparison with existing methods
- Clinical/practical applications

**Code Availability:**
- GitHub repository
- Reproducible analysis

---

## Tips for NKUST Application

1. **Emphasize nonlinear analysis** - This matches their research focus perfectly
2. **Show understanding of signal processing theory** - Explain what each metric means
3. **Provide real examples** - Collect data and analyze it
4. **Create visualizations** - Spectrum plots, entropy over time, etc.
5. **Write clear documentation** - Shows research maturity

**Example statement for application:**
> "I have developed an eye tracking system with advanced signal processing capabilities, including frequency domain analysis (FFT, PSD), nonlinear dynamics analysis (sample entropy, fractal dimension, Lyapunov exponents), and real-time Kalman filtering. This work demonstrates my understanding of biomedical signal processing and readiness for research at NKUST."

---

## References

1. Richman & Moorman (2000). Sample Entropy for physiological time-series
2. Higuchi (1988). Fractal dimension calculation
3. Rosenstein et al. (1993). Lyapunov exponent estimation
4. Peng et al. (1994). Detrended Fluctuation Analysis
5. Salvucci & Goldberg (2000). I-VT algorithm for fixation detection

---

## Contributing

This module is designed for research purposes. Suggestions for improvements:
- Additional nonlinear metrics
- Machine learning integration
- Real-time optimization
- More advanced filtering techniques