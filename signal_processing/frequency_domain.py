import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

class FrequencyAnalyzer:
    """
    OPTIMIZED: Frequency domain analysis for eye tracking signals
    
    KEY OPTIMIZATION: Cache PSD/FFT results to avoid redundant calculations
    """
    
    def __init__(self, sampling_rate=30):
        """
        Args:
            sampling_rate: Sampling frequency in Hz (typical webcam: 30 FPS)
        """
        self.sampling_rate = sampling_rate
        # Cache for avoiding redundant calculations
        self._psd_cache = {}
        self._fft_cache = {}
    
    def _get_signal_hash(self, signal_data):
        """Create hash for signal caching"""
        signal_array = np.array(signal_data)
        # Simple hash using length and first/last values
        return hash((len(signal_array), signal_array[0], signal_array[-1], signal_array.sum()))
    
    def compute_fft(self, signal_data, use_cache=True):
        """
        Compute Fast Fourier Transform
        
        Args:
            signal_data: Time domain signal
            use_cache: Use cached result if available
            
        Returns:
            dict: Frequencies, magnitudes, and phases
        """
        if use_cache:
            sig_hash = self._get_signal_hash(signal_data)
            if sig_hash in self._fft_cache:
                return self._fft_cache[sig_hash]
        
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) == 0:
            return None
        
        # Remove DC component (mean)
        signal_array = signal_array - np.mean(signal_array)
        
        # Apply window to reduce spectral leakage
        window = np.hanning(len(signal_array))
        windowed_signal = signal_array * window
        
        # Compute FFT
        N = len(windowed_signal)
        fft_values = fft(windowed_signal)
        frequencies = fftfreq(N, 1/self.sampling_rate)
        
        # Take only positive frequencies
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        magnitudes = np.abs(fft_values[positive_freq_idx])
        phases = np.angle(fft_values[positive_freq_idx])
        
        # Normalize
        magnitudes = magnitudes / N
        
        result = {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'phases': phases,
            'sampling_rate': self.sampling_rate
        }
        
        if use_cache:
            self._fft_cache[sig_hash] = result
        
        return result
    
    def compute_psd(self, signal_data, method='welch', use_cache=True):
        """
        OPTIMIZED: Compute Power Spectral Density with caching
        
        Args:
            signal_data: Time domain signal
            method: 'welch' or 'periodogram'
            use_cache: Use cached result if available
            
        Returns:
            dict: Frequencies and power spectral density
        """
        if use_cache:
            sig_hash = self._get_signal_hash(signal_data)
            cache_key = (sig_hash, method)
            if cache_key in self._psd_cache:
                return self._psd_cache[cache_key]
        
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 4:
            return None
        
        if method == 'welch':
            # Welch's method: more robust, smoothed estimate
            nperseg = min(256, len(signal_array) // 4)
            frequencies, psd = signal.welch(
                signal_array,
                fs=self.sampling_rate,
                nperseg=nperseg,
                scaling='density'
            )
        else:
            # Periodogram: direct FFT-based estimate
            frequencies, psd = signal.periodogram(
                signal_array,
                fs=self.sampling_rate,
                scaling='density'
            )
        
        result = {
            'frequencies': frequencies,
            'psd': psd,
            'method': method
        }
        
        if use_cache:
            cache_key = (sig_hash, method)
            self._psd_cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear cached results (call when analyzing new signal)"""
        self._psd_cache.clear()
        self._fft_cache.clear()
    
    def get_dominant_frequency(self, signal_data, freq_range=(0.5, 10)):
        """
        Find dominant frequency in signal
        
        Args:
            signal_data: Time domain signal
            freq_range: Tuple of (min_freq, max_freq) to search
            
        Returns:
            float: Dominant frequency in Hz
        """
        fft_result = self.compute_fft(signal_data, use_cache=True)
        
        if fft_result is None:
            return None
        
        frequencies = fft_result['frequencies']
        magnitudes = fft_result['magnitudes']
        
        # Filter to frequency range of interest
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        filtered_freqs = frequencies[mask]
        filtered_mags = magnitudes[mask]
        
        if len(filtered_mags) == 0:
            return None
        
        # Find peak
        peak_idx = np.argmax(filtered_mags)
        dominant_freq = filtered_freqs[peak_idx]
        
        return float(dominant_freq)
    
    def analyze_frequency_bands(self, signal_data, psd_result=None):
        """
        OPTIMIZED: Analyze power in different frequency bands
        
        Args:
            signal_data: Time domain signal
            psd_result: Pre-computed PSD result (avoids recalculation)
        
        Frequency bands for eye movements:
            - Very Low (0-0.5 Hz): Drift, slow pursuit
            - Low (0.5-2 Hz): Normal saccades
            - Medium (2-5 Hz): Fast saccades, microsaccades
            - High (5+ Hz): Noise, tremor
            
        Returns:
            dict: Power distribution across bands
        """
        if psd_result is None:
            psd_result = self.compute_psd(signal_data, use_cache=True)
        
        if psd_result is None:
            return None
        
        frequencies = psd_result['frequencies']
        psd = psd_result['psd']
        
        # Define frequency bands
        bands = {
            'very_low': (0, 0.5),
            'low': (0.5, 2),
            'medium': (2, 5),
            'high': (5, self.sampling_rate/2)
        }
        
        total_power = np.trapz(psd, frequencies)
        
        band_powers = {}
        for band_name, (f_low, f_high) in bands.items():
            mask = (frequencies >= f_low) & (frequencies < f_high)
            band_power = np.trapz(psd[mask], frequencies[mask])
            band_powers[band_name] = {
                'absolute_power': float(band_power),
                'relative_power': float(band_power / total_power * 100) if total_power > 0 else 0,
                'frequency_range': (f_low, f_high)
            }
        
        return band_powers
    
    def compute_spectral_entropy(self, signal_data, psd_result=None):
        """
        OPTIMIZED: Calculate spectral entropy with optional pre-computed PSD
        """
        if psd_result is None:
            psd_result = self.compute_psd(signal_data, use_cache=True)
        
        if psd_result is None:
            return None
        
        psd = psd_result['psd']
        
        # Normalize PSD to probability distribution
        psd_norm = psd / np.sum(psd)
        
        # Remove zero values
        psd_norm = psd_norm[psd_norm > 0]
        
        # Calculate entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(psd_norm))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    def compute_spectral_centroid(self, signal_data, psd_result=None):
        """
        OPTIMIZED: Calculate spectral centroid with optional pre-computed PSD
        """
        if psd_result is None:
            psd_result = self.compute_psd(signal_data, use_cache=True)
        
        if psd_result is None:
            return None
        
        frequencies = psd_result['frequencies']
        psd = psd_result['psd']
        
        # Weighted average
        centroid = np.sum(frequencies * psd) / np.sum(psd)
        
        return float(centroid)
    
    def compute_spectral_rolloff(self, signal_data, rolloff_percent=0.85, psd_result=None):
        """
        OPTIMIZED: Calculate spectral rolloff with optional pre-computed PSD
        """
        if psd_result is None:
            psd_result = self.compute_psd(signal_data, use_cache=True)
        
        if psd_result is None:
            return None
        
        frequencies = psd_result['frequencies']
        psd = psd_result['psd']
        
        # Cumulative sum
        cumsum = np.cumsum(psd)
        total = cumsum[-1]
        
        # Find frequency where cumsum reaches threshold
        threshold = rolloff_percent * total
        rolloff_idx = np.where(cumsum >= threshold)[0][0]
        rolloff_freq = frequencies[rolloff_idx]
        
        return float(rolloff_freq)
    
    def extract_spectral_features(self, signal_data):
        """
        OPTIMIZED: Extract comprehensive spectral features
        
        KEY OPTIMIZATION: Compute PSD once, reuse for all metrics
        
        Returns:
            dict: All spectral features
        """
        features = {}
        
        # Clear cache for new signal
        self.clear_cache()
        
        # 1. Compute PSD ONCE
        psd_result = self.compute_psd(signal_data, use_cache=True)
        
        if psd_result is None:
            return features
        
        # 2. Dominant frequency - uses FFT (separate from PSD)
        dom_freq = self.get_dominant_frequency(signal_data)
        features['dominant_frequency'] = dom_freq
        
        # 3. ALL other features reuse the SAME PSD
        features['frequency_bands'] = self.analyze_frequency_bands(signal_data, psd_result)
        features['spectral_entropy'] = self.compute_spectral_entropy(signal_data, psd_result)
        features['spectral_centroid'] = self.compute_spectral_centroid(signal_data, psd_result)
        features['spectral_rolloff'] = self.compute_spectral_rolloff(signal_data, psd_result=psd_result)
        
        # 4. Total power
        features['total_power'] = float(np.trapz(psd_result['psd'], psd_result['frequencies']))
        
        return features
    
    def create_spectrogram(self, signal_data, window_size=128, overlap=64):
        """
        Create time-frequency spectrogram
        
        Args:
            signal_data: Time domain signal
            window_size: Window size in samples
            overlap: Overlap between windows in samples
            
        Returns:
            dict: Time, frequency, and magnitude arrays
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < window_size:
            return None
        
        # Compute spectrogram
        frequencies, times, Sxx = signal.spectrogram(
            signal_array,
            fs=self.sampling_rate,
            nperseg=window_size,
            noverlap=overlap,
            scaling='spectrum'
        )
        
        return {
            'times': times,
            'frequencies': frequencies,
            'spectrogram': Sxx
        }
