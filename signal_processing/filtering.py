import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

# Optional: PyWavelets for advanced denoising
try:
    import pywt # type: ignore
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

class SignalFilter:
    """
    Signal filtering and smoothing techniques
    Noise reduction and signal enhancement
    """
    
    def __init__(self):
        # Kalman filter state variables
        self.kalman_x = None
        self.kalman_P = None
        self.kalman_initialized = False
    
    def moving_average(self, signal_data, window_size=5):
        """
        Simple moving average filter
        
        Args:
            signal_data: Input signal
            window_size: Window size for averaging
            
        Returns:
            array: Smoothed signal
        """
        signal_array = np.array(signal_data)
        
        if len(signal_array) < window_size:
            return signal_array
        
        # Use convolution for efficiency
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(signal_array, window, mode='same')
        
        return smoothed
    
    def exponential_moving_average(self, signal_data, alpha=0.3):
        """
        Exponential moving average (gives more weight to recent data)
        
        Args:
            signal_data: Input signal
            alpha: Smoothing factor (0 < alpha < 1)
                  Higher alpha = more responsive, less smooth
                  
        Returns:
            array: Smoothed signal
        """
        signal_array = np.array(signal_data)
        smoothed = np.zeros_like(signal_array)
        
        smoothed[0] = signal_array[0]
        for i in range(1, len(signal_array)):
            smoothed[i] = alpha * signal_array[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    def median_filter_1d(self, signal_data, kernel_size=5):
        """
        Median filter - good for removing outliers/spikes
        
        Args:
            signal_data: Input signal
            kernel_size: Size of median kernel
            
        Returns:
            array: Filtered signal
        """
        signal_array = np.array(signal_data)
        filtered = median_filter(signal_array, size=kernel_size)
        return filtered
    
    def butterworth_lowpass(self, signal_data, cutoff=5, fs=30, order=4):
        """
        Butterworth low-pass filter
        Removes high-frequency noise
        
        Args:
            signal_data: Input signal
            cutoff: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order (higher = sharper cutoff)
            
        Returns:
            array: Filtered signal
        """
        signal_array = np.array(signal_data)
        
        # Design filter
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply filter (forward-backward to avoid phase shift)
        filtered = signal.filtfilt(b, a, signal_array)
        
        return filtered
    
    def butterworth_highpass(self, signal_data, cutoff=0.5, fs=30, order=4):
        """
        Butterworth high-pass filter
        Removes low-frequency drift
        
        Args:
            signal_data: Input signal
            cutoff: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order
            
        Returns:
            array: Filtered signal
        """
        signal_array = np.array(signal_data)
        
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        
        filtered = signal.filtfilt(b, a, signal_array)
        
        return filtered
    
    def butterworth_bandpass(self, signal_data, lowcut=0.5, highcut=5, fs=30, order=4):
        """
        Butterworth band-pass filter
        Keeps frequencies in specified range
        
        Args:
            signal_data: Input signal
            lowcut: Low cutoff frequency in Hz
            highcut: High cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order
            
        Returns:
            array: Filtered signal
        """
        signal_array = np.array(signal_data)
        
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        
        filtered = signal.filtfilt(b, a, signal_array)
        
        return filtered
    
    def savitzky_golay(self, signal_data, window_length=11, polyorder=3):
        """
        Savitzky-Golay filter
        Preserves features better than moving average
        
        Args:
            signal_data: Input signal
            window_length: Length of filter window (must be odd)
            polyorder: Order of polynomial fit
            
        Returns:
            array: Smoothed signal
        """
        signal_array = np.array(signal_data)
        
        if len(signal_array) < window_length:
            return signal_array
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        filtered = signal.savgol_filter(signal_array, window_length, polyorder)
        
        return filtered
    
    def kalman_filter_init(self, initial_value, process_variance=0.01, measurement_variance=0.1):
        """
        Initialize Kalman filter
        
        Args:
            initial_value: Initial state estimate
            process_variance: Process noise variance (Q)
            measurement_variance: Measurement noise variance (R)
        """
        self.kalman_x = initial_value
        self.kalman_P = 1.0
        self.Q = process_variance
        self.R = measurement_variance
        self.kalman_initialized = True
    
    def kalman_filter_step(self, measurement):
        """
        Single Kalman filter update step
        Optimal for real-time filtering
        
        Args:
            measurement: New measurement value
            
        Returns:
            float: Filtered estimate
        """
        if not self.kalman_initialized:
            self.kalman_filter_init(measurement)
            return measurement
        
        # Prediction
        x_pred = self.kalman_x
        P_pred = self.kalman_P + self.Q
        
        # Update
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.kalman_x = x_pred + K * (measurement - x_pred)
        self.kalman_P = (1 - K) * P_pred
        
        return self.kalman_x
    
    def kalman_filter_batch(self, signal_data, process_variance=0.01, measurement_variance=0.1):
        """
        Apply Kalman filter to entire signal
        
        Args:
            signal_data: Input signal
            process_variance: Process noise (smaller = smoother)
            measurement_variance: Measurement noise
            
        Returns:
            array: Filtered signal
        """
        signal_array = np.array(signal_data)
        filtered = np.zeros_like(signal_array)
        
        # Initialize
        x = signal_array[0]
        P = 1.0
        
        for i in range(len(signal_array)):
            # Prediction
            x_pred = x
            P_pred = P + process_variance
            
            # Update
            K = P_pred / (P_pred + measurement_variance)
            x = x_pred + K * (signal_array[i] - x_pred)
            P = (1 - K) * P_pred
            
            filtered[i] = x
        
        return filtered
    
    def outlier_removal(self, signal_data, threshold=3):
        """
        Remove outliers using z-score method
        
        Args:
            signal_data: Input signal
            threshold: Z-score threshold (default 3 = 3 standard deviations)
            
        Returns:
            tuple: (filtered_signal, outlier_mask)
        """
        signal_array = np.array(signal_data)
        
        # Calculate z-scores
        mean = np.mean(signal_array)
        std = np.std(signal_array)
        z_scores = np.abs((signal_array - mean) / (std + 1e-10))
        
        # Identify outliers
        outlier_mask = z_scores > threshold
        
        # Replace outliers with interpolated values
        filtered = signal_array.copy()
        outlier_indices = np.where(outlier_mask)[0]
        
        for idx in outlier_indices:
            # Find nearest non-outlier values
            left_idx = idx - 1
            right_idx = idx + 1
            
            while left_idx >= 0 and outlier_mask[left_idx]:
                left_idx -= 1
            while right_idx < len(signal_array) and outlier_mask[right_idx]:
                right_idx += 1
            
            # Interpolate
            if left_idx >= 0 and right_idx < len(signal_array):
                filtered[idx] = (signal_array[left_idx] + signal_array[right_idx]) / 2
            elif left_idx >= 0:
                filtered[idx] = signal_array[left_idx]
            elif right_idx < len(signal_array):
                filtered[idx] = signal_array[right_idx]
        
        return filtered, outlier_mask
    
    def adaptive_filter(self, signal_data, noise_estimate=None):
        """
        Adaptive Wiener-like filter
        Adjusts filtering based on local signal characteristics
        
        Args:
            signal_data: Input signal
            noise_estimate: Estimated noise variance (auto if None)
            
        Returns:
            array: Filtered signal
        """
        signal_array = np.array(signal_data)
        
        if noise_estimate is None:
            # Estimate noise from high-frequency components
            diff = np.diff(signal_array)
            noise_estimate = np.var(diff) / 2
        
        # Window-based filtering
        window_size = 11
        filtered = np.zeros_like(signal_array)
        
        for i in range(len(signal_array)):
            start = max(0, i - window_size // 2)
            end = min(len(signal_array), i + window_size // 2 + 1)
            window = signal_array[start:end]
            
            # Local variance
            local_var = np.var(window)
            
            # Wiener gain
            if local_var > noise_estimate:
                gain = 1 - noise_estimate / local_var
            else:
                gain = 0
            
            # Apply filter
            local_mean = np.mean(window)
            filtered[i] = local_mean + gain * (signal_array[i] - local_mean)
        
        return filtered
    
    def denoise_wavelet(self, signal_data, wavelet='db4', level=None):
        """
        Wavelet denoising
        Good for preserving sharp features while removing noise
        
        Args:
            signal_data: Input signal
            wavelet: Wavelet type
            level: Decomposition level (auto if None)
            
        Returns:
            array: Denoised signal
            
        Note:
            Requires PyWavelets: pip install PyWavelets
            Falls back to Savitzky-Golay if not available
        """
        if not PYWT_AVAILABLE:
            print("⚠️ PyWavelets not installed. Using Savitzky-Golay instead.")
            print("   Install with: pip install PyWavelets")
            return self.savitzky_golay(signal_data)
        
        signal_array = np.array(signal_data)
        
        if level is None:
            level = min(5, pywt.dwt_max_level(len(signal_array), wavelet))
        
        # Decompose
        coeffs = pywt.wavedec(signal_array, wavelet, level=level)
        
        # Threshold detail coefficients (soft thresholding)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Robust noise estimate
        threshold = sigma * np.sqrt(2 * np.log(len(signal_array)))
        
        coeffs_thresh = [coeffs[0]]  # Keep approximation
        for coeff in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(coeff, threshold, mode='soft'))
        
        # Reconstruct
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        
        # Handle length mismatch
        if len(denoised) > len(signal_array):
            denoised = denoised[:len(signal_array)]
        
        return denoised