import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis

class TimeDomainAnalyzer:
    """
    Time domain analysis for eye tracking signals
    Extracts temporal features and characteristics
    
    FIXED: Proper velocity conversion for normalized gaze data
    """
    
    def __init__(self, degrees_per_unit=40):
        """
        Args:
            degrees_per_unit: Conversion factor from normalized units to degrees
                            Default 40 = assumes normalized range (-1 to 1) represents ~40° FOV
        """
        self.DEGREES_PER_UNIT = degrees_per_unit
    
    def extract_features(self, signal_data):
        """
        Extract comprehensive time domain features
        
        Args:
            signal_data: numpy array or list of signal values
            
        Returns:
            dict: Dictionary of time domain features
        """
        signal_array = np.array(signal_data)
        
        # Remove NaN values
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) == 0:
            return None
        
        features = {
            # Basic statistics
            'mean': float(np.mean(signal_array)),
            'std': float(np.std(signal_array)),
            'variance': float(np.var(signal_array)),
            'min': float(np.min(signal_array)),
            'max': float(np.max(signal_array)),
            'range': float(np.ptp(signal_array)),
            'median': float(np.median(signal_array)),
            
            # Distribution characteristics
            'skewness': float(skew(signal_array)),
            'kurtosis': float(kurtosis(signal_array)),
            
            # Amplitude features
            'rms': float(np.sqrt(np.mean(signal_array**2))),
            'peak_amplitude': float(np.max(np.abs(signal_array))),
            
            # Percentiles
            'q25': float(np.percentile(signal_array, 25)),
            'q75': float(np.percentile(signal_array, 75)),
            'iqr': float(np.percentile(signal_array, 75) - np.percentile(signal_array, 25)),
        }
        
        # Zero crossing rate
        features['zero_crossings'] = self._zero_crossing_rate(signal_array)
        
        # Peak detection
        peaks = self.detect_peaks(signal_array)
        features['num_peaks'] = len(peaks)
        
        # Velocity and acceleration (derivatives) - NOW IN DEGREES
        if len(signal_array) > 1:
            velocity = np.diff(signal_array) * self.DEGREES_PER_UNIT  # FIXED: Convert to degrees
            features['mean_velocity'] = float(np.mean(np.abs(velocity)))
            features['max_velocity'] = float(np.max(np.abs(velocity)))
            
            if len(velocity) > 1:
                acceleration = np.diff(velocity)  # Already in degrees/frame
                features['mean_acceleration'] = float(np.mean(np.abs(acceleration)))
                features['max_acceleration'] = float(np.max(np.abs(acceleration)))
        
        return features
    
    def _zero_crossing_rate(self, signal_array):
        """Calculate zero crossing rate"""
        zero_crossings = np.where(np.diff(np.sign(signal_array)))[0]
        return len(zero_crossings)
    
    def detect_peaks(self, signal_array, prominence=0.1):
        """
        Detect peaks in signal
        
        Args:
            signal_array: Signal data
            prominence: Minimum prominence of peaks
            
        Returns:
            array: Indices of detected peaks
        """
        peaks, properties = signal.find_peaks(signal_array, prominence=prominence)
        return peaks
    
    def calculate_velocity(self, signal_array, sampling_rate=30):
        """
        Calculate velocity (first derivative) in degrees/second
        
        Args:
            signal_array: Position signal (normalized -1 to 1)
            sampling_rate: Sampling frequency in Hz
            
        Returns:
            array: Velocity signal in degrees/second
            
        FIXED: properly converts normalized units to degrees
        """
        # Convert normalized units to degrees
        velocity = np.diff(signal_array) * self.DEGREES_PER_UNIT * sampling_rate
        return velocity
    
    def calculate_acceleration(self, signal_array, sampling_rate=30):
        """
        Calculate acceleration (second derivative) in degrees/second²
        
        Args:
            signal_array: Position signal (normalized -1 to 1)
            sampling_rate: Sampling frequency in Hz
            
        Returns:
            array: Acceleration signal in degrees/second²
        """
        velocity = self.calculate_velocity(signal_array, sampling_rate)
        acceleration = np.diff(velocity) * sampling_rate
        return acceleration
    
    def detect_fixations(self, gaze_h, gaze_v, velocity_threshold=30, duration_threshold=100, sampling_rate=30):
        """
        Detect fixation periods using I-VT (Velocity-Threshold) algorithm
        
        Args:
            gaze_h: Horizontal gaze signal (normalized -1 to 1)
            gaze_v: Vertical gaze signal (normalized -1 to 1)
            velocity_threshold: Velocity threshold in degrees/second (default 30)
            duration_threshold: Minimum fixation duration in ms (default 100)
            sampling_rate: Sampling rate in Hz (default 30)
            
        Returns:
            list: List of fixation dictionaries with start, end, duration, position
            
        FIXED: uses proper degree-based velocity calculation
        """
        gaze_h = np.array(gaze_h)
        gaze_v = np.array(gaze_v)
        
        # 1. Calculate velocity magnitude in degrees/second
        vel_h = self.calculate_velocity(gaze_h, sampling_rate)
        vel_v = self.calculate_velocity(gaze_v, sampling_rate)
        velocity_mag = np.sqrt(vel_h**2 + vel_v**2)
        
        # 2. Threshold to identify fixations (low velocity)
        is_fixation = velocity_mag < velocity_threshold
        
        # 3. Find continuous fixation periods
        fixations = []
        in_fixation = False
        fixation_start = 0
        
        for i, fix in enumerate(is_fixation):
            if fix and not in_fixation:
                # Start of fixation
                fixation_start = i
                in_fixation = True
            elif not fix and in_fixation:
                # End of fixation
                fixation_end = i
                duration_frames = fixation_end - fixation_start
                duration_ms = (duration_frames / sampling_rate) * 1000
                
                if duration_ms >= duration_threshold:
                    # Calculate average position during fixation
                    # Note: +1 because velocity array is 1 shorter than position array
                    avg_h = np.mean(gaze_h[fixation_start:fixation_end+2])
                    avg_v = np.mean(gaze_v[fixation_start:fixation_end+2])
                    
                    fixations.append({
                        'start_frame': fixation_start,
                        'end_frame': fixation_end + 1,
                        'duration_ms': float(duration_ms),
                        'position_h': float(avg_h),
                        'position_v': float(avg_v)
                    })
                
                in_fixation = False
        
        # Handle case where fixation continues to end of signal
        if in_fixation:
            fixation_end = len(is_fixation)
            duration_frames = fixation_end - fixation_start
            duration_ms = (duration_frames / sampling_rate) * 1000
            
            if duration_ms >= duration_threshold:
                avg_h = np.mean(gaze_h[fixation_start:])
                avg_v = np.mean(gaze_v[fixation_start:])
                
                fixations.append({
                    'start_frame': fixation_start,
                    'end_frame': len(gaze_h),
                    'duration_ms': float(duration_ms),
                    'position_h': float(avg_h),
                    'position_v': float(avg_v)
                })
        
        return fixations
    
    def detect_saccades(self, gaze_h, gaze_v, velocity_threshold=30, sampling_rate=30):
        """
        Detect saccade periods (fast eye movements)
        
        Args:
            gaze_h: Horizontal gaze signal (normalized -1 to 1)
            gaze_v: Vertical gaze signal (normalized -1 to 1)
            velocity_threshold: Velocity threshold in degrees/second (default 30)
            sampling_rate: Sampling rate in Hz (default 30)
            
        Returns:
            list: List of saccade dictionaries
            
        FIXED: uses proper degree-based velocity calculation
        """
        gaze_h = np.array(gaze_h)
        gaze_v = np.array(gaze_v)
        
        # Calculate velocity in degrees/second
        vel_h = self.calculate_velocity(gaze_h, sampling_rate)
        vel_v = self.calculate_velocity(gaze_v, sampling_rate)
        velocity_mag = np.sqrt(vel_h**2 + vel_v**2)
        
        # Saccades are high velocity periods
        is_saccade = velocity_mag >= velocity_threshold
        
        saccades = []
        in_saccade = False
        saccade_start = 0
        
        for i, sacc in enumerate(is_saccade):
            if sacc and not in_saccade:
                saccade_start = i
                in_saccade = True
            elif not sacc and in_saccade:
                saccade_end = i
                duration_frames = saccade_end - saccade_start
                duration_ms = (duration_frames / sampling_rate) * 1000
                
                # Calculate saccade amplitude (in normalized units first, then degrees)
                amplitude_h = gaze_h[saccade_end+1] - gaze_h[saccade_start]
                amplitude_v = gaze_v[saccade_end+1] - gaze_v[saccade_start]
                amplitude_norm = np.sqrt(amplitude_h**2 + amplitude_v**2)
                amplitude_deg = amplitude_norm * self.DEGREES_PER_UNIT
                
                # Peak velocity during saccade
                peak_velocity = np.max(velocity_mag[saccade_start:saccade_end+1])
                
                saccades.append({
                    'start_frame': saccade_start,
                    'end_frame': saccade_end + 1,
                    'duration_ms': float(duration_ms),
                    'amplitude': float(amplitude_deg),
                    'amplitude_norm': float(amplitude_norm),
                    'peak_velocity': float(peak_velocity)
                })
                
                in_saccade = False
        
        # Handle saccade at end of signal
        if in_saccade:
            saccade_end = len(is_saccade)
            duration_frames = saccade_end - saccade_start
            duration_ms = (duration_frames / sampling_rate) * 1000
            
            amplitude_h = gaze_h[-1] - gaze_h[saccade_start]
            amplitude_v = gaze_v[-1] - gaze_v[saccade_start]
            amplitude_norm = np.sqrt(amplitude_h**2 + amplitude_v**2)
            amplitude_deg = amplitude_norm * self.DEGREES_PER_UNIT
            
            peak_velocity = np.max(velocity_mag[saccade_start:])
            
            saccades.append({
                'start_frame': saccade_start,
                'end_frame': len(gaze_h),
                'duration_ms': float(duration_ms),
                'amplitude': float(amplitude_deg),
                'amplitude_norm': float(amplitude_norm),
                'peak_velocity': float(peak_velocity)
            })
        
        return saccades
    
    def analyze_temporal_pattern(self, signal_array, sampling_rate=30):
        """
        Comprehensive temporal pattern analysis
        
        Returns:
            dict: Temporal characteristics
        """
        signal_array = np.array(signal_array)
        
        # Trend analysis (linear fit)
        x = np.arange(len(signal_array))
        coeffs = np.polyfit(x, signal_array, 1)
        trend_slope = coeffs[0]
        
        # Stationarity measure (variance in sliding windows)
        window_size = min(60, len(signal_array) // 4)  # ~2 seconds window
        if window_size > 0:
            windows = [signal_array[i:i+window_size] for i in range(0, len(signal_array)-window_size, window_size)]
            variances = [np.var(w) for w in windows if len(w) > 0]
            if len(variances) > 0 and np.mean(variances) > 0:
                stationarity = 1.0 - (np.std(variances) / (np.mean(variances) + 1e-10))
            else:
                stationarity = 1.0
        else:
            stationarity = 1.0
        
        return {
            'trend_slope': float(trend_slope),
            'stationarity': float(np.clip(stationarity, 0, 1)),
            'duration_seconds': len(signal_array) / sampling_rate
        }