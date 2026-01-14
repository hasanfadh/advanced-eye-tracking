import numpy as np
from scipy import stats

class QualityAssessor:
    """
    Assess signal quality and tracking reliability
    Important for research: know when data is trustworthy
    """
    
    def __init__(self):
        pass
    
    def calculate_snr(self, signal_data, noise_data=None):
        """
        Calculate Signal-to-Noise Ratio
        
        Args:
            signal_data: Clean signal or full signal
            noise_data: Noise signal (if None, estimate from high-freq)
            
        Returns:
            float: SNR in dB
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 2:
            return None
        
        # Signal power
        signal_power = np.mean(signal_array**2)
        
        if noise_data is None:
            # Estimate noise from differences (high-frequency component)
            diff = np.diff(signal_array)
            noise_power = np.var(diff) / 2
        else:
            noise_array = np.array(noise_data)
            noise_power = np.mean(noise_array**2)
        
        if noise_power == 0:
            return float('inf')
        
        # SNR in dB
        snr = 10 * np.log10(signal_power / noise_power)
        
        return float(snr)
    
    def calculate_rmse(self, signal_data, reference_data):
        """
        Root Mean Square Error compared to reference
        
        Args:
            signal_data: Measured signal
            reference_data: Ground truth or reference signal
            
        Returns:
            float: RMSE value
        """
        signal_array = np.array(signal_data)
        reference_array = np.array(reference_data)
        
        if len(signal_array) != len(reference_array):
            # Interpolate to match lengths
            min_len = min(len(signal_array), len(reference_array))
            signal_array = signal_array[:min_len]
            reference_array = reference_array[:min_len]
        
        mse = np.mean((signal_array - reference_array)**2)
        rmse = np.sqrt(mse)
        
        return float(rmse)
    
    def calculate_mae(self, signal_data, reference_data):
        """
        Mean Absolute Error
        
        Returns:
            float: MAE value
        """
        signal_array = np.array(signal_data)
        reference_array = np.array(reference_data)
        
        if len(signal_array) != len(reference_array):
            min_len = min(len(signal_array), len(reference_array))
            signal_array = signal_array[:min_len]
            reference_array = reference_array[:min_len]
        
        mae = np.mean(np.abs(signal_array - reference_array))
        
        return float(mae)
    
    def calculate_correlation(self, signal_data, reference_data):
        """
        Pearson correlation coefficient
        
        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        signal_array = np.array(signal_data)
        reference_array = np.array(reference_data)
        
        if len(signal_array) != len(reference_array):
            min_len = min(len(signal_array), len(reference_array))
            signal_array = signal_array[:min_len]
            reference_array = reference_array[:min_len]
        
        if len(signal_array) < 2:
            return None
        
        corr, _ = stats.pearsonr(signal_array, reference_array)
        
        return float(corr)
    
    def assess_tracking_stability(self, gaze_h, gaze_v, window_size=30):
        """
        Assess stability of gaze tracking
        
        High stability = consistent tracking, low jitter
        Low stability = noisy, unreliable
        
        Args:
            gaze_h: Horizontal gaze signal
            gaze_v: Vertical gaze signal
            window_size: Window for local stability calculation
            
        Returns:
            float: Stability index (0 to 1, higher is better)
        """
        gaze_h = np.array(gaze_h)
        gaze_v = np.array(gaze_v)
        
        if len(gaze_h) < window_size:
            window_size = len(gaze_h) // 2
        
        if window_size < 2:
            return None
        
        # Calculate local variability
        stabilities = []
        for i in range(0, len(gaze_h) - window_size, window_size // 2):
            window_h = gaze_h[i:i+window_size]
            window_v = gaze_v[i:i+window_size]
            
            # Velocity in window
            vel_h = np.diff(window_h)
            vel_v = np.diff(window_v)
            velocity_mag = np.sqrt(vel_h**2 + vel_v**2)
            
            # Stability inversely proportional to velocity variance
            vel_std = np.std(velocity_mag)
            stability = 1.0 / (1.0 + vel_std)
            stabilities.append(stability)
        
        avg_stability = np.mean(stabilities) if stabilities else 0.0
        
        return float(avg_stability)
    
    def detect_missing_data(self, data_history):
        """
        Detect missing or invalid data points
        
        Args:
            data_history: List of tracking data dictionaries
            
        Returns:
            dict: Missing data statistics
        """
        total_frames = len(data_history)
        
        if total_frames == 0:
            return None
        
        # Count frames with missing face detection
        no_face = sum(1 for d in data_history if not d.get('face_detected', False))
        
        # Count NaN values in gaze data
        nan_h = sum(1 for d in data_history if d.get('gaze_h_ratio') is None or np.isnan(d.get('gaze_h_ratio', 0)))
        nan_v = sum(1 for d in data_history if d.get('gaze_v_ratio') is None or np.isnan(d.get('gaze_v_ratio', 0)))
        
        return {
            'total_frames': total_frames,
            'frames_no_face': no_face,
            'missing_face_percent': (no_face / total_frames) * 100,
            'frames_nan_h': nan_h,
            'frames_nan_v': nan_v,
            'data_completeness': ((total_frames - no_face) / total_frames) * 100
        }
    
    def assess_sampling_consistency(self, timestamps):
        """
        Check if sampling rate is consistent
        
        Args:
            timestamps: List of timestamp strings or datetime objects
            
        Returns:
            dict: Sampling statistics
        """
        if len(timestamps) < 2:
            return None
        
        # Convert to seconds if needed
        if isinstance(timestamps[0], str):
            from datetime import datetime
            times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in timestamps]
            intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        else:
            intervals = np.diff(timestamps)
        
        intervals = np.array(intervals)
        
        return {
            'mean_interval': float(np.mean(intervals)),
            'std_interval': float(np.std(intervals)),
            'min_interval': float(np.min(intervals)),
            'max_interval': float(np.max(intervals)),
            'estimated_fps': float(1.0 / np.mean(intervals)) if np.mean(intervals) > 0 else 0,
            'consistency_score': float(1.0 - (np.std(intervals) / np.mean(intervals))) if np.mean(intervals) > 0 else 0
        }
    
    def calculate_confidence_score(self, data_history):
        """
        Overall confidence/quality score for tracking session
        
        Combines multiple quality metrics into single score
        
        Returns:
            float: Confidence score (0 to 1)
        """
        if not data_history:
            return 0.0
        
        scores = []
        
        # Data completeness
        missing = self.detect_missing_data(data_history)
        if missing:
            completeness_score = missing['data_completeness'] / 100
            scores.append(completeness_score)
        
        # Extract gaze signals
        gaze_h = [d['gaze_h_ratio'] for d in data_history if d.get('face_detected') and d.get('gaze_h_ratio') is not None]
        gaze_v = [d['gaze_v_ratio'] for d in data_history if d.get('face_detected') and d.get('gaze_v_ratio') is not None]
        
        if len(gaze_h) > 30 and len(gaze_v) > 30:
            # Tracking stability
            stability = self.assess_tracking_stability(gaze_h, gaze_v)
            if stability:
                scores.append(stability)
            
            # Signal quality (lower variance in velocity = better)
            vel_h = np.diff(gaze_h)
            vel_v = np.diff(gaze_v)
            vel_mag = np.sqrt(vel_h**2 + vel_v**2)
            
            # Normalize velocity variance (empirical threshold)
            vel_std = np.std(vel_mag)
            quality_score = 1.0 / (1.0 + vel_std)
            scores.append(quality_score)
        
        # Sampling consistency
        timestamps = [d['timestamp'] for d in data_history]
        sampling = self.assess_sampling_consistency(timestamps)
        if sampling:
            scores.append(max(0, sampling['consistency_score']))
        
        # Overall confidence
        if scores:
            confidence = np.mean(scores)
        else:
            confidence = 0.0
        
        return float(np.clip(confidence, 0, 1))
    
    def assess_signal_quality(self, data_history):
        """
        Comprehensive signal quality assessment
        
        Returns:
            dict: Complete quality report
        """
        if not data_history:
            return None
        
        # Extract signals
        gaze_h = [d['gaze_h_ratio'] for d in data_history if d.get('face_detected') and d.get('gaze_h_ratio') is not None]
        gaze_v = [d['gaze_v_ratio'] for d in data_history if d.get('face_detected') and d.get('gaze_v_ratio') is not None]
        
        report = {}
        
        # Missing data
        report['missing_data'] = self.detect_missing_data(data_history)
        
        # 1. Signal-to-Noise Ratio
        if len(gaze_h) > 10:
            report['snr_horizontal'] = self.calculate_snr(gaze_h)
            report['snr_vertical'] = self.calculate_snr(gaze_v)
        # Example Output: SNR = 23.5 dB -> Good quality
        
        # 2. Tracking Stability
        if len(gaze_h) > 30:
            report['tracking_stability'] = self.assess_tracking_stability(gaze_h, gaze_v)
        # Example Output: Stability = 0.85 -> Stable tracking

        # 3. Sampling consistency
        timestamps = [d['timestamp'] for d in data_history]
        report['sampling'] = self.assess_sampling_consistency(timestamps)
        # Example Output: Estimated FPS = 29.8, Consistency = 0.95

        # 4. Overall confidence
        report['confidence_score'] = self.calculate_confidence_score(data_history)
        # Example Output: Confidence Score = 0.88 -> Reliable data

        # Quality rating
        confidence = report['confidence_score']
        if confidence >= 0.8:
            report['quality_rating'] = 'Excellent'
        elif confidence >= 0.6:
            report['quality_rating'] = 'Good'
        elif confidence >= 0.4:
            report['quality_rating'] = 'Fair'
        else:
            report['quality_rating'] = 'Poor'
        
        return report