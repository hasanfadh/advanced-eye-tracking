import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class DataLogger:
    """
    Handles data logging, export, and analysis
    """
    
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_to_csv(self, data_history, filename=None):
        """Export tracking data to CSV"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"eye_tracking_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        df = pd.DataFrame(data_history)
        df.to_csv(filepath, index=False)
        print(f"Data exported to: {filepath}")
        return filepath
    
    def export_to_json(self, data_history, filename=None):
        """Export tracking data to JSON"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"eye_tracking_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data_history, f, indent=2, default=self.convert_numpy)
        print(f"Data exported to: {filepath}")
        return filepath
    
    def generate_summary_report(self, data_history):
        """Generate summary statistics"""
        df = pd.DataFrame(data_history)
        
        # Filter only frames where face was detected
        df_detected = df[df['face_detected'] == True]
        
        if len(df_detected) == 0:
            print("⚠️ No face detected in the data")
            return None
        
        report = {
            'total_frames': len(df),
            'frames_with_face': len(df_detected),
            'detection_rate': len(df_detected) / len(df) * 100,
            'total_blinks': int(df['blink_count'].max()) if 'blink_count' in df else 0,
            'gaze_distribution': {},
            'average_ear': {
                'left': float(df_detected['left_ear'].mean()) if 'left_ear' in df_detected else None,
                'right': float(df_detected['right_ear'].mean()) if 'right_ear' in df_detected else None
            },
            'gaze_ratios': {
                'horizontal_mean': float(df_detected['gaze_h_ratio'].mean()) if 'gaze_h_ratio' in df_detected else None,
                'horizontal_std': float(df_detected['gaze_h_ratio'].std()) if 'gaze_h_ratio' in df_detected else None,
                'vertical_mean': float(df_detected['gaze_v_ratio'].mean()) if 'gaze_v_ratio' in df_detected else None,
                'vertical_std': float(df_detected['gaze_v_ratio'].std()) if 'gaze_v_ratio' in df_detected else None
            }
        }
        
        # Gaze direction distribution
        if 'gaze_direction' in df_detected.columns:
            gaze_counts = df_detected['gaze_direction'].value_counts()
            total = len(df_detected)
            report['gaze_distribution'] = {
                direction: {
                    'count': int(count),
                    'percentage': float(count / total * 100)
                }
                for direction, count in gaze_counts.items()
            }
        
        return report
    
    def generate_signal_analysis_report(self, data_history):
        """
        Generate comprehensive signal processing analysis
        """
        try:
            from signal_processing import (
                TimeDomainAnalyzer,
                FrequencyAnalyzer,
                NonlinearAnalyzer,  
                QualityAssessor
            )
        except ImportError as e:
            print(f"⚠️ Signal processing module not found: {e}")
            print("   Make sure signal_processing/ folder exists with all modules.")
            return None
        
        # Extract signals
        df = pd.DataFrame(data_history)
        df_detected = df[df['face_detected'] == True]

        if len(df_detected) < 30:
            print("⚠️ Not enough data for signal analysis (need at least 30 frames with face detected)")
            return None
        
        gaze_h = df_detected['gaze_h_ratio'].dropna().values
        gaze_v = df_detected['gaze_v_ratio'].dropna().values

        if len(gaze_h) < 30:
            print("⚠️ Not enough valid gaze horizontal ratio data for signal analysis")
            return None
        
        print("\n Performing signal processing analysis...")

        report = {}

        # 1. Time Domain Analysis
        print("  Time domain analysis...")
        td_analyzer = TimeDomainAnalyzer()
        # a. Basic Statistics
        report['time_domain'] = {
            'horizontal': td_analyzer.extract_features(gaze_h),
            'vertical': td_analyzer.extract_features(gaze_v)
        }

        # b. Fixation and Saccade Detection
        fixations = td_analyzer.detect_fixations(gaze_h, gaze_v)
        saccades = td_analyzer.detect_saccades(gaze_h, gaze_v)

        report['fixations'] = {
            'count': len(fixations),
            'avg_duration_ms': float(np.mean([f['duration_ms'] for f in fixations])) if fixations else 0,
            'total_duration_ms': float(np.sum([f['duration_ms'] for f in fixations])),
            'fixations': fixations[:10]  # save first 10 for inspection
        }

        report['saccades'] = {
            'count': len(saccades),
            'avg_amplitude': float(np.mean([s['amplitude'] for s in saccades])) if saccades else 0,
            'avg_peak_velocity': float(np.mean([s['peak_velocity'] for s in saccades])) if saccades else 0,
            'saccades': saccades[:10]  # save first 10 for inspection
        }

        # 2. Frequency Domain Analysis
        print("  Frequency domain analysis...")
        freq_analyzer = FrequencyAnalyzer(sampling_rate=30)
        report['frequency_domain'] = {
            'horizontal': freq_analyzer.extract_spectral_features(gaze_h),
            'vertical': freq_analyzer.extract_spectral_features(gaze_v)
        }

        # 3. Non-Linear Analysis
        print("  Nonlinear dynamics analysis...")
        nl_analyzer = NonlinearAnalyzer()
        report['nonlinear'] = {
            'horizontal': nl_analyzer.extract_nonlinear_features(gaze_h),
            'vertical': nl_analyzer.extract_nonlinear_features(gaze_v)
        }

        # 4. Signal Quality Assessment
        print("  Signal quality assessment...")
        quality_assessor = QualityAssessor()
        report['quality'] = quality_assessor.assess_signal_quality(data_history)

        print("Signal processing analysis complete!")

        return report

    def print_summary(self, report):
        """Print summary report in readable format"""
        if not report:
            return
        
        print("\n" + "="*50)
        print(" EYE TRACKING SUMMARY REPORT")
        print("="*50)
        
        print(f"\n Recording Statistics:")
        print(f"  Total frames: {report['total_frames']}")
        print(f"  Frames with face: {report['frames_with_face']}")
        print(f"  Detection rate: {report['detection_rate']:.2f}%")
        print(f"  Total blinks: {report['total_blinks']}")
        
        print(f"\n Eye Aspect Ratio (EAR):")
        if report['average_ear']['left']:
            print(f"  Left eye: {report['average_ear']['left']:.4f}")
            print(f"  Right eye: {report['average_ear']['right']:.4f}")
        
        print(f"\n Gaze Direction Distribution:")
        for direction, stats in sorted(report['gaze_distribution'].items(), 
                                       key=lambda x: x[1]['percentage'], 
                                       reverse=True):
            print(f"  {direction:15s}: {stats['count']:5d} frames ({stats['percentage']:5.2f}%)")
        
        print(f"\n Gaze Ratios (normalized -1 to 1):")
        gr = report['gaze_ratios']
        if gr['horizontal_mean'] is not None:
            print(f"  Horizontal: {gr['horizontal_mean']:+.4f} ± {gr['horizontal_std']:.4f}")
            print(f"  Vertical:   {gr['vertical_mean']:+.4f} ± {gr['vertical_std']:.4f}")
        
        print("="*50 + "\n")

    def print_signal_analysis(self, report):
        """
        Print comprehensive signal processing report
        """
        if not report:
            return
        
        print("\n" + "="*70)
        print(" SIGNAL PROCESSING ANALYSIS REPORT")
        print("="*70)

        # Time Domain
        if 'time_domain' in report:
            print("\n TIME DOMAIN ANALYSIS:")
            td_h = report['time_domain']['horizontal']
            td_v = report['time_domain']['vertical']

            if td_h:
                print("  Horizontal gaze:")
                print(f"    Mean: {td_h['mean']:+.4f}, Std: {td_h['std']:.4f}")
                print(f"    Velocity: {td_h.get('mean_velocity', 0):.2f} °/s (avg), {td_h.get('max_velocity', 0):.2f} °/s (max)")  # ✅ FIXED
                print(f"    Peaks detected: {td_h.get('num_peaks', 0)}")

            if td_v:
                print("  Vertical gaze:")
                print(f"    Mean: {td_v['mean']:+.4f}, Std: {td_v['std']:.4f}")
                print(f"    Velocity: {td_v.get('mean_velocity', 0):.2f} °/s (avg), {td_v.get('max_velocity', 0):.2f} °/s (max)")  # ✅ FIXED
                print(f"    Peaks detected: {td_v.get('num_peaks', 0)}")

        # Fixations and Saccades
        if 'fixations' in report:
            print(f"\n EYE MOVEMENTS:")
            print(f"  Fixations:")
            print(f"    Count: {report['fixations']['count']}")
            print(f"    Avg duration: {report['fixations']['avg_duration_ms']:.1f} ms")
            print(f"    Total duration: {report['fixations']['total_duration_ms']:.1f} ms")
            
            print(f"  Saccades:")
            print(f"    Count: {report['saccades']['count']}")
            print(f"    Avg amplitude: {report['saccades']['avg_amplitude']:.4f}")
            print(f"    Avg peak velocity: {report['saccades']['avg_peak_velocity']:.2f} °/s")

        # Frequency Domain
        if 'frequency_domain' in report:
            print(f"\n FREQUENCY DOMAIN ANALYSIS:")
            fd_h = report['frequency_domain']['horizontal']
            fd_v = report['frequency_domain']['vertical']

            if fd_h and fd_h.get('dominant_frequency'):
                print(f"  Horizontal:")
                print(f"    Dominant frequency: {fd_h.get('dominant_frequency', 0):.2f} Hz")  # ✅ FIXED
                print(f"    Spectral entropy: {fd_h.get('spectral_entropy', 0):.4f}")

                if 'frequency_bands' in fd_h and fd_h['frequency_bands']:
                    bands = fd_h['frequency_bands']
                    print(f"    Power distribution:")
                    for band_name, band_data in bands.items():
                        print(f"      {band_name}: {band_data['relative_power']:.1f}%")
                        
            if fd_v and fd_v.get('dominant_frequency'):
                print(f"  Vertical:")
                print(f"    Dominant frequency: {fd_v.get('dominant_frequency', 0):.2f} Hz")  # ✅ FIXED
                print(f"    Spectral entropy: {fd_v.get('spectral_entropy', 0):.4f}")

        # Nonlinear Analysis
        if 'nonlinear' in report:
            print(f"\n NONLINEAR DYNAMICS ANALYSIS:")
            nl_h = report['nonlinear']['horizontal']
            nl_v = report['nonlinear']['vertical']

            if nl_h:
                print(f"  Horizontal:")
                if nl_h.get('sample_entropy') is not None:
                    print(f"    Sample Entropy: {nl_h['sample_entropy']:.4f}")
                if nl_h.get('fractal_dimension') is not None:
                    print(f"    Fractal Dimension: {nl_h['fractal_dimension']:.4f}")
                if nl_h.get('lyapunov_exponent') is not None:
                    print(f"    Lyapunov Exponent: {nl_h['lyapunov_exponent']:+.4f}")
                if nl_h.get('dfa_exponent') is not None:
                    print(f"    DFA Exponent: {nl_h['dfa_exponent']:.4f}")

            if nl_v:
                print(f"  Vertical:")
                if nl_v.get('sample_entropy') is not None:
                    print(f"    Sample Entropy: {nl_v['sample_entropy']:.4f}")
                if nl_v.get('fractal_dimension') is not None:
                    print(f"    Fractal Dimension: {nl_v['fractal_dimension']:.4f}")
                if nl_v.get('lyapunov_exponent') is not None:
                    print(f"    Lyapunov Exponent: {nl_v['lyapunov_exponent']:+.4f}")
                if nl_v.get('dfa_exponent') is not None:
                    print(f"    DFA Exponent: {nl_v['dfa_exponent']:.4f}")
        
        # Quality Assessment
        if 'quality' in report:
            print(f"\nSIGNAL QUALITY:")
            quality = report['quality']

            if quality.get('confidence_score') is not None:
                print(f"  Overall confidence: {quality['confidence_score']:.2f}")
                print(f"  Quality rating: {quality.get('quality_rating', 'N/A')}")

            if quality.get('snr_horizontal') is not None:
                print(f"  SNR horizontal: {quality['snr_horizontal']:.2f} dB")
            if quality.get('snr_vertical') is not None:
                print(f"  SNR vertical: {quality['snr_vertical']:.2f} dB")

            if quality.get('tracking_stability') is not None:
                print(f"  Tracking stability: {quality['tracking_stability']:.2f}")

            if quality.get('missing_data'):
                missing = quality['missing_data']
                print(f"  Data completeness: {missing['data_completeness']:.1f}%")

        print("\n" + "="*70 + "\n")

    @staticmethod
    def convert_numpy(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def export_summary(self, report, filename=None):
        """Export summary report to JSON"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"summary_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=self.convert_numpy)
        print(f"Summary exported to: {filepath}")
        return filepath
    
    def export_signal_analysis(self, report, filename=None):
        """Export signal analysis report to JSON"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"signal_analysis_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=self.convert_numpy)
        print(f"Signal analysis exported to: {filepath}")
        return filepath