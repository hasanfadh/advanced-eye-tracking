import cv2
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(__file__))

from eye_tracker import EyeTracker
from utils.data_logger import DataLogger
from utils.visualizer import EyeTrackingVisualizer
from window_processor import WindowProcessor
from feature_printer import print_features
from attention_classifier import AttentionClassifier
from attention_display import AttentionDisplay

def main():
    """
    Main Eye Tracking Application
    
    Controls:
    - 'q': Quit and save data
    - 's': Save current frame as image
    - 'r': Reset tracking data
    - 'SPACE': Pause/Resume tracking
    """
    
    print("="*60)
    print("EYE TRACKING SYSTEM - NKUST Research Preparation")
    print("="*60)
    print("\nInitializing camera...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("Camera initialized successfully")
    
    # Initialize components
    tracker = EyeTracker()
    logger = DataLogger()
    logger.start_window_log()
    window_counter = 0
    visualizer = EyeTrackingVisualizer()
    processor = WindowProcessor(
        window_sec=3.0,
        step_sec=1.0,
        sampling_rate=30,
    )
    classifier = AttentionClassifier()
    att_display = AttentionDisplay()
    last_attention_result = None
    last_features = {}
    window_counter = 0
    
    print("\nControls:")
    print("  'q'     - Quit and save data")
    print("  's'     - Save current frame")
    print("  'r'     - Reset tracking data")
    print("  'SPACE' - Pause/Resume")
    print("  'v'     - Generate visualizations")
    print("\nStarting eye tracking...\n")
    
    is_paused = False
    frame_count = 0
    
    try:
        while True:
            if not is_paused:
                # 1. Capture frame
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                if not ret:
                    print("Error: Cannot read frame")
                    break
                
                # 2. Process frame
                processed_frame, data = tracker.process_frame(frame)
                
                # 3. Add to history if face detected
                if data['face_detected']:
                    tracker.add_to_history(data)
                    frame_count += 1

                    # Real-time windowed feature extraction
                    features = processor.update(data)
                    if features:
                        window_counter += 1
                        print_features(features, window_num=window_counter)
                        logger.log_window_features(features, window_num=window_counter)
                        # Attention classification
                        last_attention_result = classifier.classify(features)
                        last_features = features
                        # Print state to terminal (optional)
                        r = last_attention_result
                        print(f"  [{r.emoji} {r.state}] confidence={r.confidence:.0%}  score={r.score:+.1f}")
                
                # Add status text
                status = "PAUSED" if is_paused else "TRACKING"
                color = (0, 0, 255) if is_paused else (0, 255, 0)
                cv2.putText(processed_frame, status, (processed_frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add frame count
                cv2.putText(processed_frame, f"Frames: {frame_count}", 
                           (processed_frame.shape[1] - 150, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw attention overlay
                if last_attention_result is not None:
                    att_display.render(
                        frame,
                        last_attention_result,
                        last_features,
                        buffer_fill=processor.buffer_fill_ratio,
                    )
                elif processor.buffer_fill_ratio < 1.0:
                    # Show buffering progress before first window
                    _fill = processor.buffer_fill_ratio
                    cv2.putText(frame, f"Buffering... {_fill:.0%}",
                                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 220), 1, cv2.LINE_AA)
        
                # 4. Display
                cv2.imshow('Eye Tracking System - Press Q to quit', processed_frame)
            else:
                # Show paused message
                pause_frame = processed_frame.copy()
                overlay = pause_frame.copy()
                cv2.rectangle(overlay, (0, 0), (pause_frame.shape[1], 100), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, pause_frame, 0.3, 0, pause_frame)
                cv2.putText(pause_frame, "PAUSED - Press SPACE to resume", 
                           (pause_frame.shape[1]//2 - 250, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Eye Tracking System - Press Q to quit', pause_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nStopping tracking...")
                break
            elif key == ord('s'):
                # Save frame
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"frame_{timestamp}.jpg"
                cv2.imwrite(os.path.join('data', filename), processed_frame)
                print(f"Frame saved: {filename}")
            elif key == ord('r'):
                # Reset data
                tracker.clear_history()
                processor.reset()
                classifier.reset()
                frame_count = 0
                print("Tracking data reset")
            elif key == ord(' '):
                # Toggle pause
                is_paused = not is_paused
                status = "paused" if is_paused else "resumed"
                print(f"Tracking {status}")
            elif key == ord('v'):
                # Generate visualizations
                if len(tracker.get_history()) > 0:
                    print("\nGenerating visualizations...")
                    visualizer.create_comprehensive_report(tracker.get_history())
                else:
                    print("No data to visualize yet")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save data if any was collected
        logger.finalize_window_log()
        history = tracker.get_history()
        if len(history) > 0:
            print(f"\nSaving {len(history)} frames of tracking data...")
            
            # 1. Export raw data
            csv_file = logger.export_to_csv(history)
            json_file = logger.export_to_json(history)
            
            # 2. Generate summary
            report = logger.generate_summary_report(history)
            if report:
                logger.print_summary(report)
                logger.export_summary(report)

            # Generate signal processing analysis
            print("\n Generate signal processing analysis? (y/n): ", end='')
            response = input().strip().lower()
            if response == 'y':
                print("\n Generate signal processing analysis...")
                signal_report = logger.generate_signal_analysis_report(history)
                if signal_report:
                    logger.print_signal_analysis(signal_report)
                    logger.export_signal_analysis(signal_report)
            
            # Ask if user wants visualizations
            print("\nGenerate visualizations? (y/n): ", end='')
            response = input().strip().lower()
            if response == 'y':
                print("\nCreating visualizations...")
                visualizer.create_comprehensive_report(history)
            
            print(f"\nAll data saved to '{logger.output_dir}/' directory")
        else:
            print("\nNo tracking data collected")
        
        print("\nEye tracking session ended")
        print("="*60)

if __name__ == "__main__":
    # Required for proper imports
    import pandas as pd
    main()