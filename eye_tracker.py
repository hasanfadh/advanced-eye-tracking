import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time

class EyeTracker:
    """
    Advanced Eye Tracker using MediaPipe Face Mesh
    Tracks gaze direction, blink detection, and eye movements
    
    IMPROVEMENTS:
    - Better blink detection with adjustable threshold
    - Degree conversion helper function
    - Enhanced calibration UI
    """
    
    def __init__(self):
        self.calibration_frames = []
        self.is_calibrated = False
        self.vertical_offset = 0.0
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmark indices (MediaPipe Face Mesh)
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # Landmark statis kelopak mata (UNTUK VERTICAL GAZE)
        self.LEFT_EYE_TOP = 386
        self.LEFT_EYE_BOTTOM = 374
        self.RIGHT_EYE_TOP = 159
        self.RIGHT_EYE_BOTTOM = 145
        
        # Landmark khusus untuk EAR (blink)
        self.LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

        # IMPROVED: Adjusted blink detection parameters
        self.EAR_THRESHOLD = 0.21  # Increased from 0.15 (more standard)
        self.EAR_CONSEC_FRAMES = 1  # Decreased from 2 (catch faster blinks)
        self.blink_frame_counter = 0
        self.last_blink_time = 0
        
        # Conversion factors
        self.DEGREES_PER_UNIT_H = 40  # Horizontal FOV in degrees
        self.DEGREES_PER_UNIT_V = 30  # Vertical FOV in degrees

        # Data storage
        self.gaze_history = []
        self.blink_count = 0
        
    def compute_ear(self, eye_idx, landmarks, frame_w, frame_h):
        """Calculate Eye Aspect Ratio for blink detection"""
        p = []
        for i in eye_idx:
            p.append(np.array([
                landmarks[i].x * frame_w,
                landmarks[i].y * frame_h
            ]))

        # EAR formula 6 points
        vertical1 = np.linalg.norm(p[1] - p[5])
        vertical2 = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])

        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def convert_to_degrees(self, normalized_value, axis='horizontal'):
        """
        Convert normalized gaze ratio to degrees
        
        Args:
            normalized_value: Gaze ratio (-1 to 1)
            axis: 'horizontal' or 'vertical'
            
        Returns:
            float: Gaze angle in degrees
            
        Helper function for degree conversion
        """
        if axis == 'horizontal':
            return normalized_value * (self.DEGREES_PER_UNIT_H / 2)
        else:
            return normalized_value * (self.DEGREES_PER_UNIT_V / 2)
    
    def get_iris_position(self, iris_points, eye_points, landmarks, frame_w, frame_h, eye_top_idx, eye_bottom_idx):
        """Calculate iris position relative to eye boundaries"""
        # Iris center
        iris_coords = []
        for idx in iris_points:
            iris_coords.append((
                landmarks[idx].x * frame_w,
                landmarks[idx].y * frame_h
            ))
        iris_center = np.mean(iris_coords, axis=0) # (x, y)

        # Horizontal bounds (tetap pakai semua titik)
        eye_coords = []
        for idx in eye_points:
            eye_coords.append((
                landmarks[idx].x * frame_w,
                landmarks[idx].y * frame_h
            ))

        eye_left = min(p[0] for p in eye_coords)
        eye_right = max(p[0] for p in eye_coords)

        # Vertical bounds pakai landmark statis
        eye_top = landmarks[eye_top_idx].y * frame_h
        eye_bottom = landmarks[eye_bottom_idx].y * frame_h

        eye_width = eye_right - eye_left
        eye_height = eye_bottom - eye_top

        horizontal_ratio = 0
        vertical_ratio = 0

        if eye_width > 0:
            horizontal_ratio = ((iris_center[0] - eye_left) / eye_width - 0.5) * 2

        if eye_height > 0:
            vertical_ratio = ((iris_center[1] - eye_top) / eye_height - 0.5) * 2

        return horizontal_ratio, vertical_ratio, iris_center

    
    def determine_gaze_direction(self, h_ratio, v_ratio):
        """Determine gaze direction from iris position"""
        H_THRESH = 0.15
        V_THRESH = 0.35
        horizontal = "center"
        vertical = "center"
        
        # Determine horizontal direction
        if h_ratio < -H_THRESH:
            horizontal = "left"
        elif h_ratio > H_THRESH:
            horizontal = "right"
        
        # Determine vertical direction
        if v_ratio < -V_THRESH:
            vertical = "up"
        elif v_ratio > V_THRESH:
            vertical = "down"
        
        # Combine directions
        if horizontal == "center" and vertical == "center":
            return "center"
        elif horizontal == "center":
            return vertical # "up" or "down"
        elif vertical == "center":
            return horizontal # "left" or "right"
        else:
            return f"{vertical}-{horizontal}" # e.g., "up-left"
    
    def process_frame(self, frame):
        """Process a single frame and return tracking data"""
        frame_h, frame_w = frame.shape[:2]
        # 1. Face mesh detection
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect face and landmarks
        results = self.face_mesh.process(rgb_frame)
        
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'face_detected': False,
            'gaze_direction': None,
            'gaze_h_ratio': None,
            'gaze_v_ratio': None,
            'gaze_h_degrees': None,  # Degrees
            'gaze_v_degrees': None,  # Degrees
            'left_ear': None,
            'right_ear': None,
            'blink_detected': False,
            'blink_count': self.blink_count
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark # 478 landmarks
            data['face_detected'] = True
            
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # 2. Blink detection
            # Compute EAR (Eye Aspect Ratio)
            left_ear = self.compute_ear(self.LEFT_EYE_EAR, landmarks, frame_w, frame_h)
            right_ear = self.compute_ear(self.RIGHT_EYE_EAR, landmarks, frame_w, frame_h)
            avg_ear = (left_ear + right_ear) / 2.0

            data['left_ear'] = left_ear
            data['right_ear'] = right_ear

            # Blink detection with debouncing
            current_time = time.time()
            if avg_ear < self.EAR_THRESHOLD:
                self.blink_frame_counter += 1
            else:
                if self.blink_frame_counter >= self.EAR_CONSEC_FRAMES:
                    # Check debouncing (prevent double-counting)
                    if current_time - self.last_blink_time > 0.15:  # 150ms debounce
                        self.blink_count += 1
                        data['blink_detected'] = True
                        data['blink_count'] = self.blink_count
                        self.last_blink_time = current_time
                self.blink_frame_counter = 0

            # 3. Get iris positions
            left_h, left_v, left_center = self.get_iris_position(
                self.LEFT_IRIS,
                self.LEFT_EYE,
                landmarks,
                frame_w,
                frame_h,
                self.LEFT_EYE_TOP,
                self.LEFT_EYE_BOTTOM
            )
            right_h, right_v, right_center = self.get_iris_position(
                self.RIGHT_IRIS,
                self.RIGHT_EYE,
                landmarks,
                frame_w,
                frame_h,
                self.RIGHT_EYE_TOP,
                self.RIGHT_EYE_BOTTOM
            )

            # 4. Combine both eyes
            avg_h = (left_h + right_h) / 2
            avg_v = (left_v + right_v) / 2

            # IMPROVED: Amplify vertical movement sensitivity (better for small vertical shifts)
            avg_v *= 1.5
            avg_v = np.clip(avg_v, -1, 1) # Keep within bounds -1 to 1

            # 5. Calibration
            if not self.is_calibrated:
                self.calibration_frames.append(avg_v)
                if len(self.calibration_frames) >= 60:  # ±2 detik at 30 FPS
                    self.vertical_offset = np.mean(self.calibration_frames)
                    self.is_calibrated = True
                    print("\n Calibration complete!")

            # Show calibration status on screen
            if not self.is_calibrated:
                progress = len(self.calibration_frames) / 60
                cv2.rectangle(frame, (frame_w//2 - 100, 10), 
                            (frame_w//2 - 100 + int(200*progress), 30), 
                            (0, 255, 255), -1)
                cv2.putText(frame, "CALIBRATING...", (frame_w//2 - 80, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            data['gaze_h_ratio'] = avg_h
            corrected_v = avg_v - self.vertical_offset
            data['gaze_v_ratio'] = corrected_v
            
            # 6. Convert to degrees
            data['gaze_h_degrees'] = self.convert_to_degrees(avg_h, 'horizontal')
            data['gaze_v_degrees'] = self.convert_to_degrees(corrected_v, 'vertical')
            
            # 7. Determine gaze direction
            gaze_direction = self.determine_gaze_direction(avg_h, corrected_v)
            data['gaze_direction'] = gaze_direction

            # 8. Store Data: was done gradually
                        
            # Draw iris centers
            cv2.circle(frame, (int(left_center[0]), int(left_center[1])), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_center[0]), int(right_center[1])), 3, (0, 255, 0), -1)
            
            # Draw gaze direction text
            cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"H: {avg_h:.2f} V: {corrected_v:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            
            # Draw EAR
            color = (0, 0, 255) if avg_ear < self.EAR_THRESHOLD else (100, 255, 100)
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame, data
    
    def add_to_history(self, data):
        """Add tracking data to history"""
        self.gaze_history.append(data)
    
    def get_history(self):
        """Get tracking history"""
        return self.gaze_history
    
    def clear_history(self):
        """Clear tracking history"""
        self.gaze_history = []
        self.blink_count = 0
    
    def __del__(self):
        """Cleanup"""
        self.face_mesh.close()