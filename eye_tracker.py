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
        self.h_min, self.h_max = float('inf'), float('-inf')
        self.v_min, self.v_max = float('inf'), float('-inf')
        self.is_minmax_calibrated = False
        self.minmax_samples = []
        self.calibration_phase = 0  # 0=vertikal awal, 1=4pojok
        self.CORNERS = ["KIRI ATAS", "KANAN ATAS", "KANAN BAWAH", "KIRI BAWAH"]
        self.corner_index = 0
        self.corner_frames = 0
        self.CORNER_FRAMES_NEEDED = 60
        
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

        # Landmark statis untuk referensi vertikal
        # Titik-titik ini tidak bergerak saat bola mata bergerak
        self.NOSE_TIP = 1          # ujung hidung
        self.FOREHEAD = 10         # dahi atas
        self.FACE_TOP = 10
        self.FACE_BOTTOM = 152     # dagu
        
        # Conversion factors
        self.DEGREES_PER_UNIT_H = 40  # Horizontal FOV in degrees
        self.DEGREES_PER_UNIT_V = 30  # Vertical FOV in degrees

        # Gaze point visualization
        self.gaze_trail = []
        self.max_trail_length = 15

        # Data storage
        self.gaze_history = []
        self.blink_count = 0

    def calibrate_corners(self, h_ratio, v_ratio, frame):
        """Fase kalibrasi 4 pojok layar"""
        frame_h, frame_w = frame.shape[:2]

        if self.corner_index >= len(self.CORNERS):
            if not self.is_minmax_calibrated:
                h_vals = [s[0] for s in self.minmax_samples]
                v_vals = [s[1] for s in self.minmax_samples]
                self.h_min = min(h_vals)
                self.h_max = max(h_vals)
                self.v_min = min(v_vals)
                self.v_max = max(v_vals)

                # Padding 20% untuk kompensasi under-recording
                h_range = self.h_max - self.h_min
                v_range = self.v_max - self.v_min
                self.h_min -= h_range * 0.1
                self.h_max += h_range * 0.1
                self.v_min -= v_range * 0.1
                self.v_max += v_range * 0.1

                self.is_minmax_calibrated = True
                print(f"✅ Kalibrasi selesai!")
                print(f"   h=[{self.h_min:.3f}, {self.h_max:.3f}]")
                print(f"   v=[{self.v_min:.3f}, {self.v_max:.3f}]")

            cv2.putText(frame, "KALIBRASI SELESAI!", (frame_w//2 - 150, frame_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            return

        # Instruksi pojok aktif
        corner_name = self.CORNERS[self.corner_index]
        progress = self.corner_frames / self.CORNER_FRAMES_NEEDED

        cv2.putText(frame, f"Lihat pojok: {corner_name}", (frame_w//2 - 150, frame_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"({self.corner_index + 1}/4)", (frame_w//2 - 20, frame_h//2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Progress bar
        cv2.rectangle(frame, (frame_w//2 - 100, frame_h//2 + 60),
                    (frame_w//2 + 100, frame_h//2 + 80),
                    (50, 50, 50), -1)
        cv2.rectangle(frame, (frame_w//2 - 100, frame_h//2 + 60),
                    (frame_w//2 - 100 + int(200 * progress), frame_h//2 + 80),
                    (0, 255, 0), -1)

        # Rekam sample per pojok saja (bukan akumulasi semua)
        self.minmax_samples.append((h_ratio, v_ratio))
        self.corner_frames += 1

        if self.corner_frames >= self.CORNER_FRAMES_NEEDED:
            # Ambil hanya sample dari pojok ini
            corner_samples = self.minmax_samples[-self.CORNER_FRAMES_NEEDED:]
            h_vals = [s[0] for s in corner_samples]
            v_vals = [s[1] for s in corner_samples]

            # Update global min/max dengan percentile untuk hindari outlier
            corner_h_min = np.percentile(h_vals, 10)
            corner_h_max = np.percentile(h_vals, 90)
            corner_v_min = np.percentile(v_vals, 10)
            corner_v_max = np.percentile(v_vals, 90)

            self.h_min = min(self.h_min, corner_h_min)
            self.h_max = max(self.h_max, corner_h_max)
            self.v_min = min(self.v_min, corner_v_min)
            self.v_max = max(self.v_max, corner_v_max)

            print(f"✅ {corner_name} selesai")
            print(f"   pojok h=[{corner_h_min:.3f}, {corner_h_max:.3f}] v=[{corner_v_min:.3f}, {corner_v_max:.3f}]")
            print(f"   global h=[{self.h_min:.3f}, {self.h_max:.3f}] v=[{self.v_min:.3f}, {self.v_max:.3f}]")

            self.corner_index += 1
            self.corner_frames = 0
        
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

        # print(f"EAR={ear:.3f} | v1={vertical1:.1f} v2={vertical2:.1f} h={horizontal:.1f}") # Debug EAR values
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
        # Iris center
        iris_coords = []
        for idx in iris_points:
            iris_coords.append((
                landmarks[idx].x * frame_w,
                landmarks[idx].y * frame_h
            ))
        iris_center = np.mean(iris_coords, axis=0)

        # Horizontal
        eye_coords = []
        for idx in eye_points:
            eye_coords.append((
                landmarks[idx].x * frame_w,
                landmarks[idx].y * frame_h
            ))
        eye_left = min(p[0] for p in eye_coords)
        eye_right = max(p[0] for p in eye_coords)
        eye_width = eye_right - eye_left

        horizontal_ratio = 0
        if eye_width > 0:
            horizontal_ratio = ((iris_center[0] - eye_left) / eye_width - 0.5) * 2

        # Vertical — referensi sudut mata (karna sudut mata statis, tidak ikut gerak)
        # LEFT eye corners: 362 (kiri) dan 263 (kanan)
        # RIGHT eye corners: 33 (kiri) dan 133 (kanan)
        # Y dari sudut mata = posisi vertikal "netral" kelopak
        if eye_points == self.LEFT_EYE:
            corner_left_y = landmarks[362].y * frame_h
            corner_right_y = landmarks[263].y * frame_h
        else:
            corner_left_y = landmarks[33].y * frame_h
            corner_right_y = landmarks[133].y * frame_h

        #  Mean y kedua sudut = garis tengah horizontal mata
        eye_center_y = (corner_left_y + corner_right_y) / 2

        vertical_ratio = 0
        if eye_width > 0:
            # Iris relatif terhadap garis tengah sudut mata
            # Dinormalisasi raw pixel differenece
            vertical_ratio = (iris_center[1] - eye_center_y) / (eye_width * 0.3)

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
        
    def gaze_to_screen_point(self, gaze_h_ratio, gaze_v_ratio, frame_w, frame_h):
        if not self.is_minmax_calibrated:
            # Fallback ke scaling biasa sambil nunggu kalibrasi
            SCALE_H = 5.5
            SCALE_V = 5.0
            gaze_h_scaled = np.clip(gaze_h_ratio * SCALE_H, -1, 1)
            gaze_v_scaled = np.clip(gaze_v_ratio * SCALE_V, -1, 1)
        else:
            # Normalisasi berdasarkan range aktual hasil kalibrasi
            h_range = self.h_max - self.h_min
            v_range = self.v_max - self.v_min

            gaze_h_scaled = np.clip(
                (gaze_h_ratio - self.h_min) / h_range * 2 - 1, -1, 1
            ) if h_range > 0 else 0

            gaze_v_scaled = np.clip(
                (gaze_v_ratio - self.v_min) / v_range * 2 - 1, -1, 1
            ) if v_range > 0 else 0

        screen_x = int((gaze_h_scaled + 1) / 2 * frame_w)
        screen_y = int((gaze_v_scaled + 1) / 2 * frame_h)

        return np.clip(screen_x, 0, frame_w-1), np.clip(screen_y, 0, frame_h-1)

    def debug_ear_points(self, frame, landmarks, frame_w, frame_h):
        """Gambar titik EAR dengan label p1-p6 untuk verifikasi"""
        
        ear_indices = {
            "LEFT":  self.LEFT_EYE_EAR,   # [362, 385, 387, 263, 373, 380]
            "RIGHT": self.RIGHT_EYE_EAR   # [33, 160, 158, 133, 153, 144]
        }
        
        colors = {
            "LEFT":  (0, 255, 0),    # hijau
            "RIGHT": (255, 100, 0)   # biru
        }
        
        for eye_name, indices in ear_indices.items():
            for i, idx in enumerate(indices):
                x = int(landmarks[idx].x * frame_w)
                y = int(landmarks[idx].y * frame_h)
                
                # Gambar titik
                cv2.circle(frame, (x, y), 4, colors[eye_name], -1)
                
                # Label p1, p2, ... p6
                label = f"p{i+1}({idx})"
                cv2.putText(frame, label, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            colors[eye_name], 1)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame and return tracking data"""
        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'face_detected': False,
            'gaze_direction': None,
            'gaze_h_ratio': None,
            'gaze_v_ratio': None,
            'gaze_h_degrees': None,
            'gaze_v_degrees': None,
            'left_ear': None,
            'right_ear': None,
            'blink_detected': False,
            'blink_count': self.blink_count
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            data['face_detected'] = True
            
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Blink detection
            left_ear = self.compute_ear(self.LEFT_EYE_EAR, landmarks, frame_w, frame_h)
            right_ear = self.compute_ear(self.RIGHT_EYE_EAR, landmarks, frame_w, frame_h)
            avg_ear = (left_ear + right_ear) / 2.0
            data['left_ear'] = left_ear
            data['right_ear'] = right_ear

            current_time = time.time()
            if avg_ear < self.EAR_THRESHOLD:
                self.blink_frame_counter += 1
            else:
                if self.blink_frame_counter >= self.EAR_CONSEC_FRAMES:
                    if current_time - self.last_blink_time > 0.15:
                        self.blink_count += 1
                        data['blink_detected'] = True
                        data['blink_count'] = self.blink_count
                        self.last_blink_time = current_time
                self.blink_frame_counter = 0

            # Get iris positions
            left_h, left_v, left_center = self.get_iris_position(
                self.LEFT_IRIS, self.LEFT_EYE, landmarks, frame_w, frame_h,
                self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM
            )
            right_h, right_v, right_center = self.get_iris_position(
                self.RIGHT_IRIS, self.RIGHT_EYE, landmarks, frame_w, frame_h,
                self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM
            )

            print(f"left_v={left_v:.3f} | right_v={right_v:.3f}") #debug vertical ratios

            # Combine both eyes
            avg_h = (left_h + right_h) / 2
            weight_left = np.clip(0.5 - avg_h * 1.5, 0.2, 0.8)
            weight_right = 1.0 - weight_left
            avg_v = left_v * weight_left + right_v * weight_right

            # Calibration
            if not self.is_calibrated:
                progress = len(self.calibration_frames) / 60
                cv2.putText(frame, "Lihat ke TENGAH layar", (frame_w//2 - 150, frame_h//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.rectangle(frame, (frame_w//2 - 100, frame_h//2 + 10),(frame_w//2 - 100 + int(200*progress), frame_h//2 + 30),(0, 255, 255), -1)
                
                self.calibration_frames.append(avg_v)
                if len(self.calibration_frames) >= 60:
                    self.vertical_offset = np.mean(self.calibration_frames)
                    self.is_calibrated = True
                    print("\n✅ Calibration complete!")

            corrected_v = avg_v - self.vertical_offset
            corrected_v *= 2.5
            corrected_v = np.clip(corrected_v, -1, 1)
            
            if self.is_calibrated and not self.is_minmax_calibrated:
                self.calibrate_corners(avg_h, corrected_v, frame)
            
            print(f"Kalibrasi: {self.is_calibrated} | MinMax: {self.is_minmax_calibrated} | h_min: {self.h_min:.3f} | h_max: {self.h_max:.3f} | v_min: {self.v_min:.3f} | v_max: {self.v_max:.3f}") # Debug calibration status
            
            data['gaze_h_ratio'] = avg_h
            
            print(f"H: {avg_h:.3f} | V: {corrected_v:.3f}") # Debug gaze ratios
            data['gaze_v_ratio'] = corrected_v
            data['gaze_h_degrees'] = self.convert_to_degrees(avg_h, 'horizontal')
            data['gaze_v_degrees'] = self.convert_to_degrees(corrected_v, 'vertical')
            
            gaze_direction = self.determine_gaze_direction(avg_h, corrected_v)
            data['gaze_direction'] = gaze_direction
            
            # Draw iris centers
            cv2.circle(frame, (int(left_center[0]), int(left_center[1])), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_center[0]), int(right_center[1])), 3, (0, 255, 0), -1)

            # frame = self.debug_ear_points(frame, landmarks, frame_w, frame_h) # Debug EAR points
            
            # Draw gaze point indicator with trail
            if self.is_calibrated:
                gaze_x, gaze_y = self.gaze_to_screen_point(avg_h, corrected_v, frame_w, frame_h)
                
                # Add to trail
                self.gaze_trail.append((gaze_x, gaze_y))
                if len(self.gaze_trail) > self.max_trail_length:
                    self.gaze_trail.pop(0)
                
                # Draw trail with fading effect
                for i, (tx, ty) in enumerate(self.gaze_trail):
                    alpha = (i + 1) / len(self.gaze_trail)
                    radius = int(5 + alpha * 15)
                    color_intensity = int(255 * alpha)
                    
                    overlay = frame.copy()
                    cv2.circle(overlay, (tx, ty), radius, (0, color_intensity, 255), -1)
                    cv2.addWeighted(overlay, 0.3 * alpha, frame, 1 - 0.3 * alpha, 0, frame)
                
                # Draw current gaze point (bright yellow)
                cv2.circle(frame, (gaze_x, gaze_y), 12, (0, 255, 255), -1)
                cv2.circle(frame, (gaze_x, gaze_y), 12, (255, 255, 255), 2)
                
                # Crosshair
                cv2.line(frame, (gaze_x - 8, gaze_y), (gaze_x + 8, gaze_y), 
                        (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(frame, (gaze_x, gaze_y - 8), (gaze_x, gaze_y + 8), 
                        (255, 255, 255), 1, cv2.LINE_AA)
            
            # Show calibration progress
            if not self.is_calibrated:
                progress = len(self.calibration_frames) / 60
                cv2.rectangle(frame, (frame_w//2 - 100, 10), 
                            (frame_w//2 - 100 + int(200*progress), 30), 
                            (0, 255, 255), -1)
                cv2.putText(frame, "CALIBRATING...", (frame_w//2 - 80, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw info text (existing)
            cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"H: {avg_h:.2f} V: {corrected_v:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            
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