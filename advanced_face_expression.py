import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

@dataclass
class ExpressionConfig:
    """Configuration for expression detection"""
    max_num_faces: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    expression_buffer_size: int = 7
    calibration_frames: int = 30
    
    # Thresholds (will be calibrated)
    eye_aspect_ratio_threshold: float = 0.21
    mouth_aspect_ratio_threshold: float = 0.35
    smile_ratio_threshold: float = 0.15
    eyebrow_raise_threshold: float = 0.02

class AdvancedFaceExpressionDetector:
    def __init__(self, config: ExpressionConfig = None):
        self.config = config or ExpressionConfig()
        
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        
        # Facial landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [300, 293, 334, 296, 336]
        self.MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        self.LIPS = [13, 14]  # Upper and lower lip center
        
        # Calibration data
        self.calibration_data = {
            'eye_ratio': deque(maxlen=self.config.calibration_frames),
            'mouth_ratio': deque(maxlen=self.config.calibration_frames),
            'smile_ratio': deque(maxlen=self.config.calibration_frames),
            'eyebrow_pos': deque(maxlen=self.config.calibration_frames)
        }
        self.is_calibrated = False
        self.baseline_values = {}
        
        # Expression history for smoothing
        self.expression_history = deque(maxlen=self.config.expression_buffer_size)
        
        # Blink detection
        self.blink_counter = 0
        self.blink_detected = False
        
    def euclidean_distance(self, p1, p2) -> float:
        """Calculate Euclidean distance between two points"""
        return math.hypot(p2.x - p1.x, p2.y - p1.y)
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        Based on the paper: "Real-Time Eye Blink Detection using Facial Landmarks"
        """
        # Vertical eye distances
        v1 = self.euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        v2 = self.euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        
        # Horizontal eye distance
        h = self.euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def calculate_mouth_aspect_ratio(self, landmarks) -> float:
        """Calculate MAR (Mouth Aspect Ratio) for mouth opening detection"""
        # Vertical distances
        v1 = self.euclidean_distance(landmarks[13], landmarks[14])  # Center
        v2 = self.euclidean_distance(landmarks[78], landmarks[308])  # Sides
        
        # Horizontal distance
        h = self.euclidean_distance(landmarks[61], landmarks[291])
        
        mar = (v1 + v2 * 0.5) / (h + 1e-6)
        return mar
    
    def calculate_smile_ratio(self, landmarks) -> float:
        """Detect smile by measuring mouth corner elevation"""
        # Mouth corners
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        
        # Upper lip center
        upper_lip = landmarks[13]
        
        # Calculate vertical positions
        corner_y = (left_corner.y + right_corner.y) / 2
        lip_y = upper_lip.y
        
        # Smile ratio (corners higher than upper lip = smile)
        ratio = lip_y - corner_y
        return ratio
    
    def calculate_eyebrow_raise(self, landmarks) -> float:
        """Detect eyebrow raise by measuring distance from eyebrow to eye"""
        # Left eyebrow center to left eye center
        left_brow_center = landmarks[70]
        left_eye_center = landmarks[159]
        left_dist = self.euclidean_distance(left_brow_center, left_eye_center)
        
        # Right eyebrow center to right eye center
        right_brow_center = landmarks[300]
        right_eye_center = landmarks[386]
        right_dist = self.euclidean_distance(right_brow_center, right_eye_center)
        
        avg_dist = (left_dist + right_dist) / 2
        return avg_dist
    
    def calculate_head_pose(self, landmarks, img_shape) -> Tuple[float, float, float]:
        """
        Estimate head pose (yaw, pitch, roll) using PnP algorithm
        """
        h, w = img_shape[:2]
        
        # 3D model points (generic human face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # 2D image points from landmarks
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),      # Nose tip
            (landmarks[152].x * w, landmarks[152].y * h),  # Chin
            (landmarks[33].x * w, landmarks[33].y * h),    # Left eye left corner
            (landmarks[263].x * w, landmarks[263].y * h),  # Right eye right corner
            (landmarks[61].x * w, landmarks[61].y * h),    # Left mouth corner
            (landmarks[291].x * w, landmarks[291].y * h)   # Right mouth corner
        ], dtype=np.float64)
        
        # Camera internals
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Convert rotation vector to euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = euler_angles.flatten()[:3]
        return pitch, yaw, roll
    
    def calibrate(self, landmarks):
        """Collect baseline measurements for adaptive thresholding"""
        left_ear = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2
        
        mar = self.calculate_mouth_aspect_ratio(landmarks)
        smile = self.calculate_smile_ratio(landmarks)
        eyebrow = self.calculate_eyebrow_raise(landmarks)
        
        self.calibration_data['eye_ratio'].append(avg_ear)
        self.calibration_data['mouth_ratio'].append(mar)
        self.calibration_data['smile_ratio'].append(smile)
        self.calibration_data['eyebrow_pos'].append(eyebrow)
        
        if len(self.calibration_data['eye_ratio']) >= self.config.calibration_frames:
            self.baseline_values = {
                'eye_ratio': np.mean(self.calibration_data['eye_ratio']),
                'mouth_ratio': np.mean(self.calibration_data['mouth_ratio']),
                'smile_ratio': np.mean(self.calibration_data['smile_ratio']),
                'eyebrow_pos': np.mean(self.calibration_data['eyebrow_pos'])
            }
            self.is_calibrated = True
    
    def detect_expression(self, landmarks, img_shape) -> Tuple[str, Dict[str, float]]:
        """
        Comprehensive expression detection with confidence scores
        """
        # Calculate all metrics
        left_ear = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2
        
        mar = self.calculate_mouth_aspect_ratio(landmarks)
        smile = self.calculate_smile_ratio(landmarks)
        eyebrow = self.calculate_eyebrow_raise(landmarks)
        pitch, yaw, roll = self.calculate_head_pose(landmarks, img_shape)
        
        metrics = {
            'ear': avg_ear,
            'mar': mar,
            'smile': smile,
            'eyebrow': eyebrow,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll
        }
        
        # Use adaptive thresholds if calibrated
        if self.is_calibrated:
            ear_thresh = self.baseline_values['eye_ratio'] * 0.7
            mar_thresh = self.baseline_values['mouth_ratio'] * 1.5
            smile_thresh = self.baseline_values['smile_ratio'] + 0.01
            eyebrow_thresh = self.baseline_values['eyebrow_pos'] * 1.15
        else:
            ear_thresh = self.config.eye_aspect_ratio_threshold
            mar_thresh = self.config.mouth_aspect_ratio_threshold
            smile_thresh = self.config.smile_ratio_threshold
            eyebrow_thresh = self.config.eyebrow_raise_threshold
        
        # Detect blinks
        if avg_ear < ear_thresh:
            if not self.blink_detected:
                self.blink_counter += 1
                self.blink_detected = True
        else:
            self.blink_detected = False
        
        # Expression priority logic
        expression = "Neutral"
        
        # Check for eye closure
        if avg_ear < ear_thresh:
            if left_ear < ear_thresh and right_ear < ear_thresh:
                expression = "Eyes Closed"
            elif left_ear < ear_thresh:
                expression = "Left Wink"
            elif right_ear < ear_thresh:
                expression = "Right Wink"
        
        # Check for mouth expressions
        elif mar > mar_thresh:
            if smile > smile_thresh:
                expression = "Surprised/Excited"
            else:
                expression = "Mouth Open"
        
        # Check for smile
        elif smile > smile_thresh:
            if eyebrow > eyebrow_thresh:
                expression = "Happy/Joyful"
            else:
                expression = "Smiling"
        
        # Check for eyebrow raise
        elif eyebrow > eyebrow_thresh:
            expression = "Surprised"
        
        # Check for frown (negative smile ratio)
        elif smile < -0.01:
            expression = "Sad/Frown"
        
        # Head pose based expressions
        if abs(yaw) > 20:
            expression += " (Looking Away)"
        
        return expression, metrics
    
    def smooth_expression(self, expression: str) -> str:
        """Apply temporal smoothing to reduce jitter"""
        self.expression_history.append(expression)
        
        if len(self.expression_history) < self.config.expression_buffer_size:
            return expression
        
        # Count occurrences
        expr_counts = {}
        for e in self.expression_history:
            expr_counts[e] = expr_counts.get(e, 0) + 1
        
        # Return most common expression
        most_common = max(expr_counts.items(), key=lambda x: x[1])
        threshold = self.config.expression_buffer_size * 0.5
        
        return most_common[0] if most_common[1] >= threshold else expression
    
    def draw_enhanced_display(self, img, landmarks, expression: str, metrics: Dict):
        """Draw enhanced visualization with metrics"""
        h, w, _ = img.shape
        
        # Draw face mesh
        self.mp_drawing.draw_landmarks(
            image=img,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # Draw contours
        self.mp_drawing.draw_landmarks(
            image=img,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        
        # Display panel
        panel_height = 180
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Expression text
        cv2.putText(img, f"Expression: {expression}", (10, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        
        # Calibration status
        status = "Calibrated" if self.is_calibrated else "Calibrating..."
        color = (0, 255, 0) if self.is_calibrated else (0, 255, 255)
        cv2.putText(img, f"Status: {status}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Metrics
        y_offset = 85
        cv2.putText(img, f"EAR: {metrics['ear']:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"MAR: {metrics['mar']:.3f}", (150, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_offset += 25
        cv2.putText(img, f"Smile: {metrics['smile']:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"Blinks: {self.blink_counter}", (150, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_offset += 25
        cv2.putText(img, f"Yaw: {metrics['yaw']:.1f}°", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"Pitch: {metrics['pitch']:.1f}°", (150, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_offset += 25
        cv2.putText(img, f"Roll: {metrics['roll']:.1f}°", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=" * 60)
        print("Advanced Face Expression Detection System")
        print("=" * 60)
        print("\nDetectable Expressions:")
        print("  • Neutral, Smiling, Happy/Joyful")
        print("  • Surprised, Sad/Frown")
        print("  • Eyes Closed, Left/Right Wink")
        print("  • Mouth Open, Surprised/Excited")
        print("\nFeatures:")
        print("  • Adaptive calibration (first 30 frames)")
        print("  • Head pose estimation")
        print("  • Blink counter")
        print("  • Real-time metrics display")
        print("\nPress 'q' to exit, 'r' to recalibrate")
        print("=" * 60)
        
        fps_history = deque(maxlen=30)
        prev_time = cv2.getTickCount()
        
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                continue
            
            # Calculate FPS
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 1.0 / time_diff if time_diff > 0 else 0
            fps_history.append(fps)
            prev_time = curr_time
            avg_fps = np.mean(fps_history)
            
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            
            # FPS display
            cv2.putText(img, f"FPS: {avg_fps:.1f}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Calibration phase
                    if not self.is_calibrated:
                        self.calibrate(landmarks)
                    
                    # Detect expression
                    expression, metrics = self.detect_expression(landmarks, img.shape)
                    smoothed_expression = self.smooth_expression(expression)
                    
                    # Draw visualization
                    self.draw_enhanced_display(img, face_landmarks, smoothed_expression, metrics)
            
            cv2.imshow('Advanced Face Expression Detection', img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset calibration
                self.is_calibrated = False
                self.calibration_data = {k: deque(maxlen=self.config.calibration_frames) 
                                        for k in self.calibration_data.keys()}
                print("Recalibrating...")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config = ExpressionConfig(
        calibration_frames=30,
        expression_buffer_size=7
    )
    detector = AdvancedFaceExpressionDetector(config)
    detector.run()