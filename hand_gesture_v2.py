import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

@dataclass
class GestureConfig:
    """Configuration for gesture detection"""
    max_num_hands: int = 2
    min_detection_confidence: float = 0.75
    min_tracking_confidence: float = 0.65
    gesture_buffer_size: int = 7  # Frames to smooth gestures
    angle_threshold: float = 155  # Degrees for finger straightness
    min_gesture_confidence: float = 0.65
    curl_threshold: float = 130  # Threshold for finger curl detection

class AdvancedHandGestureRecognizer:
    def __init__(self, config: GestureConfig = None):
        self.config = config or GestureConfig()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        
        # Landmark indices
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.pip_ids = [2, 6, 10, 14, 18]  # PIP joints
        self.mcp_ids = [1, 5, 9, 13, 17]   # MCP joints (knuckles)
        self.dip_ids = [3, 7, 11, 15, 19]  # DIP joints
        
        # Gesture history for temporal smoothing
        self.gesture_history = deque(maxlen=self.config.gesture_buffer_size)
        self.gesture_confidence_history = deque(maxlen=self.config.gesture_buffer_size)
        
    def calculate_angle(self, p1: List[float], p2: List[float], p3: List[float]) -> float:
        """Calculate angle between three points (in degrees)"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def euclidean_distance(self, p1: List[float], p2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    def get_finger_curl_state(self, landmarks: List, finger_idx: int) -> float:
        """
        Get finger curl ratio (0 = fully curled, 1 = fully extended)
        """
        if finger_idx == 0:  # Thumb - special case
            tip_to_wrist = self.euclidean_distance(
                [landmarks[4].x, landmarks[4].y],
                [landmarks[0].x, landmarks[0].y]
            )
            mcp_to_wrist = self.euclidean_distance(
                [landmarks[2].x, landmarks[2].y],
                [landmarks[0].x, landmarks[0].y]
            )
            return tip_to_wrist / (mcp_to_wrist + 1e-6)
        
        # For other fingers
        tip = self.tip_ids[finger_idx]
        pip = self.pip_ids[finger_idx]
        mcp = self.mcp_ids[finger_idx]
        
        # Distance from tip to MCP vs PIP to MCP
        tip_to_mcp = self.euclidean_distance(
            [landmarks[tip].x, landmarks[tip].y],
            [landmarks[mcp].x, landmarks[mcp].y]
        )
        pip_to_mcp = self.euclidean_distance(
            [landmarks[pip].x, landmarks[pip].y],
            [landmarks[mcp].x, landmarks[mcp].y]
        )
        
        return tip_to_mcp / (pip_to_mcp + 1e-6)
    
    def is_finger_extended(self, landmarks: List, finger_idx: int, handedness: str) -> Tuple[bool, float]:
        """
        Advanced finger extension detection using multiple metrics
        Returns: (is_extended, confidence)
        """
        if finger_idx == 0:  # Thumb
            return self._is_thumb_extended(landmarks, handedness)
        
        # Multi-metric approach for other fingers
        tip = finger_idx
        tip_id = self.tip_ids[tip]
        pip_id = self.pip_ids[tip]
        mcp_id = self.mcp_ids[tip]
        dip_id = self.dip_ids[tip]
        
        tip_point = [landmarks[tip_id].x, landmarks[tip_id].y]
        dip_point = [landmarks[dip_id].x, landmarks[dip_id].y]
        pip_point = [landmarks[pip_id].x, landmarks[pip_id].y]
        mcp_point = [landmarks[mcp_id].x, landmarks[mcp_id].y]
        
        # 1. Angle at PIP joint
        pip_angle = self.calculate_angle(tip_point, pip_point, mcp_point)
        
        # 2. Angle at DIP joint
        dip_angle = self.calculate_angle(tip_point, dip_point, pip_point)
        
        # 3. Tip position relative to PIP (Y-axis check)
        tip_above_pip = landmarks[tip_id].y < landmarks[pip_id].y
        
        # 4. Curl ratio
        curl_ratio = self.get_finger_curl_state(landmarks, finger_idx)
        
        # Combined decision
        angle_extended = pip_angle > self.config.angle_threshold and dip_angle > self.config.curl_threshold
        position_extended = tip_above_pip
        curl_extended = curl_ratio > 1.2
        
        # Confidence calculation
        angle_conf = min(pip_angle / 180.0, 1.0)
        curl_conf = min(curl_ratio / 2.0, 1.0)
        position_conf = 1.0 if tip_above_pip else 0.3
        
        confidence = (angle_conf * 0.5 + curl_conf * 0.3 + position_conf * 0.2)
        is_extended = (angle_extended and curl_extended) or (angle_extended and position_extended)
        
        return is_extended, confidence
    
    def _is_thumb_extended(self, landmarks: List, handedness: str) -> Tuple[bool, float]:
        """Enhanced thumb detection"""
        thumb_tip = [landmarks[4].x, landmarks[4].y]
        thumb_ip = [landmarks[3].x, landmarks[3].y]
        thumb_mcp = [landmarks[2].x, landmarks[2].y]
        thumb_cmc = [landmarks[1].x, landmarks[1].y]
        index_mcp = [landmarks[5].x, landmarks[5].y]
        wrist = [landmarks[0].x, landmarks[0].y]
        
        # Multiple angles for better accuracy
        angle1 = self.calculate_angle(thumb_tip, thumb_ip, thumb_mcp)
        angle2 = self.calculate_angle(thumb_tip, thumb_mcp, thumb_cmc)
        
        # Distance metrics
        tip_to_index = self.euclidean_distance(thumb_tip, index_mcp)
        tip_to_wrist = self.euclidean_distance(thumb_tip, wrist)
        mcp_to_wrist = self.euclidean_distance(thumb_mcp, wrist)
        
        # Handedness-specific X position check
        if handedness == "Right":
            x_extended = thumb_tip[0] < thumb_mcp[0] - 0.02
        else:
            x_extended = thumb_tip[0] > thumb_mcp[0] + 0.02
        
        # Combined metrics
        angle_check = angle1 > 140 or angle2 > 100
        distance_check = tip_to_index > 0.08 or (tip_to_wrist / mcp_to_wrist) > 1.4
        
        is_extended = (angle_check or distance_check) and x_extended
        confidence = min((angle1 / 180.0 + tip_to_index * 8 + (1 if x_extended else 0)) / 3, 1.0)
        
        return is_extended, max(confidence, 0.5 if is_extended else 0.3)
    
    def get_finger_states(self, landmarks: List, handedness: str) -> Tuple[List[int], List[float], float]:
        """
        Get detailed finger states with individual and average confidence
        Returns: (finger_states, individual_confidences, average_confidence)
        """
        fingers = []
        confidences = []
        
        for i in range(5):
            is_extended, confidence = self.is_finger_extended(landmarks, i, handedness)
            fingers.append(1 if is_extended else 0)
            confidences.append(confidence)
        
        avg_confidence = np.mean(confidences)
        return fingers, confidences, avg_confidence
    
    def calculate_palm_center(self, landmarks: List) -> Tuple[float, float]:
        """Calculate center of palm"""
        palm_indices = [0, 1, 5, 9, 13, 17]
        cx = np.mean([landmarks[i].x for i in palm_indices])
        cy = np.mean([landmarks[i].y for i in palm_indices])
        return cx, cy
    
    def get_palm_orientation(self, landmarks: List) -> float:
        """Calculate palm rotation angle"""
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
        vector = middle_mcp - wrist
        angle = np.arctan2(vector[1], vector[0])
        return np.degrees(angle)
    
    def detect_gesture(self, fingers: List[int], landmarks: List, handedness: str, 
                       finger_confidences: List[float]) -> Tuple[Optional[str], float]:
        """
        Comprehensive gesture detection with 30+ gestures
        Returns: (gesture_name, confidence)
        """
        gestures = []
        
        # Calculate useful metrics
        thumb_tip = [landmarks[4].x, landmarks[4].y]
        index_tip = [landmarks[8].x, landmarks[8].y]
        middle_tip = [landmarks[12].x, landmarks[12].y]
        ring_tip = [landmarks[16].x, landmarks[16].y]
        pinky_tip = [landmarks[20].x, landmarks[20].y]
        
        index_mcp = [landmarks[5].x, landmarks[5].y]
        middle_mcp = [landmarks[9].x, landmarks[9].y]
        wrist = [landmarks[0].x, landmarks[0].y]
        
        # Distance calculations
        thumb_index_dist = self.euclidean_distance(thumb_tip, index_tip)
        thumb_middle_dist = self.euclidean_distance(thumb_tip, middle_tip)
        thumb_ring_dist = self.euclidean_distance(thumb_tip, ring_tip)
        thumb_pinky_dist = self.euclidean_distance(thumb_tip, pinky_tip)
        index_middle_dist = self.euclidean_distance(index_tip, middle_tip)
        ring_pinky_dist = self.euclidean_distance(ring_tip, pinky_tip)
        index_pinky_dist = self.euclidean_distance(index_tip, pinky_tip)
        
        # Palm orientation
        palm_angle = self.get_palm_orientation(landmarks)
        
        # Finger sum for quick checks
        finger_sum = sum(fingers)
        
        # ==================== GESTURE DETECTION ====================
        
        # 1. FIST / ROCK
        if finger_sum == 0:
            curl_confidence = 1.0 - np.mean([self.get_finger_curl_state(landmarks, i) for i in range(1, 5)])
            gestures.append(("Fist", min(0.98, 0.85 + curl_confidence * 0.13)))
        
        # 2. OPEN PALM / PAPER / STOP
        elif finger_sum == 5:
            spread_score = (index_middle_dist + ring_pinky_dist + index_pinky_dist) / 3
            gestures.append(("Open Palm", min(0.98, 0.82 + spread_score * 0.16)))
        
        # 3. PEACE / VICTORY / SCISSORS
        elif fingers == [0, 1, 1, 0, 0]:
            spread = index_middle_dist
            v_angle = self.calculate_angle(index_tip, index_mcp, middle_tip)
            if spread > 0.05 and v_angle < 100:
                gestures.append(("Peace/Victory", min(0.96, 0.75 + spread * 3)))
        
        # 4. THUMBS UP
        elif fingers == [1, 0, 0, 0, 0]:
            if landmarks[4].y < landmarks[8].y - 0.05:
                fist_tightness = 1.0 - np.mean([self.get_finger_curl_state(landmarks, i) for i in range(1, 5)])
                gestures.append(("Thumbs Up", min(0.96, 0.80 + fist_tightness * 0.16)))
            elif landmarks[4].y > landmarks[8].y + 0.05:
                gestures.append(("Thumbs Down", 0.92))
        
        # 5. POINTING / ONE
        elif fingers == [0, 1, 0, 0, 0]:
            straightness = finger_confidences[1]
            gestures.append(("Pointing/One", min(0.95, 0.82 + straightness * 0.13)))
        
        # 6. TWO FINGERS
        elif fingers == [0, 1, 1, 0, 0] and index_middle_dist < 0.04:
            gestures.append(("Two", 0.93))
        
        # 7. THREE FINGERS
        elif fingers == [0, 1, 1, 1, 0]:
            gestures.append(("Three", 0.94))
        
        # 8. FOUR FINGERS
        elif fingers == [0, 1, 1, 1, 1]:
            gestures.append(("Four", 0.95))
        
        # 9. OK SIGN (Circle: Thumb + Index)
        elif thumb_index_dist < 0.04 and fingers[2] == 1 and fingers[3] == 1:
            circle_quality = 1.0 - (thumb_index_dist / 0.04)
            gestures.append(("OK Sign", min(0.94, 0.78 + circle_quality * 0.16)))
        
        # 10. PINCH (Thumb + Index close)
        elif fingers[0] == 1 and fingers[1] == 1 and thumb_index_dist < 0.035:
            gestures.append(("Pinch", min(0.90, 0.75 + (1 - thumb_index_dist / 0.035) * 0.15)))
        
        # 11. ROCK ON / HEAVY METAL / DEVIL HORNS
        elif fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]:
            spread = self.euclidean_distance(index_tip, pinky_tip)
            gestures.append(("Rock On", min(0.94, 0.80 + spread * 2)))
        
        # 12. SPIDER-MAN / I LOVE YOU
        elif fingers == [1, 1, 0, 0, 1]:
            spread = (thumb_index_dist + thumb_pinky_dist + index_pinky_dist) / 3
            gestures.append(("I Love You", min(0.93, 0.78 + spread * 2)))
        
        # 13. CALL ME / SHAKA / HANG LOOSE
        elif fingers == [1, 0, 0, 0, 1]:
            spread = thumb_pinky_dist
            if spread > 0.15:
                gestures.append(("Call Me/Shaka", min(0.94, 0.80 + spread)))
        
        # 14. MIDDLE FINGER
        elif fingers == [0, 0, 1, 0, 0]:
            gestures.append(("Middle Finger", 0.92))
        
        # 15. RING FINGER
        elif fingers == [0, 0, 0, 1, 0]:
            gestures.append(("Ring Finger", 0.90))
        
        # 16. PINKY UP
        elif fingers == [0, 0, 0, 0, 1]:
            gestures.append(("Pinky", 0.90))
        
        # 17. GUN / PISTOL
        elif fingers == [1, 1, 0, 0, 0]:
            angle = self.calculate_angle(thumb_tip, index_mcp, index_tip)
            if 80 < angle < 120:
                gestures.append(("Gun/Pistol", 0.89))
        
        # 18. FINGER HEART (Thumb + Index crossed)
        elif fingers[0] == 1 and fingers[1] == 1 and thumb_index_dist < 0.06:
            cross_check = abs(thumb_tip[0] - index_tip[0]) < 0.03
            if cross_check:
                gestures.append(("Finger Heart", 0.88))
        
        # 19. VULCAN SALUTE (Split between middle and ring)
        elif fingers == [0, 1, 1, 1, 1]:
            middle_ring_dist = self.euclidean_distance(middle_tip, ring_tip)
            index_middle_close = index_middle_dist < 0.04
            ring_pinky_close = ring_pinky_dist < 0.04
            if middle_ring_dist > 0.06 and index_middle_close and ring_pinky_close:
                gestures.append(("Vulcan Salute", 0.91))
        
        # 20. FINGER SNAP POSITION (Thumb + Middle touching)
        elif fingers[0] == 1 and fingers[2] == 1 and thumb_middle_dist < 0.04:
            gestures.append(("Snap Position", 0.87))
        
        # 21. THUMB + RING (Custom gesture)
        elif fingers == [1, 0, 0, 1, 0] and thumb_ring_dist < 0.04:
            gestures.append(("Thumb-Ring Touch", 0.86))
        
        # 22. THUMB + PINKY (Different from Call Me - closer)
        elif fingers == [1, 0, 0, 0, 1] and thumb_pinky_dist < 0.08:
            gestures.append(("Thumb-Pinky Close", 0.85))
        
        # 23. INDEX + MIDDLE CROSSED
        elif fingers == [0, 1, 1, 0, 0]:
            cross_metric = abs(index_tip[0] - middle_tip[0])
            if cross_metric < 0.02 and index_middle_dist < 0.03:
                gestures.append(("Fingers Crossed", 0.88))
        
        # 24. PRAYER / NAMASTE (requires two hands - placeholder for single hand)
        elif finger_sum == 5 and palm_angle > 45:
            gestures.append(("Prayer Gesture", 0.82))
        
        # 25. CLICK GESTURE (Thumb + Middle in snapping position)
        elif fingers[0] == 1 and fingers[2] == 1 and fingers[1] == 0:
            if thumb_middle_dist < 0.05:
                gestures.append(("Click Gesture", 0.87))
        
        # 26. NUMBER SIX (ASL: Thumb + Pinky extended)
        elif fingers == [1, 0, 0, 0, 1] and thumb_pinky_dist > 0.12:
            gestures.append(("Six (ASL)", 0.89))
        
        # 27. NUMBER SEVEN (ASL: Thumb + All fingers)
        elif finger_sum == 5:
            # This overlaps with Open Palm, so check palm angle
            if abs(palm_angle) < 30:
                gestures.append(("Seven (ASL)", 0.83))
        
        # 28. NUMBER EIGHT (ASL: Thumb + Middle + Ring + Pinky)
        elif fingers == [1, 0, 1, 1, 1]:
            gestures.append(("Eight (ASL)", 0.88))
        
        # 29. NUMBER NINE (ASL: Thumb + Index + Middle + Ring)
        elif fingers == [1, 1, 1, 1, 0]:
            gestures.append(("Nine (ASL)", 0.89))
        
        # 30. LIVE LONG AND PROSPER (Vulcan with thumb)
        elif fingers == [1, 1, 1, 1, 1]:
            middle_ring_gap = self.euclidean_distance(middle_tip, ring_tip)
            if middle_ring_gap > 0.07:
                gestures.append(("Live Long & Prosper", 0.90))
        
        # 31. KARATE CHOP (All fingers together, extended)
        elif finger_sum == 4 and fingers[0] == 0:
            fingers_together = index_middle_dist < 0.025 and ring_pinky_dist < 0.025
            if fingers_together:
                gestures.append(("Karate Chop", 0.86))
        
        # 32. FINGER WAG (Single index pointing up)
        elif fingers == [0, 1, 0, 0, 0]:
            if landmarks[8].y < landmarks[6].y - 0.1:
                gestures.append(("Finger Wag", 0.87))
        
        # 33. BECKONING (Index curved down)
        elif fingers == [0, 1, 0, 0, 0]:
            curl = self.get_finger_curl_state(landmarks, 1)
            if 1.0 < curl < 1.5:
                gestures.append(("Beckoning", 0.84))
        
        # Return best gesture
        if gestures:
            gestures.sort(key=lambda x: x[1], reverse=True)
            best_gesture = gestures[0]
            # Boost confidence if finger confidences are high
            avg_finger_conf = np.mean(finger_confidences)
            final_confidence = min(0.99, best_gesture[1] * 0.9 + avg_finger_conf * 0.1)
            return best_gesture[0], final_confidence
        
        return None, 0.0
    
    def smooth_gesture(self, current_gesture: Optional[str], current_confidence: float) -> Tuple[Optional[str], float]:
        """Enhanced temporal smoothing with confidence weighting"""
        self.gesture_history.append(current_gesture)
        self.gesture_confidence_history.append(current_confidence)
        
        if len(self.gesture_history) < self.config.gesture_buffer_size:
            return current_gesture, current_confidence
        
        # Weighted voting based on confidence
        gesture_scores = {}
        for gesture, conf in zip(self.gesture_history, self.gesture_confidence_history):
            if gesture is not None:
                gesture_scores[gesture] = gesture_scores.get(gesture, 0) + conf
        
        if not gesture_scores:
            return None, 0.0
        
        # Get best gesture
        best_gesture = max(gesture_scores.items(), key=lambda x: x[1])
        avg_confidence = best_gesture[1] / len(self.gesture_history)
        
        # Require minimum presence in history
        gesture_count = sum(1 for g in self.gesture_history if g == best_gesture[0])
        threshold = self.config.gesture_buffer_size * 0.5
        
        if gesture_count >= threshold:
            return best_gesture[0], min(avg_confidence, 0.98)
        else:
            return current_gesture, current_confidence
    
    def draw_enhanced_landmarks(self, img, hand_landmarks, handedness: str, 
                                gesture: str, confidence: float, fingers: List[int]):
        """Enhanced visualization with detailed information"""
        h, w, _ = img.shape
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Get wrist position
        wrist = hand_landmarks.landmark[0]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        
        # Display gesture with confidence
        if gesture:
            # Confidence color (green to yellow based on confidence)
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            
            text = f"{gesture}"
            conf_text = f"{confidence:.0%}"
            
            # Background for gesture name
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (wx - 10, wy - text_h - 40), 
                         (wx + text_w + 10, wy - 25), (0, 0, 0), -1)
            cv2.putText(img, text, (wx, wy - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Confidence bar
            bar_width = 100
            bar_height = 8
            filled_width = int(bar_width * confidence)
            cv2.rectangle(img, (wx, wy - 18), (wx + bar_width, wy - 18 + bar_height), (50, 50, 50), -1)
            cv2.rectangle(img, (wx, wy - 18), (wx + filled_width, wy - 18 + bar_height), color, -1)
            cv2.putText(img, conf_text, (wx + bar_width + 5, wy - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display handedness and finger states
        cv2.putText(img, f"{handedness}", (wx, wy + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Finger state indicators
        finger_names = ['T', 'I', 'M', 'R', 'P']
        for i, (name, state) in enumerate(zip(finger_names, fingers)):
            color = (0, 255, 0) if state == 1 else (100, 100, 100)
            cv2.circle(img, (wx + i * 20, wy + 45), 6, color, -1)
            cv2.putText(img, name, (wx + i * 20 - 4, wy + 49),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("=" * 70)
        print(" " * 15 + "ADVANCED HAND GESTURE RECOGNITION")
        print("=" * 70)
        print("\n📋 SUPPORTED GESTURES (33+ gestures):")
        print("\n  Basic Numbers & Counting:")
        print("    • Fist, One, Two, Three, Four, Open Palm")
        print("    • ASL Numbers: Six, Seven, Eight, Nine")
        print("\n  Common Gestures:")
        print("    • Thumbs Up/Down, Pointing, Peace/Victory")
        print("    • OK Sign, Pinch, Gun/Pistol")
        print("\n  Special Signs:")
        print("    • Rock On, I Love You, Call Me/Shaka")
        print("    • Vulcan Salute, Live Long & Prosper")
        print("    • Finger Heart, Fingers Crossed")
        print("\n  Advanced Gestures:")
        print("    • Middle Finger, Snap Position, Click Gesture")
        print("    • Karate Chop, Beckoning, Prayer Gesture")
        print("    • Thumb-Ring Touch, and more...")
        print("\n🎯 Features:")
        print("    • Multi-angle finger detection")
        print("    • Temporal smoothing for stability")
        print("    • Confidence scoring for each gesture")
        print("    • Real-time finger state visualization")
        print("\n⌨️  Press 'q' to exit")
        print("=" * 70 + "\n")
        
        fps_history = deque(maxlen=30)
        prev_time = cv2.getTickCount()
        
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Calculate FPS
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 1.0 / time_diff if time_diff > 0 else 0
            fps_history.append(fps)
            prev_time = curr_time
            avg_fps = np.mean(fps_history)
            
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # Get image dimensions
            h, w, _ = img.shape
            
            # Draw FPS and info panel
            cv2.rectangle(img, (0, 0), (250, 60), (0, 0, 0), -1)
            cv2.putText(img, f"FPS: {avg_fps:.1f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Hands: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, 
                                                      results.multi_handedness):
                    # Get handedness
                    handedness = hand_info.classification[0].label
                    
                    # Detect finger states with confidences
                    fingers, finger_confidences, avg_finger_conf = self.get_finger_states(
                        hand_landmarks.landmark, handedness
                    )
                    
                    # Detect gesture
                    gesture, gesture_confidence = self.detect_gesture(
                        fingers, hand_landmarks.landmark, handedness, finger_confidences
                    )
                    
                    # Apply temporal smoothing
                    smoothed_gesture, smoothed_confidence = self.smooth_gesture(
                        gesture, gesture_confidence
                    )
                    
                    # Draw visualization
                    if smoothed_gesture and smoothed_confidence > self.config.min_gesture_confidence:
                        self.draw_enhanced_landmarks(
                            img, hand_landmarks, handedness, 
                            smoothed_gesture, smoothed_confidence, fingers
                        )
                    else:
                        self.draw_enhanced_landmarks(
                            img, hand_landmarks, handedness, "", 0.0, fingers
                        )
            else:
                # Show instruction when no hands detected
                cv2.putText(img, "Show your hand to the camera", 
                           (w//2 - 180, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            
            cv2.imshow('Advanced Hand Gesture Recognition System', img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        print("\n👋 Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Application closed successfully")

if __name__ == "__main__":
    # Optimized configuration for best accuracy
    config = GestureConfig(
        max_num_hands=2,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.65,
        gesture_buffer_size=7,
        angle_threshold=155,
        curl_threshold=130,
        min_gesture_confidence=0.65
    )
    
    recognizer = AdvancedHandGestureRecognizer(config)
    recognizer.run()