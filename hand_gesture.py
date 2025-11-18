import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class GestureConfig:
    """Configuration for gesture detection"""
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    gesture_buffer_size: int = 5  # Frames to smooth gestures
    angle_threshold: float = 160  # Degrees for finger straightness
    min_gesture_confidence: float = 0.7

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
        
        # Gesture history for temporal smoothing
        self.gesture_history = deque(maxlen=self.config.gesture_buffer_size)
        
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
    
    def is_finger_extended(self, landmarks: List, finger_idx: int, handedness: str) -> Tuple[bool, float]:
        """
        Advanced finger extension detection using angles and distances
        Returns: (is_extended, confidence)
        """
        if finger_idx == 0:  # Thumb
            return self._is_thumb_extended(landmarks, handedness)
        
        # For other fingers, check angle at PIP joint
        tip = finger_idx
        pip = finger_idx - 1
        mcp = finger_idx - 2
        
        tip_point = [landmarks[self.tip_ids[tip]].x, landmarks[self.tip_ids[tip]].y]
        pip_point = [landmarks[self.pip_ids[tip]].x, landmarks[self.pip_ids[tip]].y]
        mcp_point = [landmarks[self.mcp_ids[tip]].x, landmarks[self.mcp_ids[tip]].y]
        
        # Calculate angle at PIP joint
        angle = self.calculate_angle(tip_point, pip_point, mcp_point)
        
        # Also check if tip is higher than PIP (Y-axis)
        tip_above_pip = landmarks[self.tip_ids[tip]].y < landmarks[self.pip_ids[tip]].y
        
        # Confidence based on angle straightness
        confidence = min(angle / 180.0, 1.0) if angle > self.config.angle_threshold else 0.3
        
        is_extended = angle > self.config.angle_threshold and tip_above_pip
        return is_extended, confidence
    
    def _is_thumb_extended(self, landmarks: List, handedness: str) -> Tuple[bool, float]:
        """Special logic for thumb detection based on handedness"""
        thumb_tip = [landmarks[self.tip_ids[0]].x, landmarks[self.tip_ids[0]].y]
        thumb_ip = [landmarks[3].x, landmarks[3].y]
        thumb_mcp = [landmarks[2].x, landmarks[2].y]
        index_mcp = [landmarks[5].x, landmarks[5].y]
        
        # Calculate angle
        angle = self.calculate_angle(thumb_tip, thumb_ip, thumb_mcp)
        
        # Distance from index finger
        distance = self.euclidean_distance(thumb_tip, index_mcp)
        
        # Check orientation based on handedness
        if handedness == "Right":
            x_extended = thumb_tip[0] < thumb_mcp[0]
        else:  # Left hand
            x_extended = thumb_tip[0] > thumb_mcp[0]
        
        is_extended = (angle > 140 or distance > 0.1) and x_extended
        confidence = min((angle / 180.0 + distance * 5) / 2, 1.0)
        
        return is_extended, confidence
    
    def get_finger_states(self, landmarks: List, handedness: str) -> Tuple[List[int], float]:
        """
        Get detailed finger states with confidence
        Returns: (finger_states, average_confidence)
        """
        fingers = []
        confidences = []
        
        for i in range(5):
            is_extended, confidence = self.is_finger_extended(landmarks, i, handedness)
            fingers.append(1 if is_extended else 0)
            confidences.append(confidence)
        
        avg_confidence = np.mean(confidences)
        return fingers, avg_confidence
    
    def calculate_palm_center(self, landmarks: List) -> Tuple[float, float]:
        """Calculate center of palm"""
        palm_indices = [0, 1, 5, 9, 13, 17]  # Wrist and base knuckles
        cx = np.mean([landmarks[i].x for i in palm_indices])
        cy = np.mean([landmarks[i].y for i in palm_indices])
        return cx, cy
    
    def detect_gesture(self, fingers: List[int], landmarks: List, handedness: str) -> Tuple[Optional[str], float]:
        """
        Advanced gesture detection with multiple algorithms
        Returns: (gesture_name, confidence)
        """
        # Calculate additional features
        palm_x, palm_y = self.calculate_palm_center(landmarks)
        thumb_index_dist = self.euclidean_distance(
            [landmarks[4].x, landmarks[4].y],
            [landmarks[8].x, landmarks[8].y]
        )
        
        # Gesture patterns with confidence scoring
        gestures = []
        
        # Fist / Rock
        if sum(fingers) == 0:
            gestures.append(("Fist", 0.95))
        
        # Open Palm / Paper
        elif sum(fingers) == 5:
            gestures.append(("Open Palm", 0.95))
        
        # Peace / Victory / Scissors
        elif fingers == [0, 1, 1, 0, 0]:
            # Check if fingers are spread apart
            index_tip = [landmarks[8].x, landmarks[8].y]
            middle_tip = [landmarks[12].x, landmarks[12].y]
            spread = self.euclidean_distance(index_tip, middle_tip)
            confidence = min(0.95, 0.7 + spread * 2)
            gestures.append(("Peace/Scissors", confidence))
        
        # Thumbs Up
        elif fingers == [1, 0, 0, 0, 0]:
            # Verify thumb is pointing up
            thumb_y = landmarks[4].y
            index_y = landmarks[8].y
            if thumb_y < index_y:
                gestures.append(("Thumbs Up", 0.9))
            else:
                gestures.append(("Thumbs Down", 0.9))
        
        # Pointing
        elif fingers == [0, 1, 0, 0, 0]:
            gestures.append(("Pointing", 0.9))
        
        # OK Sign (Thumb + Index circle)
        elif fingers == [0, 0, 1, 1, 1] or fingers == [1, 0, 1, 1, 1]:
            if thumb_index_dist < 0.05:
                gestures.append(("OK Sign", 0.85))
        
        # Rock On / Heavy Metal
        elif fingers == [0, 1, 0, 0, 1]:
            gestures.append(("Rock On", 0.9))
        
        # Spider-Man / I Love You
        elif fingers == [1, 1, 0, 0, 1]:
            gestures.append(("Spider-Man", 0.85))
        
        # Call Me (Thumb + Pinky)
        elif fingers == [1, 0, 0, 0, 1]:
            gestures.append(("Call Me", 0.85))
        
        # Three Fingers
        elif fingers == [0, 1, 1, 1, 0]:
            gestures.append(("Three", 0.9))
        
        # Four Fingers
        elif fingers == [0, 1, 1, 1, 1]:
            gestures.append(("Four", 0.9))
        
        # Pinch (Thumb + Index close)
        elif fingers[0] == 1 and fingers[1] == 1:
            if thumb_index_dist < 0.04:
                gestures.append(("Pinch", 0.8))
        
        # Return best gesture or None
        if gestures:
            gestures.sort(key=lambda x: x[1], reverse=True)
            return gestures[0]
        
        return None, 0.0
    
    def smooth_gesture(self, current_gesture: Optional[str]) -> Optional[str]:
        """Apply temporal smoothing to reduce jitter"""
        self.gesture_history.append(current_gesture)
        
        if len(self.gesture_history) < self.config.gesture_buffer_size:
            return current_gesture
        
        # Count occurrences
        gesture_counts = {}
        for g in self.gesture_history:
            if g is not None:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        if not gesture_counts:
            return None
        
        # Return most common gesture if it appears in majority of frames
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        threshold = self.config.gesture_buffer_size * 0.6
        
        return most_common[0] if most_common[1] >= threshold else current_gesture
    
    def draw_enhanced_landmarks(self, img, hand_landmarks, handedness: str, gesture: str, confidence: float):
        """Draw enhanced visualization with gesture info"""
        h, w, _ = img.shape
        
        # Draw landmarks with custom style
        self.mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Get wrist position for text
        wrist = hand_landmarks.landmark[0]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        
        # Display gesture with confidence
        if gesture:
            # Background rectangle for better visibility
            text = f"{gesture} ({confidence:.0%})"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (wx - 10, wy - text_h - 20), 
                         (wx + text_w + 10, wy - 10), (0, 0, 0), -1)
            cv2.putText(img, text, (wx, wy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display handedness
        cv2.putText(img, f"{handedness} Hand", (wx, wy + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=" * 60)
        print("Advanced Hand Gesture Recognition System")
        print("=" * 60)
        print("\nSupported Gestures:")
        print("  • Fist, Open Palm, Peace/Scissors")
        print("  • Thumbs Up/Down, Pointing, OK Sign")
        print("  • Rock On, Spider-Man, Call Me")
        print("  • Pinch, Three, Four")
        print("\nPress 'q' to exit")
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
            results = self.hands.process(img_rgb)
            
            # Draw FPS
            cv2.putText(img, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, 
                                                      results.multi_handedness):
                    # Get handedness
                    handedness = hand_info.classification[0].label
                    
                    # Detect finger states
                    fingers, finger_confidence = self.get_finger_states(
                        hand_landmarks.landmark, handedness
                    )
                    
                    # Detect gesture
                    gesture, gesture_confidence = self.detect_gesture(
                        fingers, hand_landmarks.landmark, handedness
                    )
                    
                    # Apply temporal smoothing
                    smoothed_gesture = self.smooth_gesture(gesture)
                    
                    # Draw enhanced visualization
                    if smoothed_gesture and gesture_confidence > self.config.min_gesture_confidence:
                        self.draw_enhanced_landmarks(
                            img, hand_landmarks, handedness, 
                            smoothed_gesture, gesture_confidence
                        )
                    else:
                        self.draw_enhanced_landmarks(
                            img, hand_landmarks, handedness, "", 0.0
                        )
            
            cv2.imshow('Advanced Hand Gesture Recognition', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config = GestureConfig(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        gesture_buffer_size=5
    )
    recognizer = AdvancedHandGestureRecognizer(config)
    recognizer.run()