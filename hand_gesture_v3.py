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
    min_gesture_confidence: float = 0.7
    pinch_threshold: float = 0.05
    two_hand_proximity_threshold: float = 0.3

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
        self.two_hand_history = deque(maxlen=self.config.gesture_buffer_size)
        
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
    
    def calculate_3d_distance(self, l1, l2) -> float:
        """Calculate 3D distance between landmarks"""
        return math.sqrt((l1.x - l2.x)**2 + (l1.y - l2.y)**2 + (l1.z - l2.z)**2)
    
    def is_finger_extended(self, landmarks: List, finger_idx: int, handedness: str) -> Tuple[bool, float]:
        """
        Advanced finger extension detection using multiple metrics
        Returns: (is_extended, confidence)
        """
        if finger_idx == 0:  # Thumb
            return self._is_thumb_extended(landmarks, handedness)
        
        # For other fingers, check multiple conditions
        tip = finger_idx
        dip = finger_idx
        pip = finger_idx
        mcp = finger_idx
        
        tip_point = [landmarks[self.tip_ids[tip]].x, landmarks[self.tip_ids[tip]].y]
        dip_point = [landmarks[self.dip_ids[dip]].x, landmarks[self.dip_ids[dip]].y]
        pip_point = [landmarks[self.pip_ids[pip]].x, landmarks[self.pip_ids[pip]].y]
        mcp_point = [landmarks[self.mcp_ids[mcp]].x, landmarks[self.mcp_ids[mcp]].y]
        
        # Calculate angles
        angle_pip = self.calculate_angle(tip_point, pip_point, mcp_point)
        angle_dip = self.calculate_angle(tip_point, dip_point, pip_point)
        
        # Check if tip is higher than multiple joints
        tip_y = landmarks[self.tip_ids[tip]].y
        pip_y = landmarks[self.pip_ids[pip]].y
        mcp_y = landmarks[self.mcp_ids[mcp]].y
        
        tip_above_pip = tip_y < pip_y
        tip_above_mcp = tip_y < mcp_y
        
        # Distance check
        tip_to_mcp_dist = self.euclidean_distance(tip_point, mcp_point)
        pip_to_mcp_dist = self.euclidean_distance(pip_point, mcp_point)
        distance_extended = tip_to_mcp_dist > pip_to_mcp_dist * 1.2
        
        # Combined conditions for better accuracy
        angle_extended = (angle_pip > self.config.angle_threshold and 
                         angle_dip > self.config.angle_threshold - 20)
        
        is_extended = (angle_extended and tip_above_pip) or (tip_above_pip and tip_above_mcp and distance_extended)
        
        # Confidence calculation
        angle_conf = min((angle_pip + angle_dip) / 360.0, 1.0)
        position_conf = 1.0 if tip_above_pip and tip_above_mcp else 0.5
        confidence = (angle_conf + position_conf) / 2
        
        return is_extended, confidence
    
    def _is_thumb_extended(self, landmarks: List, handedness: str) -> Tuple[bool, float]:
        """Enhanced thumb detection with multiple checks"""
        thumb_tip = [landmarks[self.tip_ids[0]].x, landmarks[self.tip_ids[0]].y]
        thumb_ip = [landmarks[3].x, landmarks[3].y]
        thumb_mcp = [landmarks[2].x, landmarks[2].y]
        thumb_cmc = [landmarks[1].x, landmarks[1].y]
        index_mcp = [landmarks[5].x, landmarks[5].y]
        wrist = [landmarks[0].x, landmarks[0].y]
        
        # Multiple angle checks
        angle_ip = self.calculate_angle(thumb_tip, thumb_ip, thumb_mcp)
        angle_mcp = self.calculate_angle(thumb_ip, thumb_mcp, thumb_cmc)
        
        # Distance checks
        tip_to_index = self.euclidean_distance(thumb_tip, index_mcp)
        tip_to_wrist = self.euclidean_distance(thumb_tip, wrist)
        mcp_to_wrist = self.euclidean_distance(thumb_mcp, wrist)
        
        # Handedness-based orientation
        if handedness == "Right":
            x_extended = thumb_tip[0] < thumb_mcp[0]
        else:
            x_extended = thumb_tip[0] > thumb_mcp[0]
        
        # Combined conditions
        angle_ok = angle_ip > 140 and angle_mcp > 100
        distance_ok = tip_to_index > 0.08 and tip_to_wrist > mcp_to_wrist * 0.9
        
        is_extended = (angle_ok or distance_ok) and x_extended
        confidence = min((angle_ip / 180.0 + tip_to_index * 5 + (1.0 if x_extended else 0.0)) / 3, 1.0)
        
        return is_extended, confidence
    
    def is_finger_curled(self, landmarks: List, finger_idx: int) -> bool:
        """Check if finger is curled (not just not extended)"""
        if finger_idx == 0:
            return False  # Skip thumb for curl detection
        
        tip = landmarks[self.tip_ids[finger_idx]]
        mcp = landmarks[self.mcp_ids[finger_idx]]
        
        # Curled if tip is below or very close to MCP
        return tip.y > mcp.y * 0.95
    
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
        palm_indices = [0, 1, 5, 9, 13, 17]
        cx = np.mean([landmarks[i].x for i in palm_indices])
        cy = np.mean([landmarks[i].y for i in palm_indices])
        return cx, cy
    
    def get_hand_rotation(self, landmarks: List) -> float:
        """Get hand rotation angle (for palm orientation)"""
        wrist = [landmarks[0].x, landmarks[0].y]
        middle_mcp = [landmarks[9].x, landmarks[9].y]
        
        angle = math.atan2(middle_mcp[1] - wrist[1], middle_mcp[0] - wrist[0])
        return math.degrees(angle)
    
    def detect_single_hand_gesture(self, fingers: List[int], landmarks: List, 
                                   handedness: str) -> Tuple[Optional[str], float]:
        """
        Comprehensive single-hand gesture detection
        Returns: (gesture_name, confidence)
        """
        gestures = []
        
        # Calculate useful metrics
        palm_x, palm_y = self.calculate_palm_center(landmarks)
        thumb_tip = [landmarks[4].x, landmarks[4].y]
        index_tip = [landmarks[8].x, landmarks[8].y]
        middle_tip = [landmarks[12].x, landmarks[12].y]
        ring_tip = [landmarks[16].x, landmarks[16].y]
        pinky_tip = [landmarks[20].x, landmarks[20].y]
        
        thumb_index_dist = self.euclidean_distance(thumb_tip, index_tip)
        index_middle_dist = self.euclidean_distance(index_tip, middle_tip)
        middle_ring_dist = self.euclidean_distance(middle_tip, ring_tip)
        ring_pinky_dist = self.euclidean_distance(ring_tip, pinky_tip)
        
        # Hand rotation
        rotation = self.get_hand_rotation(landmarks)
        
        # ==================== NUMBER GESTURES ====================
        
        # Zero (OK sign or closed fist with thumb out)
        if fingers == [1, 0, 0, 0, 0] and thumb_index_dist < 0.08:
            gestures.append(("Zero/OK", 0.92))
        
        # One
        elif fingers == [0, 1, 0, 0, 0]:
            gestures.append(("One", 0.95))
        
        # Two (Peace/Victory)
        elif fingers == [0, 1, 1, 0, 0]:
            if index_middle_dist > 0.08:
                gestures.append(("Two/Peace", 0.93))
            else:
                gestures.append(("Two", 0.88))
        
        # Three
        elif fingers == [0, 1, 1, 1, 0]:
            gestures.append(("Three", 0.94))
        
        # Four
        elif fingers == [0, 1, 1, 1, 1]:
            gestures.append(("Four", 0.94))
        
        # Five (Open Palm)
        elif fingers == [1, 1, 1, 1, 1]:
            gestures.append(("Five/Open Palm", 0.95))
        
        # Six (Thumb and Pinky)
        elif fingers == [1, 0, 0, 0, 1]:
            if thumb_index_dist > 0.15:
                gestures.append(("Six/Shaka", 0.90))
        
        # Seven (Thumb, Index, Pinky)
        elif fingers == [1, 1, 0, 0, 1]:
            gestures.append(("Seven/Rock", 0.88))
        
        # Eight (All except ring and pinky)
        elif fingers == [1, 1, 1, 0, 0]:
            gestures.append(("Eight/Three", 0.87))
        
        # Nine (All except pinky)
        elif fingers == [1, 1, 1, 1, 0]:
            gestures.append(("Nine/Four", 0.87))
        
        # Ten (Fist)
        elif fingers == [0, 0, 0, 0, 0]:
            gestures.append(("Ten/Fist", 0.92))
        
        # ==================== SYMBOL GESTURES ====================
        
        # Thumbs Up
        elif fingers == [1, 0, 0, 0, 0]:
            thumb_y = landmarks[4].y
            index_y = landmarks[8].y
            if thumb_y < index_y - 0.1:
                gestures.append(("Thumbs Up", 0.93))
            elif thumb_y > index_y + 0.1:
                gestures.append(("Thumbs Down", 0.93))
        
        # OK Sign (precise)
        elif thumb_index_dist < 0.04 and fingers[2:] == [1, 1, 1]:
            gestures.append(("OK Sign", 0.91))
        
        # Pinch
        elif fingers[0] == 1 and fingers[1] == 1 and thumb_index_dist < self.config.pinch_threshold:
            gestures.append(("Pinch", 0.89))
        
        # Rock On / Heavy Metal
        elif fingers == [0, 1, 0, 0, 1]:
            gestures.append(("Rock On", 0.92))
        
        # Spider-Man / ILY (I Love You)
        elif fingers == [1, 1, 0, 0, 1]:
            gestures.append(("ILY/Spider-Man", 0.90))
        
        # Call Me / Shaka
        elif fingers == [1, 0, 0, 0, 1] and thumb_index_dist > 0.12:
            gestures.append(("Call Me/Shaka", 0.91))
        
        # Finger Gun
        elif fingers == [1, 1, 0, 0, 0]:
            thumb_above = landmarks[4].y < landmarks[8].y
            if thumb_above:
                gestures.append(("Finger Gun", 0.88))
        
        # Vulcan Salute (Live Long and Prosper)
        elif fingers == [0, 1, 1, 1, 1]:
            # Check if there's a gap between middle and ring
            if middle_ring_dist > 0.08 and index_middle_dist < 0.05:
                gestures.append(("Vulcan Salute", 0.85))
        
        # Middle Finger
        elif fingers == [0, 0, 1, 0, 0]:
            # Verify others are curled
            if all([self.is_finger_curled(landmarks, i) for i in [1, 3, 4]]):
                gestures.append(("Middle Finger", 0.90))
        
        # Pointing (Index only, others curled)
        elif fingers == [0, 1, 0, 0, 0]:
            if self.is_finger_curled(landmarks, 2):
                gestures.append(("Pointing", 0.92))
        
        # Hang Loose / Shaka (refined)
        elif fingers == [1, 0, 0, 0, 1]:
            # Check if thumb and pinky are far apart
            thumb_pinky_dist = self.euclidean_distance(thumb_tip, pinky_tip)
            if thumb_pinky_dist > 0.2:
                gestures.append(("Hang Loose", 0.89))
        
        # Stop/Talk to the Hand
        elif fingers == [0, 1, 1, 1, 1]:
            # Check if palm is facing forward (rotation check)
            if abs(rotation) < 45 or abs(rotation) > 135:
                gestures.append(("Stop/High Five", 0.87))
        
        # Return best gesture or None
        if gestures:
            gestures.sort(key=lambda x: x[1], reverse=True)
            return gestures[0]
        
        return None, 0.0
    
    def detect_two_hand_gesture(self, hand1_data: Dict, hand2_data: Dict) -> Tuple[Optional[str], float]:
        """
        Detect gestures that require two hands
        Returns: (gesture_name, confidence)
        """
        landmarks1 = hand1_data['landmarks']
        landmarks2 = hand2_data['landmarks']
        fingers1 = hand1_data['fingers']
        fingers2 = hand2_data['fingers']
        handedness1 = hand1_data['handedness']
        handedness2 = hand1_data['handedness']
        
        gestures = []
        
        # Get key points from both hands
        l_thumb_tip = [landmarks1[4].x, landmarks1[4].y]
        l_index_tip = [landmarks1[8].x, landmarks1[8].y]
        l_palm = self.calculate_palm_center(landmarks1)
        
        r_thumb_tip = [landmarks2[4].x, landmarks2[4].y]
        r_index_tip = [landmarks2[8].x, landmarks2[8].y]
        r_palm = self.calculate_palm_center(landmarks2)
        
        # Calculate inter-hand distances
        palm_distance = self.euclidean_distance(l_palm, r_palm)
        thumb_distance = self.euclidean_distance(l_thumb_tip, r_thumb_tip)
        index_distance = self.euclidean_distance(l_index_tip, r_index_tip)
        
        # ==================== TWO-HAND GESTURES ====================
        
        # Heart (thumbs and index fingers forming heart shape)
        if (fingers1 == [1, 1, 0, 0, 0] and fingers2 == [1, 1, 0, 0, 0]):
            # Check if thumbs are close and index fingers are close
            if thumb_distance < 0.08 and index_distance < 0.15:
                # Check if they form a heart shape (thumbs at bottom, indices at top)
                thumbs_below = (landmarks1[4].y + landmarks2[4].y) / 2 > (landmarks1[8].y + landmarks2[8].y) / 2
                if thumbs_below:
                    gestures.append(("Heart", 0.92))
        
        # Prayer/Namaste (palms together)
        if (fingers1 == [1, 1, 1, 1, 1] and fingers2 == [1, 1, 1, 1, 1]):
            if palm_distance < 0.15:
                gestures.append(("Prayer/Namaste", 0.90))
        
        # Clap (palms close together)
        if palm_distance < 0.12:
            # Both hands open
            if sum(fingers1) >= 4 and sum(fingers2) >= 4:
                gestures.append(("Clap", 0.88))
        
        # High Five (both palms open and close)
        if (fingers1 == [0, 1, 1, 1, 1] or fingers1 == [1, 1, 1, 1, 1]) and \
           (fingers2 == [0, 1, 1, 1, 1] or fingers2 == [1, 1, 1, 1, 1]):
            if palm_distance < 0.15:
                gestures.append(("High Five", 0.89))
        
        # Fist Bump (both fists close together)
        if fingers1 == [0, 0, 0, 0, 0] and fingers2 == [0, 0, 0, 0, 0]:
            if palm_distance < 0.15:
                gestures.append(("Fist Bump", 0.91))
        
        # Diamond/Triangle (index fingers and thumbs forming shape)
        if (fingers1 == [1, 1, 0, 0, 0] and fingers2 == [1, 1, 0, 0, 0]):
            # Check if tips are close to form diamond
            l_thumb = [landmarks1[4].x, landmarks1[4].y]
            l_index = [landmarks1[8].x, landmarks1[8].y]
            r_thumb = [landmarks2[4].x, landmarks2[4].y]
            r_index = [landmarks2[8].x, landmarks2[8].y]
            
            thumb_dist = self.euclidean_distance(l_thumb, r_thumb)
            index_dist = self.euclidean_distance(l_index, r_index)
            
            if thumb_dist < 0.06 and index_dist < 0.06:
                gestures.append(("Diamond/Triangle", 0.87))
        
        # Time Out (T-shape with hands)
        if (fingers1 == [0, 1, 1, 1, 1] and fingers2 == [0, 1, 1, 1, 1]):
            # Check if one hand is horizontal and other is vertical
            hand1_horizontal = abs(landmarks1[8].y - landmarks1[20].y) < 0.1
            hand2_vertical = abs(landmarks2[8].x - landmarks2[20].x) < 0.1
            
            if (hand1_horizontal and hand2_vertical) or (not hand1_horizontal and not hand2_vertical):
                if palm_distance < 0.25:
                    gestures.append(("Time Out", 0.85))
        
        # Frame/Picture (thumbs and index fingers forming rectangle)
        if (fingers1 == [1, 1, 0, 0, 0] and fingers2 == [1, 1, 0, 0, 0]):
            # Check if hands are positioned to form a frame
            if palm_distance > 0.2 and palm_distance < 0.5:
                gestures.append(("Frame/Picture", 0.84))
        
        # Double Peace
        if (fingers1 == [0, 1, 1, 0, 0] and fingers2 == [0, 1, 1, 0, 0]):
            if palm_distance > 0.15:
                gestures.append(("Double Peace", 0.88))
        
        # Double Thumbs Up
        if (fingers1 == [1, 0, 0, 0, 0] and fingers2 == [1, 0, 0, 0, 0]):
            # Check if both thumbs are up
            thumb1_up = landmarks1[4].y < landmarks1[8].y
            thumb2_up = landmarks2[4].y < landmarks2[8].y
            if thumb1_up and thumb2_up:
                gestures.append(("Double Thumbs Up", 0.90))
        
        # Crossing Arms/X
        if (fingers1 == [0, 1, 1, 1, 1] and fingers2 == [0, 1, 1, 1, 1]):
            # Check if wrists are close (crossed position)
            wrist1 = [landmarks1[0].x, landmarks1[0].y]
            wrist2 = [landmarks2[0].x, landmarks2[0].y]
            wrist_dist = self.euclidean_distance(wrist1, wrist2)
            
            if wrist_dist < 0.2 and palm_distance > 0.3:
                gestures.append(("X/Cross", 0.83))
        
        # Return best gesture
        if gestures:
            gestures.sort(key=lambda x: x[1], reverse=True)
            return gestures[0]
        
        return None, 0.0
    
    def smooth_gesture(self, current_gesture: Optional[str], is_two_hand: bool = False) -> Optional[str]:
        """Apply temporal smoothing to reduce jitter"""
        history = self.two_hand_history if is_two_hand else self.gesture_history
        history.append(current_gesture)
        
        if len(history) < self.config.gesture_buffer_size:
            return current_gesture
        
        # Count occurrences
        gesture_counts = {}
        for g in history:
            if g is not None:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        if not gesture_counts:
            return None
        
        # Return most common gesture if it appears in majority of frames
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        threshold = self.config.gesture_buffer_size * 0.55
        
        return most_common[0] if most_common[1] >= threshold else current_gesture
    
    def draw_enhanced_landmarks(self, img, hand_landmarks, handedness: str, 
                               gesture: str, confidence: float, position: int = 0):
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
        
        # Offset for multiple hands
        y_offset = position * 60
        
        # Display gesture with confidence
        if gesture:
            text = f"{gesture} ({confidence:.0%})"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (wx - 10, wy - text_h - 15 + y_offset), 
                         (wx + text_w + 10, wy - 5 + y_offset), (0, 50, 0), -1)
            cv2.putText(img, text, (wx, wy - 10 + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display handedness
        cv2.putText(img, f"{handedness}", (wx, wy + 25 + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_two_hand_gesture(self, img, gesture: str, confidence: float):
        """Draw two-hand gesture at top center"""
        h, w, _ = img.shape
        
        if gesture:
            text = f"TWO-HAND: {gesture} ({confidence:.0%})"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            
            x = (w - text_w) // 2
            y = 50
            
            # Background
            cv2.rectangle(img, (x - 15, y - text_h - 10), 
                         (x + text_w + 15, y + 10), (0, 0, 100), -1)
            cv2.rectangle(img, (x - 15, y - text_h - 10), 
                         (x + text_w + 15, y + 10), (0, 255, 255), 3)
            
            # Text
            cv2.putText(img, text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    
    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=" * 70)
        print(" " * 15 + "🤚 ADVANCED HAND GESTURE RECOGNITION 🤚")
        print("=" * 70)
        print("\n📋 SINGLE-HAND GESTURES:")
        print("  Numbers: 0-10 (various finger combinations)")
        print("  Symbols: Thumbs Up/Down, Peace, OK, Pointing, Stop")
        print("  Signs: Rock On, ILY, Shaka, Finger Gun, Vulcan Salute, Pinch")
        print("\n🤝 TWO-HAND GESTURES:")
        print("  Heart, Prayer/Namaste, Clap, High Five, Fist Bump")
        print("  Diamond, Time Out, Frame, Double Peace, Double Thumbs Up, X/Cross")
        print("\n⌨️  Press 'q' to exit | 'h' to toggle help")
        print("=" * 70 + "\n")
        
        fps_history = deque(maxlen=30)
        prev_time = cv2.getTickCount()
        show_help = False
        
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
            
            h, w, _ = img.shape
            
            # Draw header
            cv2.rectangle(img, (0, 0), (w, 45), (30, 30, 30), -1)
            cv2.putText(img, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "Advanced Gesture Recognition", (w//2 - 180, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            detected_hands = []
            
            if results.multi_hand_landmarks and results.multi_handedness:
                num_hands = len(results.multi_hand_landmarks)
                
                # Process each hand
                for idx, (hand_landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks, 
                                                                       results.multi_handedness)):
                    handedness = hand_info.classification[0].label
                    
                    # Detect finger states
                    fingers, finger_confidence = self.get_finger_states(
                        hand_landmarks.landmark, handedness
                    )
                    
                    # Store hand data
                    hand_data = {
                        'landmarks': hand_landmarks.landmark,
                        'fingers': fingers,
                        'handedness': handedness,
                        'hand_landmarks': hand_landmarks
                    }
                    detected_hands.append(hand_data)
                    
                    # Detect single-hand gesture
                    gesture, gesture_confidence = self.detect_single_hand_gesture(
                        fingers, hand_landmarks.landmark, handedness
                    )
                    
                    # Apply temporal smoothing
                    smoothed_gesture = self.smooth_gesture(gesture, is_two_hand=False)
                    
                    # Draw visualization
                    if smoothed_gesture and gesture_confidence > self.config.min_gesture_confidence:
                        self.draw_enhanced_landmarks(
                            img, hand_landmarks, handedness, 
                            smoothed_gesture, gesture_confidence, idx
                        )
                    else:
                        self.draw_enhanced_landmarks(
                            img, hand_landmarks, handedness, "", 0.0, idx
                        )
                
                # Check for two-hand gestures
                if len(detected_hands) == 2:
                    two_hand_gesture, two_hand_conf = self.detect_two_hand_gesture(
                        detected_hands[0], detected_hands[1]
                    )
                    
                    # Apply temporal smoothing for two-hand gestures
                    smoothed_two_hand = self.smooth_gesture(two_hand_gesture, is_two_hand=True)
                    
                    if smoothed_two_hand and two_hand_conf > self.config.min_gesture_confidence:
                        self.draw_two_hand_gesture(img, smoothed_two_hand, two_hand_conf)
                else:
                    self.two_hand_history.clear()
            
            # Hand count indicator
            num_hands_detected = len(detected_hands) if results.multi_hand_landmarks else 0
            hand_text = f"Hands: {num_hands_detected}"
            cv2.putText(img, hand_text, (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            
            cv2.imshow('Advanced Hand Gesture Recognition', img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_help = not show_help
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config = GestureConfig(
        min_detection_confidence=0.75,
        min_tracking_confidence=0.65,
        gesture_buffer_size=7,
        angle_threshold=155,
        min_gesture_confidence=0.7
    )
    recognizer = AdvancedHandGestureRecognizer(config)
    recognizer.run()

# python advanced_gesture_recognition.py