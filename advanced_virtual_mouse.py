import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum

class MouseMode(Enum):
    IDLE = 0
    MOVE = 1
    CLICK = 2
    RIGHT_CLICK = 3
    DRAG = 4
    SCROLL = 5
    DOUBLE_CLICK = 6
    ZOOM = 7

@dataclass
class MouseConfig:
    """Configuration for virtual mouse"""
    cam_width: int = 1280
    cam_height: int = 720
    frame_reduction: int = 100
    smoothening: int = 7
    
    # Detection thresholds
    palm_confidence_threshold: float = 0.7
    click_distance: float = 25
    drag_distance: float = 30
    double_click_interval: float = 0.4
    scroll_sensitivity: float = 20
    zoom_sensitivity: float = 50
    
    # Stability
    movement_threshold: float = 3.0  # Min pixels to move
    stability_frames: int = 3
    
    # Kalman filter parameters
    process_noise: float = 0.01
    measurement_noise: float = 0.1

class KalmanFilter:
    """2D Kalman filter for smooth cursor tracking"""
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # Error covariance
        self.P = np.eye(4)
        
        self.initialized = False
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement"""
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return measurement
        
        # Prediction
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.state[:2]

class AdvancedVirtualMouse:
    def __init__(self, config: MouseConfig = None):
        self.config = config or MouseConfig()
        
        # PyAutoGUI setup
        pyautogui.FAILSAFE = False
        self.screen_width, self.screen_height = pyautogui.size()
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Landmark indices
        self.tip_ids = [4, 8, 12, 16, 20]
        self.pip_ids = [2, 6, 10, 14, 18]
        
        # Kalman filter for smooth tracking
        self.kalman_filter = KalmanFilter(
            self.config.process_noise,
            self.config.measurement_noise
        )
        
        # State tracking
        self.current_mode = MouseMode.IDLE
        self.prev_mode = MouseMode.IDLE
        self.last_click_time = 0
        self.is_dragging = False
        self.palm_detected = False
        
        # Stability tracking
        self.stable_position = None
        self.stability_counter = 0
        
        # Mode history for smoothing
        self.mode_history = deque(maxlen=5)
        
        # Click debouncing
        self.last_action_time = 0
        self.action_cooldown = 0.3
        
    def euclidean_distance(self, p1: List[float], p2: List[float]) -> float:
        """Calculate distance between two points"""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    def calculate_angle(self, p1: List[float], p2: List[float], p3: List[float]) -> float:
        """Calculate angle at p2"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def is_finger_extended(self, lm_list: List, finger_idx: int) -> bool:
        """Check if finger is extended using improved logic"""
        if finger_idx == 0:  # Thumb
            return lm_list[self.tip_ids[0]][1] < lm_list[self.pip_ids[0]][1]
        
        # Check Y-coordinate and angle
        tip_y = lm_list[self.tip_ids[finger_idx]][2]
        pip_y = lm_list[self.pip_ids[finger_idx]][2]
        
        # Calculate angle at PIP joint
        tip = lm_list[self.tip_ids[finger_idx]][1:]
        pip = lm_list[self.pip_ids[finger_idx]][1:]
        mcp = lm_list[self.pip_ids[finger_idx] - 4][1:]
        
        angle = self.calculate_angle(tip, pip, mcp)
        
        return tip_y < pip_y and angle > 140
    
    def get_fingers_state(self, lm_list: List) -> List[int]:
        """Get state of all fingers"""
        fingers = []
        for i in range(5):
            fingers.append(1 if self.is_finger_extended(lm_list, i) else 0)
        return fingers
    
    def detect_palm_open(self, fingers: List[int]) -> bool:
        """Detect if palm is open (activation gesture)"""
        return sum(fingers) >= 4
    
    def get_finger_tip(self, lm_list: List, finger_idx: int) -> Tuple[int, int]:
        """Get finger tip coordinates"""
        return lm_list[self.tip_ids[finger_idx]][1:]
    
    def interpolate_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Convert camera coordinates to screen coordinates"""
        x_screen = np.interp(x, 
                            (self.config.frame_reduction, 
                             self.config.cam_width - self.config.frame_reduction),
                            (0, self.screen_width))
        y_screen = np.interp(y,
                            (self.config.frame_reduction,
                             self.config.cam_height - self.config.frame_reduction),
                            (0, self.screen_height))
        return x_screen, y_screen
    
    def apply_kalman_smoothing(self, x: float, y: float) -> Tuple[float, float]:
        """Apply Kalman filtering for smooth movement"""
        measurement = np.array([x, y])
        smoothed = self.kalman_filter.update(measurement)
        return smoothed[0], smoothed[1]
    
    def is_stable_position(self, x: float, y: float) -> bool:
        """Check if position is stable"""
        if self.stable_position is None:
            self.stable_position = (x, y)
            self.stability_counter = 0
            return False
        
        distance = self.euclidean_distance([x, y], list(self.stable_position))
        
        if distance < self.config.movement_threshold:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            self.stable_position = (x, y)
        
        return self.stability_counter >= self.config.stability_frames
    
    def determine_mode(self, fingers: List[int], lm_list: List) -> MouseMode:
        """Determine mouse mode based on finger configuration"""
        # Palm not open - idle
        if not self.detect_palm_open(fingers):
            if not (fingers[1] == 1):  # No index finger
                return MouseMode.IDLE
        
        # Get finger positions
        index_tip = self.get_finger_tip(lm_list, 1)
        middle_tip = self.get_finger_tip(lm_list, 2)
        ring_tip = self.get_finger_tip(lm_list, 3)
        thumb_tip = self.get_finger_tip(lm_list, 0)
        pinky_tip = self.get_finger_tip(lm_list, 4)
        
        # Calculate distances
        index_middle_dist = self.euclidean_distance(index_tip, middle_tip)
        index_thumb_dist = self.euclidean_distance(index_tip, thumb_tip)
        thumb_pinky_dist = self.euclidean_distance(thumb_tip, pinky_tip)
        
        # Mode 1: Only index up - MOVE
        if fingers == [0, 1, 0, 0, 0]:
            return MouseMode.MOVE
        
        # Mode 2: Index + Middle together - LEFT CLICK
        if fingers[1] == 1 and fingers[2] == 1:
            if index_middle_dist < self.config.click_distance:
                return MouseMode.CLICK
        
        # Mode 3: Index + Middle + Ring - RIGHT CLICK
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            middle_ring_dist = self.euclidean_distance(middle_tip, ring_tip)
            if middle_ring_dist < self.config.click_distance:
                return MouseMode.RIGHT_CLICK
        
        # Mode 4: Thumb + Index pinch - DRAG
        if fingers[0] == 1 and fingers[1] == 1:
            if index_thumb_dist < self.config.drag_distance:
                return MouseMode.DRAG
        
        # Mode 5: Thumb + Pinky - SCROLL
        if fingers[0] == 1 and fingers[4] == 1 and fingers[1] == 0:
            return MouseMode.SCROLL
        
        # Mode 6: All fingers + thumb - ZOOM (pinch/spread)
        if sum(fingers) == 5:
            return MouseMode.ZOOM
        
        # Mode 7: Index + Middle moving - Check for double click
        if self.current_mode == MouseMode.CLICK:
            time_since_last = time.time() - self.last_click_time
            if time_since_last < self.config.double_click_interval:
                return MouseMode.DOUBLE_CLICK
        
        return MouseMode.MOVE if fingers[1] == 1 else MouseMode.IDLE
    
    def smooth_mode(self, mode: MouseMode) -> MouseMode:
        """Apply temporal smoothing to mode detection"""
        self.mode_history.append(mode)
        
        if len(self.mode_history) < 3:
            return mode
        
        # Count mode occurrences
        mode_counts = {}
        for m in self.mode_history:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        
        # Return most common mode
        return max(mode_counts.items(), key=lambda x: x[1])[0]
    
    def execute_action(self, mode: MouseMode, lm_list: List, img):
        """Execute mouse action based on mode"""
        current_time = time.time()
        
        # Get index finger position
        index_x, index_y = self.get_finger_tip(lm_list, 1)
        
        # Interpolate and smooth
        screen_x, screen_y = self.interpolate_coordinates(index_x, index_y)
        smooth_x, smooth_y = self.apply_kalman_smoothing(screen_x, screen_y)
        
        # Clamp to screen bounds
        smooth_x = np.clip(smooth_x, 0, self.screen_width - 1)
        smooth_y = np.clip(smooth_y, 0, self.screen_height - 1)
        
        if mode == MouseMode.MOVE:
            pyautogui.moveTo(smooth_x, smooth_y)
            cv2.circle(img, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)
            self.draw_status(img, "Moving Cursor", (255, 0, 255))
        
        elif mode == MouseMode.CLICK:
            if current_time - self.last_action_time > self.action_cooldown:
                if self.is_stable_position(index_x, index_y):
                    pyautogui.click()
                    self.last_click_time = current_time
                    self.last_action_time = current_time
                    self.draw_status(img, "Left Click!", (0, 255, 0))
        
        elif mode == MouseMode.DOUBLE_CLICK:
            if current_time - self.last_action_time > self.action_cooldown:
                pyautogui.doubleClick()
                self.last_action_time = current_time
                self.draw_status(img, "Double Click!", (0, 255, 255))
        
        elif mode == MouseMode.RIGHT_CLICK:
            if current_time - self.last_action_time > self.action_cooldown:
                if self.is_stable_position(index_x, index_y):
                    pyautogui.rightClick()
                    self.last_action_time = current_time
                    self.draw_status(img, "Right Click!", (0, 0, 255))
        
        elif mode == MouseMode.DRAG:
            if not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
            
            pyautogui.moveTo(smooth_x, smooth_y)
            self.draw_status(img, "Dragging", (0, 255, 0))
        
        elif mode == MouseMode.SCROLL:
            # Use wrist Y position for scroll direction
            wrist_y = lm_list[0][2]
            middle_y = self.config.cam_height // 2
            
            if wrist_y < middle_y - 50:
                pyautogui.scroll(int(self.config.scroll_sensitivity))
                self.draw_status(img, "Scroll Up", (255, 255, 0))
            elif wrist_y > middle_y + 50:
                pyautogui.scroll(-int(self.config.scroll_sensitivity))
                self.draw_status(img, "Scroll Down", (255, 255, 0))
        
        elif mode == MouseMode.ZOOM:
            # Calculate hand span for zoom
            thumb_tip = self.get_finger_tip(lm_list, 0)
            pinky_tip = self.get_finger_tip(lm_list, 4)
            span = self.euclidean_distance(thumb_tip, pinky_tip)
            
            # Zoom based on span change
            if hasattr(self, 'prev_span'):
                span_diff = span - self.prev_span
                if abs(span_diff) > 5:
                    if span_diff > 0:
                        pyautogui.hotkey('ctrl', '+')
                        self.draw_status(img, "Zoom In", (255, 0, 255))
                    else:
                        pyautogui.hotkey('ctrl', '-')
                        self.draw_status(img, "Zoom Out", (255, 0, 255))
            self.prev_span = span
        
        # Release drag if mode changed
        if self.is_dragging and mode != MouseMode.DRAG:
            pyautogui.mouseUp()
            self.is_dragging = False
    
    def draw_status(self, img, text: str, color: Tuple[int, int, int]):
        """Draw status text on image"""
        cv2.putText(img, text, (20, 100),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
    
    def draw_ui(self, img, mode: MouseMode, fingers: List[int], fps: float):
        """Draw user interface elements"""
        h, w, _ = img.shape
        
        # Draw active region
        cv2.rectangle(img,
                     (self.config.frame_reduction, self.config.frame_reduction),
                     (w - self.config.frame_reduction, h - self.config.frame_reduction),
                     (255, 0, 255), 2)
        
        # Info panel
        panel_h = 250
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (350, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Title
        cv2.putText(img, "Virtual Mouse", (10, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        
        # FPS
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mode
        mode_text = mode.name.replace('_', ' ')
        cv2.putText(img, f"Mode: {mode_text}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        y_offset = 130
        instructions = [
            "Index: Move cursor",
            "Index+Middle: Click",
            "Index+Mid+Ring: Right Click",
            "Thumb+Index: Drag",
            "Thumb+Pinky: Scroll"
        ]
        
        for instruction in instructions:
            cv2.putText(img, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
    
    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.cam_height)
        
        print("=" * 70)
        print("Advanced Virtual Mouse Control System")
        print("=" * 70)
        print("\nGestures:")
        print("  • Index Finger Only       → Move Cursor")
        print("  • Index + Middle Touch    → Left Click")
        print("  • Index + Mid + Ring      → Right Click")
        print("  • Thumb + Index Pinch     → Drag & Drop")
        print("  • Thumb + Pinky           → Scroll")
        print("  • All Fingers Open        → Zoom (Ctrl +/-)")
        print("\nFeatures:")
        print("  • Kalman filter smoothing")
        print("  • Adaptive gesture recognition")
        print("  • Stability-based clicking")
        print("  • Palm detection activation")
        print("\nPress 'q' to exit")
        print("=" * 70)
        
        fps_history = deque(maxlen=30)
        prev_time = cv2.getTickCount()
        
        while True:
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
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract landmark list
                    lm_list = []
                    h, w, c = img.shape
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])
                    
                    if len(lm_list) != 0:
                        # Get finger states
                        fingers = self.get_fingers_state(lm_list)
                        
                        # Determine mode
                        mode = self.determine_mode(fingers, lm_list)
                        smoothed_mode = self.smooth_mode(mode)
                        
                        # Execute action
                        self.execute_action(smoothed_mode, lm_list, img)
                        self.current_mode = smoothed_mode
                        
                        # Draw UI
                        self.draw_ui(img, smoothed_mode, fingers, avg_fps)
            else:
                # No hand detected - reset states
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                self.current_mode = MouseMode.IDLE
                self.draw_ui(img, MouseMode.IDLE, [0]*5, avg_fps)
            
            cv2.imshow('Advanced Virtual Mouse', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        if self.is_dragging:
            pyautogui.mouseUp()
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config = MouseConfig(
        smoothening=7,
        stability_frames=3,
        process_noise=0.01,
        measurement_noise=0.1
    )
    mouse = AdvancedVirtualMouse(config)
    mouse.run()