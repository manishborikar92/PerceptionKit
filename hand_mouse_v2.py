#!/usr/bin/env python3
"""
Responsive Hand Gesture Mouse Control System (v2)
=================================================

Updates:
1. Relaxed thresholds for better responsiveness.
2. Screenshot gesture changed to FIST.
3. Jitter tolerance increased for standard webcams.

REQUIREMENTS:
    pip install opencv-python mediapipe pyautogui numpy

CONTROLS:
    - Move index finger: Cursor control
    - Pinch (thumb + index): Left Click
    - Double Pinch: Double Click
    - Pinch + Hold + Move: Drag
    - Three-finger Pinch: Right Click
    - Two fingers (Index+Middle) vertical: Scroll
    - FIST (Clenched hand): Screenshot
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from typing import Tuple, Optional, List, Dict
import time
from datetime import datetime
import os
from collections import deque
from enum import Enum

# ============================================================================
# CONFIGURATION CONSTANTS - RETUNED FOR RESPONSIVENESS
# ============================================================================

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Screen mapping
FRAME_REDUCTION = 0.15

# Mouse smoothing
SMOOTHING_FACTOR = 0.5  # Lowered slightly for snappier movement

# GESTURE TIMING (RELAXED)
MIN_GESTURE_FRAMES = 2      # Was 5 (Faster recognition)
MIN_SCREENSHOT_FRAMES = 5   # Was 10
MIN_CLICK_FRAMES = 1        # Was 3 (Instant click)

# GESTURE DISTANCES (RELAXED)
CLICK_DISTANCE = 0.06       # Was 0.035 (Easier to pinch)
THREE_FINGER_DISTANCE = 0.08 # Was 0.05
FIST_THRESHOLD = 0.28       # Threshold for fist detection (Tip dist from wrist)
SCROLL_SENSITIVITY = 15
MIN_SCROLL_DISTANCE = 0.04

# STABILITY / JITTER (RELAXED)
MAX_LANDMARK_JITTER = 0.06  # Was 0.02 (Allows normal hand shake)
STABILITY_BUFFER_SIZE = 3   # Was 5

# Timing thresholds
DOUBLE_CLICK_INTERVAL = 0.4
DRAG_HOLD_TIME = 0.4
DEBOUNCE_TIME = 0.1
SCREENSHOT_COOLDOWN = 3.0

# Confidence
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MIN_GESTURE_CONFIDENCE = 0.75 # Slightly lower required confidence

# Visual settings
SHOW_LANDMARKS = True
SHOW_CONNECTIONS = True
SHOW_FRAME_REDUCTION_BOX = True
SHOW_DEBUG_OVERLAY = False

# Screenshot settings
SCREENSHOT_DIR = "screenshots"

# ============================================================================
# MediaPipe Setup
# ============================================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0

# ============================================================================
# STATE MACHINE
# ============================================================================

class MouseState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    PINCH_DETECTED = "pinch_detected"
    CLICK_READY = "click_ready"
    DRAGGING = "dragging"
    DRAG_READY = "drag_ready"

# ============================================================================
# HELPER CLASSES
# ============================================================================

class GestureConfidence:
    def __init__(self, name: str, required_frames: int, min_confidence: float = MIN_GESTURE_CONFIDENCE):
        self.name = name
        self.required_frames = required_frames
        self.min_confidence = min_confidence
        self.detection_buffer = deque(maxlen=required_frames)
        self.active = False
        self.lock_time = None
    
    def update(self, detected: bool, quality: float = 1.0):
        self.detection_buffer.append(detected and quality > 0.5)
    
    def get_confidence(self) -> float:
        if len(self.detection_buffer) == 0: return 0.0
        return sum(self.detection_buffer) / len(self.detection_buffer)
    
    def is_confident(self) -> bool:
        confidence = self.get_confidence()
        is_confident = confidence >= self.min_confidence
        
        if is_confident and not self.active:
            self.lock_time = time.time()
            self.active = True
        elif not is_confident and self.active:
            self.active = False
            self.lock_time = None
        return is_confident
    
    def reset(self):
        self.detection_buffer.clear()
        self.active = False
        self.lock_time = None
    
    def get_lock_duration(self) -> float:
        if self.lock_time:
            return time.time() - self.lock_time
        return 0.0

class LandmarkStabilityChecker:
    def __init__(self, buffer_size: int = STABILITY_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.landmark_history = deque(maxlen=buffer_size)
    
    def update(self, landmarks) -> bool:
        key_landmarks = [
            (landmarks.landmark[WRIST].x, landmarks.landmark[WRIST].y),
            (landmarks.landmark[INDEX_TIP].x, landmarks.landmark[INDEX_TIP].y),
        ]
        self.landmark_history.append(key_landmarks)
        
        if len(self.landmark_history) < self.buffer_size:
            return False
        
        max_movement = 0.0
        for i in range(len(key_landmarks)):
            positions = [frame[i] for frame in self.landmark_history]
            x_range = max(p[0] for p in positions) - min(p[0] for p in positions)
            y_range = max(p[1] for p in positions) - min(p[1] for p in positions)
            movement = np.sqrt(x_range**2 + y_range**2)
            max_movement = max(max_movement, movement)
        
        return max_movement < MAX_LANDMARK_JITTER
    
    def reset(self):
        self.landmark_history.clear()

# ============================================================================
# CALCULATION HELPERS
# ============================================================================

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_landmark_point(landmarks, index: int, frame_shape: Tuple[int, int]) -> Tuple[int, int]:
    h, w = frame_shape
    lm = landmarks.landmark[index]
    return int(lm.x * w), int(lm.y * h)

def get_landmark_normalized(landmarks, index: int) -> Tuple[float, float]:
    lm = landmarks.landmark[index]
    return lm.x, lm.y

def calculate_pinch_quality(landmarks) -> Tuple[bool, float]:
    thumb = get_landmark_normalized(landmarks, THUMB_TIP)
    index = get_landmark_normalized(landmarks, INDEX_TIP)
    distance = euclidean_distance(thumb, index)
    is_pinching = distance < CLICK_DISTANCE
    
    if is_pinching:
        quality = 1.0 - (distance / CLICK_DISTANCE)
    else:
        quality = min(1.0, (distance - CLICK_DISTANCE) / CLICK_DISTANCE)
    return is_pinching, quality

def calculate_three_finger_pinch_quality(landmarks) -> Tuple[bool, float]:
    thumb = get_landmark_normalized(landmarks, THUMB_TIP)
    index = get_landmark_normalized(landmarks, INDEX_TIP)
    middle = get_landmark_normalized(landmarks, MIDDLE_TIP)
    
    max_dist = max(
        euclidean_distance(thumb, index),
        euclidean_distance(thumb, middle),
        euclidean_distance(index, middle)
    )
    
    is_pinching = max_dist < THREE_FINGER_DISTANCE
    if is_pinching:
        quality = 1.0 - (max_dist / THREE_FINGER_DISTANCE)
    else:
        quality = min(1.0, (max_dist - THREE_FINGER_DISTANCE) / THREE_FINGER_DISTANCE)
    return is_pinching, quality

def calculate_fist_quality(landmarks) -> Tuple[bool, float]:
    """Calculate if hand is a fist (fingertips close to wrist)."""
    wrist = get_landmark_normalized(landmarks, WRIST)
    tips_indices = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    
    total_dist = 0.0
    for idx in tips_indices:
        tip = get_landmark_normalized(landmarks, idx)
        total_dist += euclidean_distance(tip, wrist)
    
    avg_dist = total_dist / len(tips_indices)
    is_fist = avg_dist < FIST_THRESHOLD
    
    if is_fist:
        quality = 1.0 - (avg_dist / FIST_THRESHOLD)
    else:
        quality = 0.0
        
    return is_fist, quality

def get_two_finger_position(landmarks) -> Optional[Tuple[float, float]]:
    index = get_landmark_normalized(landmarks, INDEX_TIP)
    middle = get_landmark_normalized(landmarks, MIDDLE_TIP)
    if euclidean_distance(index, middle) < 0.1: # Relaxed from 0.08
        return ((index[0] + middle[0]) / 2, (index[1] + middle[1]) / 2)
    return None

def map_to_screen(x: float, y: float, frame_shape: Tuple[int, int], 
                  screen_size: Tuple[int, int], reduction: float = FRAME_REDUCTION) -> Tuple[int, int]:
    h, w = frame_shape
    screen_w, screen_h = screen_size
    margin_x, margin_y = w * reduction, h * reduction
    
    x = max(margin_x, min(x, w - margin_x))
    y = max(margin_y, min(y, h - margin_y))
    
    norm_x = (x - margin_x) / (w - 2 * margin_x)
    norm_y = (y - margin_y) / (h - 2 * margin_y)
    
    return int((1 - norm_x) * screen_w), int(norm_y * screen_h)

# ============================================================================
# CONTROLLER
# ============================================================================

class HighAccuracyGestureController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.smooth_x, self.smooth_y = None, None
        self.smoothing_factor = SMOOTHING_FACTOR
        self.mouse_state = MouseState.IDLE
        
        # Gesture Trackers
        self.pinch_confidence = GestureConfidence("pinch", MIN_CLICK_FRAMES)
        self.three_finger_confidence = GestureConfidence("three_finger", MIN_GESTURE_FRAMES)
        self.screenshot_confidence = GestureConfidence("screenshot", MIN_SCREENSHOT_FRAMES)
        
        self.stability_checker = LandmarkStabilityChecker()
        
        self.last_click_time = 0
        self.click_count = 0
        self.pinch_start_time = None
        
        self.last_screenshot_time = 0
        self.screenshot_feedback = ""
        self.screenshot_feedback_time = 0
        
        self.last_scroll_y = None
        self.scroll_buffer = deque(maxlen=5)
        
        self.current_action = "Ready"
        self.gesture_feedback = {}
        self.debug_mode = SHOW_DEBUG_OVERLAY
        
        # FPS
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        if not os.path.exists(SCREENSHOT_DIR):
            os.makedirs(SCREENSHOT_DIR)
            
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.001

    def reset_smoothing(self):
        self.smooth_x, self.smooth_y = None, None

    def apply_smoothing(self, x: int, y: int) -> Tuple[int, int]:
        if self.smooth_x is None:
            self.smooth_x, self.smooth_y = x, y
        else:
            self.smooth_x = self.smoothing_factor * self.smooth_x + (1 - self.smoothing_factor) * x
            self.smooth_y = self.smoothing_factor * self.smooth_y + (1 - self.smoothing_factor) * y
        return int(self.smooth_x), int(self.smooth_y)

    def handle_mouse_movement(self, landmarks, frame_shape: Tuple[int, int], is_stable: bool):
        if not is_stable: return
        
        index_x, index_y = get_landmark_point(landmarks, INDEX_TIP, frame_shape)
        screen_x, screen_y = map_to_screen(index_x, index_y, frame_shape, (self.screen_width, self.screen_height))
        smooth_x, smooth_y = self.apply_smoothing(screen_x, screen_y)
        
        try:
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            if self.mouse_state == MouseState.IDLE:
                self.mouse_state = MouseState.MOVING
                self.current_action = "Moving"
        except: pass

    def handle_click_gestures(self, landmarks, is_stable: bool):
        if not is_stable:
            self.pinch_confidence.reset()
            return
            
        is_pinching, quality = calculate_pinch_quality(landmarks)
        self.pinch_confidence.update(is_pinching, quality)
        self.gesture_feedback['pinch'] = self.pinch_confidence.get_confidence()
        
        if self.pinch_confidence.is_confident():
            if self.pinch_start_time is None:
                self.pinch_start_time = time.time()
                self.mouse_state = MouseState.PINCH_DETECTED
                self.current_action = "Pinch detected"
            
            if self.pinch_confidence.get_lock_duration() > DRAG_HOLD_TIME and self.mouse_state != MouseState.DRAGGING:
                pyautogui.mouseDown()
                self.mouse_state = MouseState.DRAGGING
                self.current_action = "Dragging"
        else:
            if self.mouse_state == MouseState.DRAGGING:
                pyautogui.mouseUp()
                self.mouse_state = MouseState.IDLE
                self.current_action = "Drag released"
            elif self.mouse_state == MouseState.PINCH_DETECTED and self.pinch_start_time:
                if time.time() - self.pinch_start_time < DRAG_HOLD_TIME:
                    if time.time() - self.last_click_time < DOUBLE_CLICK_INTERVAL:
                        pyautogui.click(clicks=2)
                        self.current_action = "Double Click"
                    else:
                        pyautogui.click()
                        self.current_action = "Click"
                        self.last_click_time = time.time()
                self.mouse_state = MouseState.IDLE
            self.pinch_start_time = None
            self.pinch_confidence.reset()

    def handle_right_click(self, landmarks, is_stable: bool):
        if not is_stable:
            self.three_finger_confidence.reset()
            return
            
        is_pinching, quality = calculate_three_finger_pinch_quality(landmarks)
        self.three_finger_confidence.update(is_pinching, quality)
        
        if (self.three_finger_confidence.is_confident() and 
            self.mouse_state not in [MouseState.DRAGGING, MouseState.PINCH_DETECTED]):
            if self.three_finger_confidence.get_lock_duration() < 0.1:
                pyautogui.rightClick()
                self.current_action = "Right Click"

    def handle_scroll(self, landmarks, is_stable: bool):
        if not is_stable:
            self.last_scroll_y = None
            self.scroll_buffer.clear()
            return
            
        pos = get_two_finger_position(landmarks)
        if pos:
            if self.last_scroll_y is not None:
                delta = pos[1] - self.last_scroll_y
                self.scroll_buffer.append(delta)
                if len(self.scroll_buffer) >= 2: # Relaxed from 3
                    avg = sum(self.scroll_buffer) / len(self.scroll_buffer)
                    if abs(avg) > MIN_SCROLL_DISTANCE:
                        amount = int(avg * SCROLL_SENSITIVITY)
                        pyautogui.scroll(amount)
                        self.current_action = "Scrolling"
            self.last_scroll_y = pos[1]
        else:
            self.last_scroll_y = None

    def handle_screenshot(self, landmarks, frame_shape: Tuple[int, int], is_stable: bool):
        """Handle screenshot gesture (FIST)."""
        if time.time() - self.last_screenshot_time < SCREENSHOT_COOLDOWN: return
        if self.mouse_state == MouseState.DRAGGING: return
        
        # Use new FIST calculation
        is_fist, quality = calculate_fist_quality(landmarks)
        self.screenshot_confidence.update(is_fist, quality)
        self.gesture_feedback['screenshot'] = self.screenshot_confidence.get_confidence()
        
        if self.screenshot_confidence.is_confident():
            if self.screenshot_confidence.get_lock_duration() < 0.2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(SCREENSHOT_DIR, f"screenshot_{timestamp}.png")
                pyautogui.screenshot().save(filename)
                self.screenshot_feedback = f"Saved: {os.path.basename(filename)}"
                self.screenshot_feedback_time = time.time()
                self.last_screenshot_time = time.time()
                self.current_action = "Screenshot (Fist)"

    def update_fps(self):
        curr = time.time()
        delta = curr - self.last_frame_time
        if delta > 0: self.fps_history.append(1.0 / delta)
        self.last_frame_time = curr

    def get_fps(self) -> float:
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0

# ============================================================================
# UI & MAIN
# ============================================================================

def draw_ui(frame, controller, is_stable):
    h, w = frame.shape[:2]
    
    # Info Box
    cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
    
    # Status
    fps = controller.get_fps()
    stab_icon = "OK" if is_stable else "..."
    color = (0, 255, 0) if is_stable else (0, 165, 255)
    
    lines = [
        f"FPS: {fps:.1f} | Stable: {stab_icon}",
        f"Action: {controller.current_action}",
        f"State: {controller.mouse_state.value}",
        "Gestures:",
        f" Pinch: {int(controller.gesture_feedback.get('pinch',0)*100)}%",
        f" Fist:  {int(controller.gesture_feedback.get('screenshot',0)*100)}%"
    ]
    
    y = 35
    for i, line in enumerate(lines):
        c = color if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        y += 25

    # Screenshot Feedback
    if time.time() - controller.screenshot_feedback_time < 2.0:
        cv2.putText(frame, "SCREENSHOT SAVED", (w//2-150, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

def main():
    print("="*60)
    print("RESPONSIVE HAND GESTURE MOUSE (v2)")
    print("="*60)
    print("GESTURES:")
    print("  Mouse Move  : Point Index Finger")
    print("  Left Click  : Pinch (Index + Thumb)")
    print("  Right Click : 3-Finger Pinch")
    print("  Drag        : Pinch and Hold")
    print("  Scroll      : Two Fingers Up/Down")
    print("  Screenshot  : MAKE A FIST")
    print("\nKey 'd' to toggle debug overlay. 'q' to quit.")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)
    
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    
    controller = HighAccuracyGestureController()
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            controller.update_fps()
            
            is_stable = False
            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                # Draw landmarks
                if SHOW_LANDMARKS:
                    mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                
                # Check stability
                is_stable = controller.stability_checker.update(lms)
                
                # Prioritize gestures
                # We process screenshot (fist) even if not perfectly stable as clenching causes shake
                controller.handle_screenshot(lms, frame.shape[:2], True) 
                
                if is_stable:
                    controller.handle_right_click(lms, is_stable)
                    controller.handle_click_gestures(lms, is_stable)
                    controller.handle_scroll(lms, is_stable)
                    controller.handle_mouse_movement(lms, frame.shape[:2], is_stable)
                else:
                    controller.current_action = "Stabilizing..."
            else:
                controller.stability_checker.reset()
            
            draw_ui(frame, controller, is_stable)
            cv2.imshow('Hand Mouse v2', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('d'): controller.debug_mode = not controller.debug_mode

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# python hand_mouse_v2.py