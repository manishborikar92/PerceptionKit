#!/usr/bin/env python3
"""
Hand Gesture Mouse Control System
==================================

A production-quality webcam-based hand gesture recognition system for controlling
the OS mouse cursor, clicks, scrolls, and screenshots using MediaPipe Hands.

REQUIREMENTS:
    pip install opencv-python mediapipe pyautogui numpy

OS PERMISSIONS:
    - macOS: System Preferences > Security & Privacy > Accessibility
              Add Terminal or your Python app to allowed apps
    - Windows: Should work without special permissions
    - Linux: May need to install python3-tk python3-dev

QUICK START:
    python hand_mouse.py

CONTROLS:
    - Move index finger to control cursor
    - Pinch (thumb + index) for left click
    - Pinch twice quickly for double click
    - Pinch and hold to drag
    - Three-finger pinch (thumb + index + middle) for right click
    - Two fingers (index + middle) vertical motion to scroll
    - Join all 5 fingertips together for screenshot
    
KEYBOARD COMMANDS:
    q - Quit
    r - Reset smoothing
    c - Calibration mode (follow on-screen instructions)
    s - Save debug frame
    + - Increase smoothing
    - - Decrease smoothing

TUNING TIPS:
    - Adjust CLICK_DISTANCE if pinch detection is too sensitive/insensitive
    - Modify FRAME_REDUCTION to change screen edge accessibility
    - Tune SMOOTHING_FACTOR for cursor stability (0.0=instant, 0.9=very smooth)
    - Adjust FINGERTIP_JOIN_THRESHOLD for screenshot gesture sensitivity

DESIGN CHOICES:
    - Right click uses 3-finger pinch for ergonomic accessibility
    - Screenshot gesture requires all fingertips to prevent accidental triggers
    - Exponential moving average smoothing balances responsiveness and stability
    - Debouncing prevents gesture flicker and accidental triggers
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from typing import Tuple, Optional, List
import time
from datetime import datetime
import os
from collections import deque

# ============================================================================
# CONFIGURATION CONSTANTS - Tune these for your setup
# ============================================================================

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Screen mapping - reduce frame area to make edges easier to reach
# 0.0 = use full frame, 0.2 = use inner 80% of frame
FRAME_REDUCTION = 0.15  # 15% margin on each side

# Mouse smoothing (exponential moving average)
# 0.0 = no smoothing (instant), 0.9 = heavy smoothing (laggy)
SMOOTHING_FACTOR = 0.5

# Gesture detection thresholds
CLICK_DISTANCE = 0.04  # Distance threshold for pinch (normalized 0-1)
THREE_FINGER_DISTANCE = 0.06  # Threshold for 3-finger pinch (right click)
FINGERTIP_JOIN_THRESHOLD = 40  # Max pixel distance for screenshot gesture
SCROLL_SENSITIVITY = 20  # Pixels per scroll step
MIN_SCROLL_DISTANCE = 0.03  # Minimum vertical movement to register scroll

# Timing thresholds (seconds)
DOUBLE_CLICK_INTERVAL = 0.5  # Max time between clicks for double-click
DRAG_HOLD_TIME = 0.3  # Hold pinch this long to start drag
DEBOUNCE_TIME = 0.05  # Stable detection time before triggering action
SCREENSHOT_COOLDOWN = 2.0  # Prevent rapid screenshot triggers

# Detection confidence
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Visual settings
SHOW_LANDMARKS = True
SHOW_CONNECTIONS = True
SHOW_FRAME_REDUCTION_BOX = True

# Screenshot settings
SCREENSHOT_DIR = "screenshots"

# ============================================================================
# MediaPipe Setup
# ============================================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark indices for fingertips
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_landmark_point(landmarks, index: int, frame_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert normalized landmark coordinates to pixel coordinates.
    
    Args:
        landmarks: MediaPipe hand landmarks
        index: Landmark index
        frame_shape: (height, width) of the frame
    
    Returns:
        (x, y) pixel coordinates
    """
    h, w = frame_shape
    lm = landmarks.landmark[index]
    return int(lm.x * w), int(lm.y * h)

def get_landmark_normalized(landmarks, index: int) -> Tuple[float, float]:
    """Get normalized (0-1) landmark coordinates."""
    lm = landmarks.landmark[index]
    return lm.x, lm.y

def is_pinching(landmarks, distance_threshold: float = CLICK_DISTANCE) -> bool:
    """
    Detect pinch gesture between thumb and index finger.
    
    Args:
        landmarks: MediaPipe hand landmarks
        distance_threshold: Normalized distance threshold
    
    Returns:
        True if pinching detected
    """
    thumb = get_landmark_normalized(landmarks, THUMB_TIP)
    index = get_landmark_normalized(landmarks, INDEX_TIP)
    distance = euclidean_distance(thumb, index)
    return distance < distance_threshold

def is_three_finger_pinch(landmarks, distance_threshold: float = THREE_FINGER_DISTANCE) -> bool:
    """
    Detect three-finger pinch (thumb + index + middle) for right click.
    
    Args:
        landmarks: MediaPipe hand landmarks
        distance_threshold: Normalized distance threshold
    
    Returns:
        True if three-finger pinch detected
    """
    thumb = get_landmark_normalized(landmarks, THUMB_TIP)
    index = get_landmark_normalized(landmarks, INDEX_TIP)
    middle = get_landmark_normalized(landmarks, MIDDLE_TIP)
    
    # All three fingertips must be close to each other
    dist_thumb_index = euclidean_distance(thumb, index)
    dist_thumb_middle = euclidean_distance(thumb, middle)
    dist_index_middle = euclidean_distance(index, middle)
    
    return (dist_thumb_index < distance_threshold and 
            dist_thumb_middle < distance_threshold and 
            dist_index_middle < distance_threshold)

def all_fingertips_joined(landmarks, frame_shape: Tuple[int, int], 
                         threshold: int = FINGERTIP_JOIN_THRESHOLD) -> bool:
    """
    Detect if all five fingertips are joined together for screenshot.
    
    Args:
        landmarks: MediaPipe hand landmarks
        frame_shape: (height, width) of the frame
        threshold: Maximum pixel distance between any two fingertips
    
    Returns:
        True if all fingertips are within threshold distance
    """
    fingertip_indices = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    fingertips = [get_landmark_point(landmarks, idx, frame_shape) for idx in fingertip_indices]
    
    # Check maximum pairwise distance
    max_distance = 0
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            dist = euclidean_distance(fingertips[i], fingertips[j])
            max_distance = max(max_distance, dist)
    
    return max_distance < threshold

def get_two_finger_position(landmarks) -> Optional[Tuple[float, float]]:
    """
    Get average position of index and middle fingers for scrolling.
    
    Returns:
        (x, y) normalized coordinates or None if not detected
    """
    index = get_landmark_normalized(landmarks, INDEX_TIP)
    middle = get_landmark_normalized(landmarks, MIDDLE_TIP)
    
    # Check if fingers are close together (scrolling gesture)
    distance = euclidean_distance(index, middle)
    if distance < 0.1:  # Fingers should be reasonably close
        return ((index[0] + middle[0]) / 2, (index[1] + middle[1]) / 2)
    return None

def map_to_screen(x: float, y: float, frame_shape: Tuple[int, int], 
                  screen_size: Tuple[int, int], reduction: float = FRAME_REDUCTION) -> Tuple[int, int]:
    """
    Map camera coordinates to screen coordinates with frame reduction.
    
    Frame reduction creates an inner rectangle that maps to the full screen,
    making screen edges easier to reach.
    
    Args:
        x, y: Pixel coordinates in frame
        frame_shape: (height, width) of camera frame
        screen_size: (width, height) of screen
        reduction: Fraction to reduce frame (0.0 = no reduction, 0.5 = half)
    
    Returns:
        (screen_x, screen_y) screen coordinates
    """
    h, w = frame_shape
    screen_w, screen_h = screen_size
    
    # Calculate reduced frame boundaries
    margin_x = w * reduction
    margin_y = h * reduction
    
    # Map from reduced frame to screen
    # Clamp to reduced boundaries
    x = max(margin_x, min(x, w - margin_x))
    y = max(margin_y, min(y, h - margin_y))
    
    # Normalize to 0-1 within reduced frame
    norm_x = (x - margin_x) / (w - 2 * margin_x)
    norm_y = (y - margin_y) / (h - 2 * margin_y)
    
    # Map to screen (flip x for mirror effect)
    screen_x = int((1 - norm_x) * screen_w)
    screen_y = int(norm_y * screen_h)
    
    return screen_x, screen_y

# ============================================================================
# MAIN GESTURE CONTROLLER CLASS
# ============================================================================

class GestureMouseController:
    """Main controller for hand gesture-based mouse control."""
    
    def __init__(self):
        """Initialize the controller with default state."""
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Smoothing
        self.smooth_x = None
        self.smooth_y = None
        self.smoothing_factor = SMOOTHING_FACTOR
        
        # Gesture state
        self.is_dragging = False
        self.pinch_start_time = None
        self.last_click_time = 0
        self.click_count = 0
        
        # Screenshot state
        self.last_screenshot_time = 0
        self.screenshot_feedback = ""
        self.screenshot_feedback_time = 0
        
        # Scroll state
        self.last_scroll_y = None
        self.scroll_debounce = deque(maxlen=3)  # Average over last 3 frames
        
        # Debouncing for stable gesture detection
        self.gesture_stable_time = {}
        
        # Calibration state
        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_complete = False
        
        # Status display
        self.current_action = "Ready"
        
        # FPS calculation
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Ensure screenshot directory exists
        if not os.path.exists(SCREENSHOT_DIR):
            os.makedirs(SCREENSHOT_DIR)
        
        # Configure pyautogui
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.001  # Minimum pause between actions
        
        print("GestureMouseController initialized")
        print(f"Screen size: {self.screen_width}x{self.screen_height}")
    
    def reset_smoothing(self):
        """Reset cursor position smoothing."""
        self.smooth_x = None
        self.smooth_y = None
        print("Smoothing reset")
    
    def apply_smoothing(self, x: int, y: int) -> Tuple[int, int]:
        """
        Apply exponential moving average smoothing to coordinates.
        
        Args:
            x, y: Raw screen coordinates
        
        Returns:
            Smoothed (x, y) coordinates
        """
        if self.smooth_x is None:
            self.smooth_x = x
            self.smooth_y = y
        else:
            self.smooth_x = self.smoothing_factor * self.smooth_x + (1 - self.smoothing_factor) * x
            self.smooth_y = self.smoothing_factor * self.smooth_y + (1 - self.smoothing_factor) * y
        
        return int(self.smooth_x), int(self.smooth_y)
    
    def is_gesture_stable(self, gesture_name: str, duration: float = DEBOUNCE_TIME) -> bool:
        """
        Check if a gesture has been stable for the required duration.
        
        Args:
            gesture_name: Unique identifier for the gesture
            duration: Required stable duration in seconds
        
        Returns:
            True if gesture has been stable long enough
        """
        current_time = time.time()
        
        if gesture_name not in self.gesture_stable_time:
            self.gesture_stable_time[gesture_name] = current_time
            return False
        
        elapsed = current_time - self.gesture_stable_time[gesture_name]
        return elapsed >= duration
    
    def reset_gesture_stability(self, gesture_name: str):
        """Reset stability tracking for a gesture."""
        if gesture_name in self.gesture_stable_time:
            del self.gesture_stable_time[gesture_name]
    
    def handle_mouse_movement(self, landmarks, frame_shape: Tuple[int, int]):
        """
        Update cursor position based on index fingertip.
        
        Args:
            landmarks: MediaPipe hand landmarks
            frame_shape: (height, width) of camera frame
        """
        index_x, index_y = get_landmark_point(landmarks, INDEX_TIP, frame_shape)
        screen_x, screen_y = map_to_screen(index_x, index_y, frame_shape, 
                                          (self.screen_width, self.screen_height))
        
        # Apply smoothing
        smooth_x, smooth_y = self.apply_smoothing(screen_x, screen_y)
        
        try:
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            if not self.is_dragging:
                self.current_action = "Moving"
        except Exception as e:
            print(f"Error moving mouse: {e}")
    
    def handle_click_gestures(self, landmarks):
        """
        Handle pinch gestures for left click, double click, and drag.
        
        Args:
            landmarks: MediaPipe hand landmarks
        """
        current_time = time.time()
        pinching = is_pinching(landmarks)
        
        if pinching:
            if not self.is_gesture_stable("pinch", DEBOUNCE_TIME):
                return
            
            # Start tracking pinch
            if self.pinch_start_time is None:
                self.pinch_start_time = current_time
            
            pinch_duration = current_time - self.pinch_start_time
            
            # Check if should start dragging
            if pinch_duration > DRAG_HOLD_TIME and not self.is_dragging:
                try:
                    pyautogui.mouseDown()
                    self.is_dragging = True
                    self.current_action = "Dragging"
                except Exception as e:
                    print(f"Error starting drag: {e}")
        
        else:  # Pinch released
            self.reset_gesture_stability("pinch")
            
            if self.pinch_start_time is not None:
                pinch_duration = current_time - self.pinch_start_time
                
                # Release drag
                if self.is_dragging:
                    try:
                        pyautogui.mouseUp()
                        self.is_dragging = False
                        self.current_action = "Drag released"
                    except Exception as e:
                        print(f"Error releasing drag: {e}")
                
                # Quick pinch = click
                elif pinch_duration < DRAG_HOLD_TIME:
                    # Check for double click
                    time_since_last_click = current_time - self.last_click_time
                    
                    if time_since_last_click < DOUBLE_CLICK_INTERVAL and self.click_count == 1:
                        try:
                            pyautogui.click(clicks=2)
                            self.current_action = "Double Click"
                            self.click_count = 0
                        except Exception as e:
                            print(f"Error double clicking: {e}")
                    else:
                        try:
                            pyautogui.click()
                            self.current_action = "Click"
                            self.click_count = 1
                            self.last_click_time = current_time
                        except Exception as e:
                            print(f"Error clicking: {e}")
                
                self.pinch_start_time = None
    
    def handle_right_click(self, landmarks):
        """
        Handle three-finger pinch for right click.
        
        Args:
            landmarks: MediaPipe hand landmarks
        """
        if is_three_finger_pinch(landmarks):
            if self.is_gesture_stable("right_click", DEBOUNCE_TIME):
                try:
                    pyautogui.rightClick()
                    self.current_action = "Right Click"
                    self.reset_gesture_stability("right_click")
                except Exception as e:
                    print(f"Error right clicking: {e}")
        else:
            self.reset_gesture_stability("right_click")
    
    def handle_scroll(self, landmarks):
        """
        Handle two-finger vertical scrolling.
        
        Args:
            landmarks: MediaPipe hand landmarks
        """
        two_finger_pos = get_two_finger_position(landmarks)
        
        if two_finger_pos:
            current_y = two_finger_pos[1]
            
            if self.last_scroll_y is not None:
                delta_y = current_y - self.last_scroll_y
                
                # Add to debounce buffer
                self.scroll_debounce.append(delta_y)
                
                # Calculate average movement
                if len(self.scroll_debounce) >= 2:
                    avg_delta = sum(self.scroll_debounce) / len(self.scroll_debounce)
                    
                    # Only scroll if movement is significant
                    if abs(avg_delta) > MIN_SCROLL_DISTANCE:
                        scroll_amount = int(avg_delta * SCROLL_SENSITIVITY)
                        try:
                            pyautogui.scroll(scroll_amount)
                            self.current_action = f"Scrolling {'up' if scroll_amount > 0 else 'down'}"
                        except Exception as e:
                            print(f"Error scrolling: {e}")
            
            self.last_scroll_y = current_y
        else:
            self.last_scroll_y = None
            self.scroll_debounce.clear()
    
    def handle_screenshot(self, landmarks, frame_shape: Tuple[int, int]):
        """
        Handle screenshot gesture (all fingertips joined).
        
        Args:
            landmarks: MediaPipe hand landmarks
            frame_shape: (height, width) of camera frame
        """
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_screenshot_time < SCREENSHOT_COOLDOWN:
            return
        
        if all_fingertips_joined(landmarks, frame_shape):
            if self.is_gesture_stable("screenshot", DEBOUNCE_TIME * 2):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SCREENSHOT_DIR, f"screenshot_{timestamp}.png")
                    screenshot = pyautogui.screenshot()
                    screenshot.save(filename)
                    
                    self.screenshot_feedback = f"Screenshot saved: {filename}"
                    self.screenshot_feedback_time = current_time
                    self.last_screenshot_time = current_time
                    self.current_action = "Screenshot captured"
                    
                    print(self.screenshot_feedback)
                    self.reset_gesture_stability("screenshot")
                except Exception as e:
                    print(f"Error taking screenshot: {e}")
        else:
            self.reset_gesture_stability("screenshot")
    
    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        if delta > 0:
            fps = 1.0 / delta
            self.fps_history.append(fps)
        self.last_frame_time = current_time
    
    def get_fps(self) -> float:
        """Get average FPS."""
        if len(self.fps_history) > 0:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def draw_landmarks_and_connections(frame, hand_landmarks):
    """Draw hand landmarks and connections on frame."""
    if SHOW_LANDMARKS:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

def draw_frame_reduction_box(frame, reduction: float = FRAME_REDUCTION):
    """Draw the reduced frame mapping rectangle."""
    if SHOW_FRAME_REDUCTION_BOX:
        h, w = frame.shape[:2]
        margin_x = int(w * reduction)
        margin_y = int(h * reduction)
        
        cv2.rectangle(frame, 
                     (margin_x, margin_y), 
                     (w - margin_x, h - margin_y),
                     (0, 255, 0), 2)
        
        # Add label
        cv2.putText(frame, "Control Zone", 
                   (margin_x + 10, margin_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def draw_ui_overlay(frame, controller: GestureMouseController):
    """Draw UI overlay with status, FPS, and feedback."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Status text
    fps = controller.get_fps()
    texts = [
        f"FPS: {fps:.1f}",
        f"Action: {controller.current_action}",
        f"Smoothing: {controller.smoothing_factor:.2f}",
        f"Press 'q' to quit | 'r' to reset | 'c' to calibrate"
    ]
    
    y_offset = 35
    for text in texts:
        cv2.putText(frame, text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
    
    # Screenshot feedback
    current_time = time.time()
    if controller.screenshot_feedback and (current_time - controller.screenshot_feedback_time < 3.0):
        # Flash effect
        alpha = max(0, 1.0 - (current_time - controller.screenshot_feedback_time) / 3.0)
        color = (0, int(255 * alpha), 0)
        
        cv2.putText(frame, "SCREENSHOT SAVED!", 
                   (w // 2 - 200, h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.putText(frame, controller.screenshot_feedback.split('/')[-1], 
                   (w // 2 - 200, h // 2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ============================================================================
# CALIBRATION MODE
# ============================================================================

def handle_calibration(frame, controller: GestureMouseController, landmarks, frame_shape):
    """
    Handle calibration mode for custom frame reduction.
    
    User places index finger at 4 corners to define control zone.
    """
    if not controller.calibration_mode:
        return
    
    h, w = frame_shape
    
    # Draw instructions
    instructions = [
        "CALIBRATION MODE",
        f"Step {len(controller.calibration_points) + 1}/4: Place index finger at corner",
        "Corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left",
        "Press SPACE to capture point, ESC to cancel"
    ]
    
    y_offset = h - 150
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (20, y_offset + i * 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw current calibration points
    for i, point in enumerate(controller.calibration_points):
        cv2.circle(frame, point, 10, (0, 255, 0), -1)
        cv2.putText(frame, str(i + 1), (point[0] + 15, point[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show current index position
    if landmarks:
        index_x, index_y = get_landmark_point(landmarks, INDEX_TIP, frame_shape)
        cv2.circle(frame, (index_x, index_y), 15, (255, 0, 0), 3)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main application loop."""
    print("=" * 60)
    print("Hand Gesture Mouse Control")
    print("=" * 60)
    print("\nStarting camera...")
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Track one hand by default
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    
    # Initialize controller
    controller = GestureMouseController()
    
    print("\nSystem ready!")
    print("\nGESTURES:")
    print("  - Move index finger: Control cursor")
    print("  - Pinch (thumb + index): Left click")
    print("  - Pinch twice quickly: Double click")
    print("  - Pinch and hold: Drag")
    print("  - Three-finger pinch: Right click")
    print("  - Two fingers vertical: Scroll")
    print("  - Join all fingertips: Screenshot")
    print("\nKEYS:")
    print("  q - Quit")
    print("  r - Reset smoothing")
    print("  c - Calibration mode")
    print("  s - Save debug frame")
    print("  +/- - Adjust smoothing")
    print("\n" + "=" * 60 + "\n")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = hands.process(frame_rgb)
            
            # Update FPS
            controller.update_fps()
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # First hand
                frame_shape = (frame.shape[0], frame.shape[1])
                
                # Draw landmarks
                draw_landmarks_and_connections(frame, hand_landmarks)
                
                # Handle calibration mode
                if controller.calibration_mode:
                    handle_calibration(frame, controller, hand_landmarks, frame_shape)
                else:
                    # Normal operation - process all gestures
                    
                    # Priority 1: Screenshot (requires all fingers, no overlap with other gestures)
                    controller.handle_screenshot(hand_landmarks, frame_shape)
                    
                    # Priority 2: Right click (3-finger pinch)
                    controller.handle_right_click(hand_landmarks)
                    
                    # Priority 3: Left click/drag (2-finger pinch)
                    controller.handle_click_gestures(hand_landmarks)
                    
                    # Priority 4: Scroll (two-finger motion)
                    controller.handle_scroll(hand_landmarks)
                    
                    # Priority 5: Mouse movement (always active)
                    controller.handle_mouse_movement(hand_landmarks, frame_shape)
            else:
                controller.current_action = "No hand detected"
            
            # Draw UI overlays
            draw_frame_reduction_box(frame)
            draw_ui_overlay(frame, controller)
            
            # Display frame
            cv2.imshow('Hand Gesture Mouse Control', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                controller.reset_smoothing()
            elif key == ord('c'):
                controller.calibration_mode = not controller.calibration_mode
                controller.calibration_points = []
                print(f"Calibration mode: {'ON' if controller.calibration_mode else 'OFF'}")
            elif key == ord('s'):
                debug_filename = f"debug_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(debug_filename, frame)
                print(f"Debug frame saved: {debug_filename}")
            elif key == ord('+') or key == ord('='):
                controller.smoothing_factor = min(0.95, controller.smoothing_factor + 0.05)
                print(f"Smoothing: {controller.smoothing_factor:.2f}")
            elif key == ord('-') or key == ord('_'):
                controller.smoothing_factor = max(0.0, controller.smoothing_factor - 0.05)
                print(f"Smoothing: {controller.smoothing_factor:.2f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        
        # Release drag if active
        if controller.is_dragging:
            try:
                pyautogui.mouseUp()
            except:
                pass
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        print("Done!")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check permissions reminder
    print("\n" + "!" * 60)
    print("IMPORTANT: OS Permissions Required")
    print("!" * 60)
    print("\nmacOS: System Preferences > Security & Privacy > Accessibility")
    print("       Add Terminal or Python to allowed apps")
    print("\nWindows: Should work without special permissions")
    print("\nLinux: May need: sudo apt-get install python3-tk python3-dev")
    print("\n" + "!" * 60 + "\n")
    
    input("Press ENTER to continue...")
    
    main()
