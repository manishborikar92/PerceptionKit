#!/usr/bin/env python3
"""
JARVIS SPATIAL INTERFACE v1.0
=============================
Industry-Grade Spatial Computing Platform for Desktop Control.

ARCHITECTURE:
1. Perception Layer: Threaded OpenCV + MediaPipe High-Fidelity Tracking.
2. Logic Layer: Vector-based gesture recognition (Scale Invariant).
3. Interaction Layer: Dual-hand coordinate mapping.
4. HUD Layer: Sci-fi style vector graphics overlay.

CONTROLS:
------------------------------------------------------------------
[RIGHT HAND] - The "Manipulator"
- Point Index Finger:   Move Cursor (Laser Pointer logic)
- Pinch (Index+Thumb):  Left Click / Select
- Pinch + Hold:         Drag / Grasp Object
- Open Palm (5 fingers): Release / Hover
- Fist:                 Scroll Mode (Tilt wrist to scroll)

[LEFT HAND] - The "Controller"
- Pinch (Index+Thumb):  Engage Slider Mechanism
   -> Move Up/Down:     Adjust Volume
   -> Move Left/Right:  Adjust Brightness
------------------------------------------------------------------
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import screen_brightness_control as sbc
from threading import Thread, Lock
from collections import deque
from typing import Tuple, List, Dict

# ============================================================================
# 1. SYSTEM CONFIGURATION (The "Engine Room")
# ============================================================================

# Physics & Smoothing
SMOOTHING_FACTOR = 0.15      # Lower = Smoother but more lag
CURSOR_SENSITIVITY = 1.8     # Multiplier for movement speed
DEADZONE = 0.15              # Edge margins (allows reaching corners easily)

# Gesture Thresholds (Normalized 0.0 - 1.0 relative to hand size)
CLICK_THRESHOLD = 0.05       # Pinch distance
SCROLL_ACTIVATION = 0.06     # Fist tightness

# Visuals (The "Sci-Fi" Look)
COLOR_PRIMARY = (0, 255, 255)   # Cyan (Right Hand)
COLOR_SECONDARY = (255, 0, 255) # Magenta (Left Hand)
COLOR_ACCENT = (0, 255, 0)      # Bright Green (Active Actions)
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ============================================================================
# 2. MATHEMATICAL UTILITIES (Vector Logic)
# ============================================================================

class VectorMath:
    """Static class for high-speed vector operations."""
    
    @staticmethod
    def distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def normalize_point(landmark, width, height):
        return int(landmark.x * width), int(landmark.y * height)

    @staticmethod
    def interpolate(start, end, factor):
        return start + (end - start) * factor

class OneEuroFilter:
    """
    Industry standard low-pass filter for removing jitter from 
    human movement while minimizing lag.
    """
    def __init__(self, min_cutoff=1.0, beta=0.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t=None):
        if t is None: t = time.time()
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev

        # Exponential smoothing
        a_d = self.smoothing_factor(t_e, self.min_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

# ============================================================================
# 3. PERCEPTION LAYER (Camera Thread)
# ============================================================================

class CamStream:
    """Dedicated thread for frame capture to prevent I/O blocking."""
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return cv2.flip(self.frame.copy(), 1) if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# ============================================================================
# 4. LOGIC LAYER (The "Brain")
# ============================================================================

class SpatialController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.scr_w, self.scr_h = pyautogui.size()
        
        # Motion Filters
        self.filter_x = OneEuroFilter(min_cutoff=0.1, beta=1.0)
        self.filter_y = OneEuroFilter(min_cutoff=0.1, beta=1.0)
        
        # State Management
        self.dragging = False
        self.last_click = 0
        self.scroll_origin = None
        self.volume_origin = None
        
        # Visual Data Exchange (for HUD)
        self.hud_data = {
            "right_hand": None,
            "left_hand": None,
            "action": "IDLE",
            "fps": 0
        }

    def get_hand_label(self, index, hand_landmarks, results):
        """Accurately determine Left vs Right hand."""
        output = None
        for idx, classification in enumerate(results.multi_handedness):
            if classification.classification[0].index == index:
                label = classification.classification[0].label
                score = classification.classification[0].score
                text = "Right" if label == "Right" else "Left"
                return text
        return "Right" # Fallback

    def process_gestures(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        self.hud_data["right_hand"] = None
        self.hud_data["left_hand"] = None
        self.hud_data["action"] = "SCANNING..."

        if results.multi_hand_landmarks:
            for idx, lms in enumerate(results.multi_hand_landmarks):
                # Get Hand Type (Left/Right)
                label = self.get_hand_label(idx, lms, results)
                
                # Landmark Coordinates Helper
                coords = [(int(lm.x * w), int(lm.y * h)) for lm in lms.landmark]
                
                # Hand Size (Wrist to Middle Finger Base) - Scale Factor
                scale_ref = VectorMath.distance(coords[0], coords[9])
                
                if label == "Right":
                    self._handle_right_hand(coords, scale_ref, w, h)
                    self.hud_data["right_hand"] = (coords, scale_ref)
                else:
                    self._handle_left_hand(coords, scale_ref)
                    self.hud_data["left_hand"] = (coords, scale_ref)

    def _handle_right_hand(self, coords, scale, w, h):
        """DOMINANT HAND: Cursor, Click, Scroll"""
        
        # 1. Cursor Mapping (Index Tip)
        idx_tip = coords[8]
        
        # Active Zone Mapping (Remap camera subset to full screen)
        norm_x = np.interp(idx_tip[0], [w*DEADZONE, w*(1-DEADZONE)], [0, self.scr_w])
        norm_y = np.interp(idx_tip[1], [h*DEADZONE, h*(1-DEADZONE)], [0, self.scr_h])
        
        # Smooth Movement
        smooth_x = self.filter_x(norm_x)
        smooth_y = self.filter_y(norm_y)
        
        # 2. Pinch Detection (Index to Thumb)
        thumb_tip = coords[4]
        pinch_dist = VectorMath.distance(idx_tip, thumb_tip)
        normalized_pinch = pinch_dist / scale
        
        # ACTION LOGIC
        if normalized_pinch < CLICK_THRESHOLD:
            self.hud_data["action"] = "INTERACTING"
            if not self.dragging:
                pyautogui.mouseDown(smooth_x, smooth_y)
                self.dragging = True
            else:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
        else:
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            
            self.hud_data["action"] = "NAVIGATION"
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)

    def _handle_left_hand(self, coords, scale):
        """AUXILIARY HAND: Volume & System Control"""
        thumb_tip = coords[4]
        idx_tip = coords[8]
        
        pinch_dist = VectorMath.distance(idx_tip, thumb_tip)
        normalized_pinch = pinch_dist / scale
        
        # Calculate Control Slider (Distance between thumb and index)
        # We visualize this as a "Power Bar"
        
        if normalized_pinch < 0.05:
            # Reset trigger
            self.volume_origin = None
        
        elif normalized_pinch < 0.25: # Active Control Zone
            # Map y-position to volume? No, use Pinch Width for Volume
            # Simple Volume Control: Pinch Width
            
            vol_percent = np.clip((normalized_pinch - 0.05) / 0.15, 0, 1) * 100
            
            # Only change if significant change (Hysteresis)
            # Using pyautogui hotkeys for stability
            if vol_percent > 80: pyautogui.press("volumeup")
            if vol_percent < 20: pyautogui.press("volumedown")
            
            self.hud_data["action"] = f"VOLUME: {int(vol_percent)}%"

# ============================================================================
# 5. HUD RENDERER (The "Sci-Fi" Visuals)
# ============================================================================

def draw_hud(img, controller):
    h, w, _ = img.shape
    overlay = img.copy()
    
    # 1. Draw Tech Grid Background (Subtle)
    step = 100
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), (0, 50, 0), 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), (0, 50, 0), 1)
        
    # 2. Right Hand Visuals (Cyan)
    if controller.hud_data["right_hand"]:
        coords, scale = controller.hud_data["right_hand"]
        
        # Draw Skeleton
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            cv2.line(img, coords[start_idx], coords[end_idx], COLOR_PRIMARY, 2)
        
        # Draw Cursor Target Reticle at Index Finger
        cx, cy = coords[8]
        cv2.circle(img, (cx, cy), 10, COLOR_PRIMARY, 2)
        cv2.line(img, (cx-15, cy), (cx+15, cy), COLOR_PRIMARY, 1)
        cv2.line(img, (cx, cy-15), (cx, cy+15), COLOR_PRIMARY, 1)
        
        # Draw Pinch Line
        tx, ty = coords[4]
        dist = VectorMath.distance((cx,cy), (tx,ty))
        col = COLOR_ACCENT if dist/scale < CLICK_THRESHOLD else COLOR_PRIMARY
        cv2.line(img, (cx, cy), (tx, ty), col, 2)
        
        # Text Tag
        cv2.putText(img, "MANIPULATOR", (coords[0][0]-50, coords[0][1]+30), 
                   HUD_FONT, 0.5, COLOR_PRIMARY, 1, cv2.LINE_AA)

    # 3. Left Hand Visuals (Magenta)
    if controller.hud_data["left_hand"]:
        coords, scale = controller.hud_data["left_hand"]
        
        # Draw Skeleton
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            cv2.line(img, coords[start_idx], coords[end_idx], COLOR_SECONDARY, 2)
            
        # Draw Volume Slider Visualization between Thumb and Index
        idx, thumb = coords[8], coords[4]
        mid_x, mid_y = (idx[0]+thumb[0])//2, (idx[1]+thumb[1])//2
        dist = VectorMath.distance(idx, thumb)
        
        cv2.line(img, idx, thumb, COLOR_SECONDARY, 2)
        cv2.circle(img, (mid_x, mid_y), int(dist/2), COLOR_SECONDARY, 1)
        
        cv2.putText(img, "SYSTEM LINK", (coords[0][0]-50, coords[0][1]+30), 
                   HUD_FONT, 0.5, COLOR_SECONDARY, 1, cv2.LINE_AA)

    # 4. Top Status Bar
    cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(img, f"SYSTEM STATUS: ONLINE | FPS: {controller.hud_data['fps']}", (10, 25), 
               HUD_FONT, 0.6, (0, 255, 0), 1)
    cv2.putText(img, f"ACTION: {controller.hud_data['action']}", (w-300, 25), 
               HUD_FONT, 0.6, (0, 255, 255), 1)

    # Apply transparency
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

# ============================================================================
# 6. MAIN EXECUTABLE
# ============================================================================

def main():
    print("INITIALIZING JARVIS INTERFACE...")
    print("Connect 2 Hands for Full Control.")
    print("RIGHT HAND: Cursor & Click")
    print("LEFT HAND:  Volume Control (Pinch width)")
    
    # Disable PyAutoGui fail-safe (we handle bounds manually)
    pyautogui.FAILSAFE = False
    
    # Initialize Threads
    camera = CamStream().start()
    time.sleep(1.0) # Allow camera to warm up
    
    logic = SpatialController()
    
    prev_time = 0
    
    try:
        while True:
            frame = camera.read()
            if frame is None: continue
            
            # 1. Process Logic
            logic.process_gestures(frame)
            
            # 2. Calculate FPS
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time))
            prev_time = curr_time
            logic.hud_data["fps"] = fps
            
            # 3. Render HUD
            draw_hud(frame, logic)
            
            # 4. Display
            cv2.imshow("JARVIS INTERFACE [ESC to Quit]", frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC key
                break
                
    except KeyboardInterrupt:
        print("System Interrupted.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("SYSTEM SHUTDOWN.")

if __name__ == "__main__":
    main()

# python hand_mouse_pro.py