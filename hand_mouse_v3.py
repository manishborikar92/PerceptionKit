#!/usr/bin/env python3
"""
JARVIS SPATIAL INTERFACE v2.0 (PROTOTYPE)
=========================================
Advanced Spatial Computing Platform with Predictive Tracking.

NEW FEATURES:
1. PREDICTIVE TRACKING: Kalman Filter implementation for "Pre-Crime" cursor prediction.
2. SPATIAL DEPTH: Z-Axis interaction (Air Tap) support.
3. DYNAMIC GESTURES: Geometric intent recognition.
4. VISUAL HAPTICS: Physics-based HUD feedback.
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
from threading import Thread, Lock
from collections import deque

# ============================================================================
# 1. ADVANCED CONFIGURATION
# ============================================================================

# Tracker Physics
PREDICTION_FACTOR = 4        # How many frames ahead to predict (The "Pre-Crime" Setting)
PROCESS_NOISE = 1e-4         # Lower = More trust in model (smoother)
MEASUREMENT_NOISE = 1e-1     # Lower = More trust in measurement (faster)

# Spatial Interaction
DEPTH_SENSITIVITY = 200      # Z-axis sensitivity for "Air Taps"
CLICK_COOLDOWN = 0.3         # Seconds between interactions

# Visuals
HUD_COLOR = (0, 255, 240)    # Tron Cyan
ALERT_COLOR = (0, 0, 255)    # Warning Red
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ============================================================================
# 2. MATHEMATICAL CORE (Kalman Filter)
# ============================================================================

class MotionPredictor:
    """
    Implements a Kalman Filter to predict hand movement trajectory.
    This reduces latency by guessing where the hand is GOING to be.
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2) # 4 state variables (x,y,dx,dy), 2 measurements (x,y)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], np.float32)
        
        # Tuning the "Crystal Ball"
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
        self.prediction = np.zeros((2, 1), np.float32)

    def update(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        
        # Extrapolate future position based on velocity
        future_x = predicted[0] + (predicted[2] * PREDICTION_FACTOR)
        future_y = predicted[1] + (predicted[3] * PREDICTION_FACTOR)
        return int(future_x), int(future_y)

# ============================================================================
# 3. PERCEPTION LAYER (Optimized)
# ============================================================================

class CamStream:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 60) # Request High FPS
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed: continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return cv2.flip(self.frame.copy(), 1)

    def stop(self):
        self.stopped = True
        self.stream.release()

# ============================================================================
# 4. SPATIAL LOGIC ENGINE
# ============================================================================

class SpatialEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,         # 1 is faster, 2 is more accurate
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.scr_w, self.scr_h = pyautogui.size()
        self.predictor = MotionPredictor()
        
        # State
        self.last_click_time = 0
        self.is_dragging = False
        self.cursor_pos = (0,0)
        self.interaction_state = "IDLE" # IDLE, HOVER, CLICK, DRAG

    def get_spatial_depth(self, landmarks):
        """
        Calculates Z-depth of Index finger relative to Wrist.
        Positive = Forward (towards screen), Negative = Backward.
        """
        wrist_z = landmarks[0].z
        index_z = landmarks[8].z
        # Invert because MP Z is negative-forward
        return (wrist_z - index_z) * 100 

    def process(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        overlay_data = {"landmarks": [], "state": "SEARCHING", "target": (0,0)}

        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                # Extract Key Coordinates
                idx_tip = lms.landmark[8]
                thumb_tip = lms.landmark[4]
                
                # 1. Coordinate Mapping
                raw_x = int(idx_tip.x * w)
                raw_y = int(idx_tip.y * h)
                
                # 2. PREDICTIVE FILTERING ("Pre-Crime")
                smooth_x, smooth_y = self.predictor.update(raw_x, raw_y)
                
                # Map to Screen (with padding for edge reaching)
                margin = 100
                screen_x = np.interp(smooth_x, [margin, w-margin], [0, self.scr_w])
                screen_y = np.interp(smooth_y, [margin, h-margin], [0, self.scr_h])
                
                # 3. INTENT RECOGNITION (Pinch vs Air Tap)
                # Calculate Euclidean Pinch Distance
                pinch_dist = math.hypot(idx_tip.x - thumb_tip.x, idx_tip.y - thumb_tip.y)
                
                # Spatial Depth (Z-Axis Push)
                depth_score = self.get_spatial_depth(lms.landmark)

                current_time = time.time()
                
                # --- LOGIC TREE ---
                
                # MODE A: PINCH (High Precision Dragging)
                if pinch_dist < 0.04: 
                    if not self.is_dragging:
                        pyautogui.mouseDown()
                        self.is_dragging = True
                        self.interaction_state = "DRAG_START"
                    else:
                        self.interaction_state = "DRAGGING"
                    
                    pyautogui.moveTo(screen_x, screen_y, duration=0)
                
                # MODE B: AIR TAP (Apple Vision Pro Style)
                # Note: We check if NOT dragging to avoid conflict
                elif depth_score > 8.0 and not self.is_dragging: # Threshold for Z-push
                     if (current_time - self.last_click_time) > CLICK_COOLDOWN:
                        pyautogui.click()
                        self.last_click_time = current_time
                        self.interaction_state = "AIR_TAP"
                
                # MODE C: HOVER
                else:
                    if self.is_dragging:
                        pyautogui.mouseUp()
                        self.is_dragging = False
                    
                    self.interaction_state = "HOVER"
                    pyautogui.moveTo(screen_x, screen_y, duration=0)

                # Data for HUD
                overlay_data["landmarks"] = lms
                overlay_data["state"] = self.interaction_state
                overlay_data["target"] = (smooth_x, smooth_y)
                overlay_data["raw"] = (raw_x, raw_y)
                
                # Only process the first detected hand for cursor control
                break 
                
        return overlay_data

# ============================================================================
# 5. HUD RENDERER (Visual Haptics)
# ============================================================================

def draw_spatial_hud(img, data):
    if not data["landmarks"]: return
    
    h, w, _ = img.shape
    cx, cy = data["target"]
    raw_x, raw_y = data["raw"]
    state = data["state"]
    
    # 1. Draw Trajectory Line (The "Ghost" Trace)
    cv2.line(img, (raw_x, raw_y), (cx, cy), (100, 100, 100), 1)
    
    # 2. Context-Aware Cursor
    if state == "DRAGGING":
        color = (0, 255, 0) # Green
        radius = 15
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.putText(img, "HOLD", (cx+20, cy), FONT, 0.5, color, 1)
        
    elif state == "AIR_TAP":
        color = (0, 0, 255) # Flash Red
        cv2.circle(img, (cx, cy), 20, color, 2)
        cv2.putText(img, "CLICK", (cx+20, cy), FONT, 0.5, color, 1)
        
    else: # HOVER
        color = HUD_COLOR
        # Sci-Fi Reticle
        cv2.circle(img, (cx, cy), 8, color, 1)
        cv2.line(img, (cx-15, cy), (cx+15, cy), color, 1)
        cv2.line(img, (cx, cy-15), (cx, cy+15), color, 1)

    # 3. Hand Skeleton with Depth Visualization
    lms = data["landmarks"]
    coords = [(int(l.x*w), int(l.y*h)) for l in lms.landmark]
    
    # Draw connections
    for conn in mp.solutions.hands.HAND_CONNECTIONS:
        p1 = coords[conn[0]]
        p2 = coords[conn[1]]
        cv2.line(img, p1, p2, (0, 50, 0), 1)
    
    # Highlight Finger Tips
    for i in [4, 8, 12, 16, 20]:
        cv2.circle(img, coords[i], 3, (0, 255, 0), -1)

    # 4. Status Bar
    cv2.rectangle(img, (0, h-40), (w, h), (0, 0, 0), -1)
    cv2.putText(img, f"MODE: {state} | LATENCY: LOW", (20, h-15), FONT, 0.6, (255, 255, 255), 1)

# ============================================================================
# 6. MAIN LOOP
# ============================================================================

def main():
    print("BOOTING SPATIAL INTERFACE v2.0...")
    print(" - Kalman Filters: ACTIVE")
    print(" - Z-Axis Depth:   ACTIVE")
    
    pyautogui.FAILSAFE = False
    camera = CamStream().start()
    time.sleep(1.0)
    
    engine = SpatialEngine()
    
    try:
        while True:
            frame = camera.read()
            if frame is None: continue
            
            # Process Spatial Logic
            hud_data = engine.process(frame)
            
            # Render Visuals
            draw_spatial_hud(frame, hud_data)
            
            cv2.imshow("JARVIS SPATIAL", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# python hand_mouse_v3.py