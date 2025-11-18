import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

class VirtualMouse:
    def __init__(self):
        # --- Configuration ---
        self.cam_width, self.cam_height = 640, 480  # Webcam resolution
        self.frame_reduction = 100  # Margin (pixels) - creates a "virtual box" for easier reach
        self.smoothening = 7  # Higher = smoother cursor, Lower = faster response (5-7 is sweet spot)
        self.click_threshold = 30 # Distance between fingers to trigger click
        
        # --- PyAutoGUI Setup ---
        pyautogui.FAILSAFE = False # Prevent crash if mouse hits corner (Use 'Ctrl+C' in terminal to kill if stuck)
        self.screen_width, self.screen_height = pyautogui.size()
        
        # --- MediaPipe Setup ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # --- State Variables ---
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self.is_clicking = False

    def calculate_distance(self, p1, p2):
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        length = math.hypot(x2 - x1, y2 - y1)
        return length, [x1, y1, x2, y2]

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, self.cam_width)
        cap.set(4, self.cam_height)
        
        print("Virtual Mouse Started.")
        print("Index Finger: Move Cursor")
        print("Pinch (Index + Thumb): Click / Drag")
        print("Press 'q' to exit.")

        prev_time = 0

        while True:
            success, img = cap.read()
            if not success: 
                continue
            
            # 1. Find Hand Landmarks
            img = cv2.flip(img, 1) # Mirror logic is essential for mouse control
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # Draw the "Active Region" box
            cv2.rectangle(img, (self.frame_reduction, self.frame_reduction), 
                         (self.cam_width - self.frame_reduction, self.cam_height - self.frame_reduction),
                         (255, 0, 255), 2)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])

                    if len(lm_list) != 0:
                        # Get coordinates of Index (8) and Thumb (4)
                        x1, y1 = lm_list[8][1:]
                        x2, y2 = lm_list[4][1:]

                        # 2. Check which fingers are up (Optional check, here we assume Index is always tracking)
                        
                        # 3. Convert Coordinates (Webcam -> Screen)
                        # We map the range of the "Active Region" to the "Screen Size"
                        # This ensures you can reach the screen edges without stretching your arm too far
                        x3 = np.interp(x1, (self.frame_reduction, self.cam_width - self.frame_reduction), (0, self.screen_width))
                        y3 = np.interp(y1, (self.frame_reduction, self.cam_height - self.frame_reduction), (0, self.screen_height))

                        # 4. Smoothen Values (Reduce Jitter)
                        # Current = Previous + (Target - Previous) / SmoothingFactor
                        self.curr_x = self.prev_x + (x3 - self.prev_x) / self.smoothening
                        self.curr_y = self.prev_y + (y3 - self.prev_y) / self.smoothening

                        # Move Mouse
                        pyautogui.moveTo(self.screen_width - self.curr_x, self.curr_y) # Invert X for mirror effect
                        
                        # Update Previous locations
                        self.prev_x, self.prev_y = self.curr_x, self.curr_y

                        # 5. Clicking Mode (Pinch Detection)
                        distance, line_info = self.calculate_distance(lm_list[4], lm_list[8])
                        cx, cy = (line_info[0] + line_info[2]) // 2, (line_info[1] + line_info[3]) // 2

                        if distance < self.click_threshold:
                            # Draw visual feedback (Green Circle = Clicked)
                            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                            
                            if not self.is_clicking:
                                pyautogui.mouseDown()
                                self.is_clicking = True
                        else:
                            # Draw visual feedback (Red Circle = Released)
                            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                            
                            if self.is_clicking:
                                pyautogui.mouseUp()
                                self.is_clicking = False

            # Frame Rate Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            cv2.imshow("PerceptionKit - Virtual Mouse", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    v_mouse = VirtualMouse()
    v_mouse.run()
    
# python virtual_mouse.py