import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

class AdvancedVirtualMouse:
    def __init__(self):
        # --- Configuration ---
        self.cam_width, self.cam_height = 640, 480
        self.frame_reduction = 100  # Active Region margin (Purple Box)
        self.smoothening = 5        # Higher = smoother cursor, Lower = faster response
        
        # --- Click Thresholds ---
        self.click_distance = 30    # Distance between fingers to trigger click
        self.drag_threshold = 30    # Distance between index & thumb to trigger drag
        
        # --- PyAutoGUI Setup ---
        pyautogui.FAILSAFE = False
        self.screen_width, self.screen_height = pyautogui.size()
        
        # --- MediaPipe Setup ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky

        # --- State Variables ---
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self.is_dragging = False
        self.last_click_time = 0

    def get_fingers(self, lm_list):
        """Returns list of 5 booleans: [Thumb, Index, Middle, Ring, Pinky]"""
        fingers = []
        
        # Thumb (Check X coord relative to joint for Left/Right hand logic)
        # Assuming Right Hand for simplicity, works for Left if flipped
        if lm_list[self.tip_ids[0]][1] < lm_list[self.tip_ids[0] - 1][1]: 
             fingers.append(1) # Thumb Open 
        else:
             fingers.append(0)

        # 4 Fingers (Check Y coord - Tip above Knuckle)
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img=None, color=(255, 0, 255)):
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        
        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
            cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)
            
        return length, [x1, y1, x2, y2, cx, cy]

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, self.cam_width)
        cap.set(4, self.cam_height)
        
        print("Advanced Virtual Mouse Started.")
        print("- Index only: Move")
        print("- Index + Middle (Touch): Left Click")
        print("- Index + Middle + Ring (Touch): Right Click")
        print("- Index + Thumb (Pinch): Drag")
        print("- Thumb + Pinky: Scroll")

        while True:
            success, img = cap.read()
            if not success: continue
            
            # Flip image for mirror view
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # Draw Active Region (Purple Box)
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
                        # 1. Identify Fingers Up
                        fingers = self.get_fingers(lm_list)
                        
                        # Coordinates of Index (8) and Middle (12)
                        x1, y1 = lm_list[8][1:]
                        x2, y2 = lm_list[12][1:]
                        
                        # --- MODE 1: MOVEMENT (Only Index Up) ---
                        # Ensure Middle, Ring, Pinky are down for pure movement
                        if fingers[1] == 1 and fingers[2] == 0:
                            
                            # Convert Coordinates (Interpolation)
                            x3 = np.interp(x1, (self.frame_reduction, self.cam_width - self.frame_reduction), (0, self.screen_width))
                            y3 = np.interp(y1, (self.frame_reduction, self.cam_height - self.frame_reduction), (0, self.screen_height))
                            
                            # Smoothening Logic
                            self.curr_x = self.prev_x + (x3 - self.prev_x) / self.smoothening
                            self.curr_y = self.prev_y + (y3 - self.prev_y) / self.smoothening
                            
                            # Move Mouse
                            pyautogui.moveTo(self.curr_x, self.curr_y)
                            
                            self.prev_x, self.prev_y = self.curr_x, self.curr_y
                            
                            # Visual Feedback
                            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                            cv2.putText(img, "Moving", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

                        # --- MODE 2: LEFT CLICK (Index + Middle Up) ---
                        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                            # Find distance between Index and Middle
                            length, line_info = self.find_distance(lm_list[8], lm_list[12], img)
                            
                            # Click if fingers touch
                            if length < self.click_distance:
                                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                                # Debounce click (prevent double clicks)
                                if time.time() - self.last_click_time > 0.3:
                                    pyautogui.click()
                                    self.last_click_time = time.time()
                                    cv2.putText(img, "Left Click", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                        # --- MODE 3: RIGHT CLICK (Index + Middle + Ring Up) ---
                        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                             cv2.putText(img, "Right Click Mode", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                             # Use distance between Middle and Ring
                             length, _ = self.find_distance(lm_list[12], lm_list[16], img, color=(0,0,255))
                             if length < self.click_distance:
                                 if time.time() - self.last_click_time > 0.5:
                                    pyautogui.rightClick()
                                    self.last_click_time = time.time()
                        
                        # --- MODE 4: DRAG & DROP (Index + Thumb Pinch) ---
                        # Always check for drag regardless of other fingers for better usability
                        dist_drag, _ = self.find_distance(lm_list[4], lm_list[8])
                        if dist_drag < self.drag_threshold:
                             if not self.is_dragging:
                                 pyautogui.mouseDown()
                                 self.is_dragging = True
                             
                             # Allow movement while dragging
                             x3 = np.interp(x1, (self.frame_reduction, self.cam_width - self.frame_reduction), (0, self.screen_width))
                             y3 = np.interp(y1, (self.frame_reduction, self.cam_height - self.frame_reduction), (0, self.screen_height))
                             
                             self.curr_x = self.prev_x + (x3 - self.prev_x) / self.smoothening
                             self.curr_y = self.prev_y + (y3 - self.prev_y) / self.smoothening
                             
                             pyautogui.moveTo(self.curr_x, self.curr_y)
                             self.prev_x, self.prev_y = self.curr_x, self.curr_y
                             
                             cv2.putText(img, "Dragging", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        elif self.is_dragging and dist_drag > self.drag_threshold:
                             pyautogui.mouseUp()
                             self.is_dragging = False

                        # --- MODE 5: SCROLL (Thumb + Pinky Up) ---
                        if fingers[0] == 1 and fingers[4] == 1 and fingers[1] == 0:
                            cv2.putText(img, "Scroll Mode", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                            # Check hand height
                            if lm_list[9][2] < self.cam_height // 2 - 50: # Upper part of screen
                                pyautogui.scroll(30)
                            elif lm_list[9][2] > self.cam_height // 2 + 50: # Lower part of screen
                                pyautogui.scroll(-30)

            cv2.imshow("PerceptionKit - Advanced Mouse", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    v_mouse = AdvancedVirtualMouse()
    v_mouse.run()
    
# python virtual_mouse.py