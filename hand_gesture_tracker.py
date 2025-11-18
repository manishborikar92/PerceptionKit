import cv2
import mediapipe as mp

class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize the Hands model
        # min_detection_confidence: Threshold for initial hand detection
        # min_tracking_confidence: Threshold for tracking after detection
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for fingertips and finger PIP joints (middle joints)
        self.tip_ids = [4, 8, 12, 16, 20]
        
    def count_fingers(self, lm_list):
        """
        Determines which fingers are open based on landmark coordinates.
        Returns a list of 5 integers (1 for open, 0 for closed).
        """
        fingers = []
        
        # Thumb detection (requires special logic compared to other fingers)
        # We check if the thumb tip (4) is to the right or left of the joint (3)
        # This logic assumes the palm is facing the camera.
        if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1) # Thumb is open
        else:
            fingers.append(0) # Thumb is closed

        # 4 Fingers (Index, Middle, Ring, Pinky)
        # Check if fingertip is above the middle joint (PIP)
        # Note: In OpenCV, Y coordinates increase downwards, so "above" means tip.y < pip.y
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def determine_gesture(self, fingers):
        """
        Classifies the gesture based on the array of open fingers.
        """
        # fingers list format: [Thumb, Index, Middle, Ring, Pinky]
        
        if sum(fingers) == 0:
            return "Fist / Rock"
        elif sum(fingers) == 5:
            return "Open Palm / Paper"
        elif fingers == [0, 1, 1, 0, 0]:
            return "Peace / Scissors"
        elif fingers == [1, 0, 0, 0, 0]:
            return "Thumbs Up" # (Depending on orientation this might need tweaking)
        elif fingers == [0, 1, 0, 0, 0]:
            return "Pointing"
        elif fingers == [0, 1, 0, 0, 1]:
            return "Rock On"
        elif fingers == [1, 1, 0, 0, 1]:
            return "Spider-Man"
        else:
            return None

    def run(self):
        cap = cv2.VideoCapture(0) # 0 is usually the default webcam
        
        print("Starting Hand Gesture Recognition. Press 'q' to exit.")

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip image horizontally for natural selfie-view interaction
            img = cv2.flip(img, 1)
            
            # Convert BGR image to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.hands.process(img_rgb)
            
            # If hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks and connections
                    self.mp_drawing.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )

                    # Extract landmark positions
                    lm_list = []
                    h, w, c = img.shape
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])

                    # Analyze fingers and gestures
                    if len(lm_list) != 0:
                        fingers_state = self.count_fingers(lm_list)
                        gesture = self.determine_gesture(fingers_state)
                        
                        # Get coordinates of the wrist to display text near the hand
                        wrist_x, wrist_y = lm_list[0][1], lm_list[0][2]

                        # Display gesture text
                        if gesture:
                            cv2.putText(img, gesture, (wrist_x - 50, wrist_y - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        
                        # Optional: Display finger count
                        finger_count = sum(fingers_state)
                        # cv2.putText(img, f"Count: {finger_count}", (10, 50), 
                        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display the final image
            cv2.imshow('MediaPipe Hand Gesture', img)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = HandGestureRecognizer()
    recognizer.run()

# python hand_gesture_tracker.py