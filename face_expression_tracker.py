import cv2
import mediapipe as mp
import math

class FaceExpressionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Face Mesh
        # refine_landmarks=True enables iris tracking (though we aren't using iris specifically here, it adds detail)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Landmark indices for eyes and mouth
        # Indices: [Left, Right, Top, Bottom] or [Left, Right, Top1, Top2, Bottom1, Bottom2]
        self.LEFT_EYE = [33, 133, 160, 158, 144, 153] 
        self.RIGHT_EYE = [362, 263, 385, 387, 380, 373]
        self.MOUTH = [61, 291, 13, 14] # Left, Right, Upper Lip, Lower Lip

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1.x, point1.y
        x2, y2 = point2.x, point2.y
        return math.hypot(x2 - x1, y2 - y1)

    def get_blink_ratio(self, img, landmarks, indices):
        # Horizontal line (width)
        p_left = landmarks[indices[0]]
        p_right = landmarks[indices[1]]
        
        # Vertical line (height) - using average of two vertical points for stability
        p_top = landmarks[indices[3]] # Using one top point
        p_bottom = landmarks[indices[5]] # Using one bottom point
        
        # Calculate distances
        horizontal_dist = self.euclidean_distance(p_left, p_right)
        vertical_dist = self.euclidean_distance(p_top, p_bottom)
        
        # Avoid division by zero
        if horizontal_dist == 0:
            return 0
            
        ratio = vertical_dist / horizontal_dist
        return ratio

    def get_mouth_ratio(self, landmarks, indices):
        # Mouth width
        p_left = landmarks[indices[0]]
        p_right = landmarks[indices[1]]
        
        # Mouth height (inner lip mostly)
        p_top = landmarks[indices[2]]
        p_bottom = landmarks[indices[3]]
        
        width = self.euclidean_distance(p_left, p_right)
        height = self.euclidean_distance(p_top, p_bottom)
        
        if width == 0:
            return 0
            
        return height / width

    def run(self):
        cap = cv2.VideoCapture(0)
        print("Starting Face Expression Tracker. Press 'q' to exit.")

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip and convert
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(img_rgb)
            h, w, _ = img.shape

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the mesh on the face
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Draw eyes and mouth contours
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

                    # --- Expression Logic ---
                    landmarks = face_landmarks.landmark
                    
                    # 1. Calculate Ratios
                    left_eye_ratio = self.get_blink_ratio(img, landmarks, self.LEFT_EYE)
                    right_eye_ratio = self.get_blink_ratio(img, landmarks, self.RIGHT_EYE)
                    mouth_ratio = self.get_mouth_ratio(landmarks, self.MOUTH)
                    
                    # 2. Determine Status
                    # Thresholds might need tweaking based on distance from camera
                    BLINK_THRESH = 0.04
                    MOUTH_OPEN_THRESH = 0.3 
                    
                    # Check for Blinks
                    if left_eye_ratio < BLINK_THRESH and right_eye_ratio < BLINK_THRESH:
                        status = "Eyes Closed"
                    elif left_eye_ratio < BLINK_THRESH:
                        status = "Left Blink"
                    elif right_eye_ratio < BLINK_THRESH:
                        status = "Right Blink"
                    elif mouth_ratio > MOUTH_OPEN_THRESH:
                        status = "Mouth Open / Surprised"
                    else:
                        status = "Neutral"

                    # Display Status
                    cv2.putText(img, f"Expression: {status}", (20, 50), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
                    
                    # Display values for debugging (optional)
                    cv2.putText(img, f"Eye Ratio: {left_eye_ratio:.2f}", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(img, f"Mouth Ratio: {mouth_ratio:.2f}", (20, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow('MediaPipe Face Expressions', img)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceExpressionDetector()
    detector.run()

# python face_expression_tracker.py