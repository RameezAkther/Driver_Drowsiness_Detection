import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import mediapipe as mp
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
from mtcnn.mtcnn import MTCNN
import io
import math

model = load_model(r"C:\Users\Lenovo-Z50-70\Desktop\drowsyDetection\models\inceptionV3_open_closed_eye.keras")

# Yawn calculation function (lip distance)
def calculate_lip_distance(shape):
    upper_lip = shape[51]  # Landmark for the upper lip
    lower_lip = shape[57]  # Landmark for the lower lip
    lip_distance = dist.euclidean(upper_lip, lower_lip)
    return lip_distance

# Function to calculate EAR
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Load the detector and predictor
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Lenovo-Z50-70\Desktop\drowsyDetection\models\shape_predictor_68_face_landmarks.dat")

# Indices for the eyes in the landmark array
(left_eye_start, left_eye_end) = (36, 41)
(right_eye_start, right_eye_end) = (42, 47)

DROWSINESS_THRESHOLD = 0.21  # EAR threshold to detect drowsiness
YAWN_THRESHOLD = 140


def process_image_1(image, current_method):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #print(current_method)
    if current_method == 'InceptionV3':
        detector = MTCNN()
        results = detector.detect_faces(cv_image)
        if len(results) == 0:
            print("No faces detected.")
        else:
            for result in results:
                x, y, width, height = result['box']
                cv2.rectangle(cv_image, (x, y), (x + width, y + height), (255, 0, 0), 2)  # Draw rectangle around face

                # Crop the face region
                face_roi = cv_image[y:y + height, x:x + width]

                # Approximate eye region (upper half of the face)
                eye_region_y1 = int(0.25 * height)  # Start slightly below the forehead
                eye_region_y2 = int(0.5 * height)   # Middle of the face
                eye_region_x1 = int(0.15 * width)   # Slightly inside from the left edge
                eye_region_x2 = int(0.85 * width)   # Slightly inside from the right edge

                # Coordinates for drawing rectangles around the eye regions
                left_eye_box = (eye_region_x1, eye_region_y1, int(width / 2) - eye_region_x1, eye_region_y2 - eye_region_y1)
                right_eye_box = (int(width / 2), eye_region_y1, eye_region_x2 - int(width / 2), eye_region_y2 - eye_region_y1)

                # Draw rectangles around the approximate left and right eye regions
                cv2.rectangle(face_roi, (left_eye_box[0], left_eye_box[1]), 
                            (left_eye_box[0] + left_eye_box[2], left_eye_box[1] + left_eye_box[3]), 
                            (0, 255, 0), 2)  # Left eye box

                cv2.rectangle(face_roi, (right_eye_box[0], right_eye_box[1]), 
                            (right_eye_box[0] + right_eye_box[2], right_eye_box[1] + right_eye_box[3]), 
                            (0, 255, 0), 2)  # Right eye box

                # Crop the left and right eye regions
                left_eye_roi = face_roi[eye_region_y1:eye_region_y2, eye_region_x1:int(width / 2)]
                right_eye_roi = face_roi[eye_region_y1:eye_region_y2, int(width / 2):eye_region_x2]

                # Preprocess each eye region for the model
                for eye, box in zip([left_eye_roi, right_eye_roi], [left_eye_box, right_eye_box]):
                    eye_resized = cv2.resize(eye, (80, 80))
                    eye_normalized = eye_resized / 255.0
                    eye_reshaped = eye_normalized.reshape(80, 80, 3)
                    eye_input = np.expand_dims(eye_reshaped, axis=0)

                    # Model prediction
                    prediction = model.predict(eye_input)

                    # Determine eye status
                    if prediction[0][0] > 0.30:  # Eyes closed
                        result_text = 'closed'
                    elif prediction[0][1] > 0.90:  # Eyes open
                        result_text = 'open'
                    else:
                        result_text = 'unknown'

                    # Display result on the image (on top of the eye box)
                    cv2.putText(face_roi, f'{result_text}', (box[0], box[1] - 10), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, 
                                color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        # Convert BGR image to RGB for displaying
    elif current_method == 'MediaPipe':
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        # Define the threshold for head down position
        HEAD_DOWN_THRESHOLD = 0.2  # Change this threshold based on camera position
        
        # Process the image with MediaPipe Pose
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(cv_image)
        
        # Draw landmarks on the image if they exist
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(cv_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Get the nose landmark
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            
            # Calculate the nose position
            nose_y = nose.y  # This is normalized between 0 and 1
            
            # Check if head is tilted downward
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(cv_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Get the nose and shoulder landmarks
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                
                # Calculate the collar (midpoint between shoulders)
                collar_x = (left_shoulder.x + right_shoulder.x) / 2
                collar_y = (left_shoulder.y + right_shoulder.y) / 2
                collar_point = type('Landmark', (object,), {'x': collar_x, 'y': collar_y})
                
                # Calculate the distance between nose and collar
                distance_nose_collar = calculate_distance(nose, collar_point)
                
                # Check if head is tilted downward based on distance
                if distance_nose_collar < HEAD_DOWN_THRESHOLD:
                    cv2.putText(cv_image, "Head Down!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 8)
                else:
                    cv2.putText(cv_image, "Head Straight!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 8)
    elif current_method == 'Dlib' or 'Combined':
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        rects = detector_dlib(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    
            left_eye = shape[left_eye_start:left_eye_end + 1]
            right_eye = shape[right_eye_start:right_eye_end + 1]
    
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
    
            ear = (left_ear + right_ear) / 2.0
    
            # Draw landmarks for the left eye
            for (x, y) in left_eye:
                cv2.circle(cv_image, (x, y), 2, (0, 255, 0), -1)  # Green circles on left eye landmarks
    
            # Draw landmarks for the right eye
            for (x, y) in right_eye:
                cv2.circle(cv_image, (x, y), 2, (0, 255, 0), -1)  # Green circles on right eye landmarks
    
            # Connect the points with lines for visualization
            for i in range(len(left_eye) - 1):
                cv2.line(cv_image, tuple(left_eye[i]), tuple(left_eye[i + 1]), (0, 255, 255), 7)  # Yellow lines for left eye
            cv2.line(cv_image, tuple(left_eye[0]), tuple(left_eye[-1]), (0, 255, 255), 7)  # Close the left eye shape
    
            for i in range(len(right_eye) - 1):
                cv2.line(cv_image, tuple(right_eye[i]), tuple(right_eye[i + 1]), (0, 255, 255), 7)  # Yellow lines for right eye
            cv2.line(cv_image, tuple(right_eye[0]), tuple(right_eye[-1]), (0, 255, 255), 7)  # Close the right eye shape
    
            # Detect if the eyes are closed or open based on EAR
            if ear < DROWSINESS_THRESHOLD:
                cv2.putText(cv_image, "Eyes Closed", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 10)
            else:
                cv2.putText(cv_image, "Eyes Open", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 10)
    
            # Calculate lip distance to detect yawning
            lip_distance = calculate_lip_distance(shape)
    
            if lip_distance > YAWN_THRESHOLD:
                cv2.putText(cv_image, "Yawn Detected", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 10)
            else:
                cv2.putText(cv_image, "Mouth Closed", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 10)
            if current_method == "Combined":
                # Initialize MediaPipe Pose
                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
                
                # Define the threshold for head down position
                HEAD_DOWN_THRESHOLD = 0.2  # Change this threshold based on camera position
                
                # Process the image with MediaPipe Pose
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    results = pose.process(cv_image)
                
                # Draw landmarks on the image if they exist
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(cv_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Get the nose landmark
                    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                    
                    # Calculate the nose position
                    nose_y = nose.y  # This is normalized between 0 and 1
                    
                    # Check if head is tilted downward
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(cv_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        # Get the nose and shoulder landmarks
                        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        
                        # Calculate the collar (midpoint between shoulders)
                        collar_x = (left_shoulder.x + right_shoulder.x) / 2
                        collar_y = (left_shoulder.y + right_shoulder.y) / 2
                        collar_point = type('Landmark', (object,), {'x': collar_x, 'y': collar_y})
                        
                        # Calculate the distance between nose and collar
                        distance_nose_collar = calculate_distance(nose, collar_point)
                        
                        # Check if head is tilted downward based on distance
                        if distance_nose_collar < HEAD_DOWN_THRESHOLD:
                            cv2.putText(cv_image, "Head Down!", (30, 600), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 8)
                        else:
                            cv2.putText(cv_image, "Head Straight!", (30, 600), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 8)
        # Encode the processed image to JPEG format
    img_io = io.BytesIO()
    is_success, buffer = cv2.imencode(".jpg", cv_image)  # Encode the image as JPEG
    if is_success:
        img_io.write(buffer)  # Write to the BytesIO stream
        img_io.seek(0)  # Move to the start of the stream
        cv2.imwrite(r"C:\Users\Lenovo-Z50-70\Desktop\drowsyDetection\web_ui\processed_imgs\processed_image.jpg", cv_image)
        return img_io
    else:
        return None
