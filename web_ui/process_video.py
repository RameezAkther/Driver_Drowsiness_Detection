import numpy as np
import cv2
import dlib
import mediapipe as mp
from tensorflow.keras.models import load_model
from pygame import mixer
from imutils import face_utils
from scipy.spatial import distance as dist
import math


# Initialize pygame for sound
mixer.init()
alarm_sound = mixer.Sound(r"C:\Users\Lenovo-Z50-70\Desktop\drowsyDetection\dataset\alert_tone\alarm.wav")
yawn_sound = mixer.Sound(r'C:\Users\Lenovo-Z50-70\Desktop\drowsyDetection\dataset\alert_tone\alarm2.wav')

# Load models and cascades
model_inceptionV3 = load_model(r"C:\Users\Lenovo-Z50-70\Desktop\drowsyDetection\models\inceptionV3_open_closed_eye.keras")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\Lenovo-Z50-70\Desktop\drowsyDetection\models\shape_predictor_68_face_landmarks.dat')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define head down threshold

HEAD_DOWN_THRESHOLD = 0.2  # Adjust based on testing
Score = 0  # For eye closure scoring

def preprocess_eye(eye):
    """Preprocess the eye image for model prediction."""
    eye = cv2.resize(eye, (80, 80))
    eye = eye / 255.0
    eye = np.expand_dims(eye, axis=0)
    return eye

# Define the calculate_distance function
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Yawn calculation function (lip distance)
def calculate_lip_distance(shape):
    upper_lip = shape[51]  # Landmark for the upper lip
    lower_lip = shape[57]  # Landmark for the lower lip
    lip_distance = dist.euclidean(upper_lip, lower_lip)
    return lip_distance

# Indices for eye landmarks
(left_eye_start, left_eye_end) = (36, 41)
(right_eye_start, right_eye_end) = (42, 47)

def generate_frames(current_method, cam_id):
    global Score
    cap = cv2.VideoCapture(cam_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Using MediaPipe if selected
        if current_method == "MediaPipe":
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
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
                    cv2.putText(frame, "Head Down!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    yawn_sound.play()
                else:
                    cv2.putText(frame, "Head Straight!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Using Dlib-based methods for eye or yawn detection
        elif current_method in "Dlib":
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Eye detection
                left_eye = shape[left_eye_start:left_eye_end + 1]
                right_eye = shape[right_eye_start:right_eye_end + 1]
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Mouth detection
                lip_distance = calculate_lip_distance(shape)

                # Draw eye landmarks
                for (x, y) in np.concatenate((left_eye, right_eye)):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                for i in range(len(left_eye) - 1):
                    cv2.line(frame, tuple(left_eye[i]), tuple(left_eye[i + 1]), (0, 255, 255), 1)
                cv2.line(frame, tuple(left_eye[0]), tuple(left_eye[-1]), (0, 255, 255), 1)
                for i in range(len(right_eye) - 1):
                    cv2.line(frame, tuple(right_eye[i]), tuple(right_eye[i + 1]), (0, 255, 255), 1)
                cv2.line(frame, tuple(right_eye[0]), tuple(right_eye[-1]), (0, 255, 255), 1)

                # Detect drowsiness based on EAR
                if ear < 0.21:
                    cv2.putText(frame, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    Score += 1
                else:
                    Score -= 1
                    if Score < 0:
                        Score = 0

                # Detect yawning based on lip distance
                if lip_distance > 26:
                    cv2.putText(frame, "Yawning", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    yawn_sound.play()
                    Score += 1

                # Play alert if score crosses threshold
                if Score > 30:
                    cv2.putText(frame, "ALERT!!!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    try:
                        alarm_sound.play()
                    except Exception as e:
                        print(f"Error playing sound: {e}")
                    Score = 0  # Reset score after alert

        elif current_method == "Combined":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
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
                    cv2.putText(frame, "Head Down!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    yawn_sound.play()
                else:
                    cv2.putText(frame, "Head Straight!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Eye detection
                left_eye = shape[left_eye_start:left_eye_end + 1]
                right_eye = shape[right_eye_start:right_eye_end + 1]
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Mouth detection
                lip_distance = calculate_lip_distance(shape)

                # Draw eye landmarks
                for (x, y) in np.concatenate((left_eye, right_eye)):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                for i in range(len(left_eye) - 1):
                    cv2.line(frame, tuple(left_eye[i]), tuple(left_eye[i + 1]), (0, 255, 255), 1)
                cv2.line(frame, tuple(left_eye[0]), tuple(left_eye[-1]), (0, 255, 255), 1)
                for i in range(len(right_eye) - 1):
                    cv2.line(frame, tuple(right_eye[i]), tuple(right_eye[i + 1]), (0, 255, 255), 1)
                cv2.line(frame, tuple(right_eye[0]), tuple(right_eye[-1]), (0, 255, 255), 1)

                # Detect drowsiness based on EAR
                if ear < 0.21:
                    cv2.putText(frame, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    Score += 1
                else:
                    Score -= 1
                    if Score < 0:
                        Score = 0

                # Detect yawning based on lip distance
                if lip_distance > 26:
                    cv2.putText(frame, "Yawning", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    yawn_sound.play()
                    Score += 1

                # Play alert if score crosses threshold
                if Score > 30:
                    cv2.putText(frame, "ALERT!!!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    try:
                        alarm_sound.play()
                    except Exception as e:
                        print(f"Error playing sound: {e}")
                    Score = 0  # Reset score after alert
        # Using InceptionV3 for eye state detection
        elif current_method == "InceptionV3":
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

            # Draw rectangle for score display
            height = frame.shape[0]
            cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

            for (ex, ey, ew, eh) in eyes:
                eye = frame[ey:ey + eh, ex:ex + ew]
                eye = preprocess_eye(eye)
                prediction = model_inceptionV3.predict(eye)

                # Eye closed prediction
                if prediction[0][0] > 0.30:
                    Score += 1
                    if Score > 15:
                        alarm_sound.play()

                    cv2.putText(frame, 'closed', (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f'Score {Score}', (100, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)

                # Eye open prediction
                elif prediction[0][1] > 0.90:
                    Score -= 1
                    if Score < 0:
                        Score = 0

                    cv2.putText(frame, 'open', (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f'Score {Score}', (100, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            