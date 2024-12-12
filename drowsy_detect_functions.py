import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer

# Load your model
model = load_model('model.keras')

# Initialize sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize score
Score = 0

def cnn_eye_detection(frame):
    global Score
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces and eyes
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    height, width = frame.shape[0:2]

    # Drawing a rectangle for the score display
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3)
    
    for (ex, ey, ew, eh) in eyes:
        # Preprocessing for CNN input
        eye = frame[ey:ey + eh, ex:ex + ew]  # Correctly using ew (eye width)
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)

        # Model prediction
        prediction = model.predict(eye)

        # If eyes are closed
        if prediction[0][0] > 0.30:
            cv2.putText(frame, 'closed', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                        thickness=1, lineType=cv2.LINE_AA)
            Score += 1
            if Score > 15:
                try:
                    sound.play()  # Play alarm when eyes are closed for too long
                except:
                    pass

        # If eyes are open
        elif prediction[0][1] > 0.90:
            cv2.putText(frame, 'open', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                        thickness=1, lineType=cv2.LINE_AA)
            Score -= 1
            if Score < 0:
                Score = 0

    # Display score on the frame
    cv2.putText(frame, 'Score: ' + str(Score), (100, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                thickness=1, lineType=cv2.LINE_AA)

    return frame, Score
