# app.py (Flask Backend Example)
from flask import Flask, Response, render_template, request, jsonify, make_response
from PIL import Image
import cv2
from process_video import generate_frames
from process_image import process_image_1
import io
import base64

app = Flask(__name__)

'''
1. Try to make it run faster
2. check the score value for InceptionV3
3. Remove NN and CNN
4. When camera is turned on and off check it properly and also check the feed type on that section
5. check for annotations
'''

camera = None  # Global camera object

current_method = 'InceptionV3'
cam_id = 1


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to provide the video feed."""
    global camera
    if camera and camera.isOpened():
        return Response(generate_frames(current_method, cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera not started", 503
    
@app.route('/upload_image', methods=['POST'])
def process_image():
    global current_method

    file = request.files['file']
    image = Image.open(file.stream)
    img_io = process_image_1(image, current_method)
    if img_io == None:
        return "Image processing failed", 500
    else:
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        return jsonify({'image_data': img_base64})

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    """Route to toggle the camera on or off."""
    global camera
    global cam_id
    if camera and camera.isOpened():
        camera.release()
        camera = None
        return "Camera turned off", 200
    else:
        camera = cv2.VideoCapture(cam_id)  # Open the camera
        if not camera.isOpened():
            return "Failed to access camera", 500
        return "Camera turned on", 200

@app.route('/change_method', methods=['POST'])
def change_method():
    global current_method
    data = request.get_json()
    current_method = data.get('method')
    return jsonify({"method": current_method})

@app.route('/change_feed', methods=['POST'])
def change_feed():
    global cam_id
    data = request.get_json()
    cam_id = int(data.get('feed'))
    return jsonify({"feed": cam_id})

'''
def generate_frames(val1, val2):
    """Generator function to yield video frames."""
    global camera
    while camera and camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
'''

if __name__ == '__main__':
    app.run(debug=True)
