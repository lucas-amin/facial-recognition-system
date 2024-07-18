import logging

import cv2
import numpy as np
from flask import Response, render_template, Flask, jsonify

from deepface_wrapper import DeepFaceWrapper
from face_tracker import FaceTracker

app = Flask(__name__)

deepface_wrapper = DeepFaceWrapper()
face_tracker = FaceTracker()


def process_faces(frame: np.ndarray):
    try:
        faces = deepface_wrapper.find_match_in_database(frame)
    except ValueError:
        cv2.putText(frame, f"Spoofing attempt detected! Please show your real face",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        logging.error("Spoof detected in the given image.")
        return
    face_tracker.set_image_shape(frame)

    if len(faces) > 1:
        message = f"Please provide only one face in the frame."
        cv2.putText(frame, message, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    elif len(faces) == 1 and not faces[0].empty:
        face_match = faces[0]
        best_face_match = face_match.loc[face_match['distance'].idxmin()]
        x1, y1, width, height = best_face_match[["source_x", "source_y", "source_w", "source_h"]].astype(int)

        face_name = best_face_match["identity"].split("/")[-1].split(".")[0]

        face_image = frame[y1:y1 + height, x1:x1 + width]

        analysis = deepface_wrapper.facial_analysis(face_image)

        emotion = analysis[0]["dominant_emotion"]

        if width > 0 and height > 0:  # Check if valid bounding box
            face_tracker.track_face(face_name, face_image, x1, y1, width, height, emotion,)

        # read stable faces from face_tracker and print them using function show_face
        stable_face = face_tracker.get_stable_face()

        if stable_face is not None:
            stable_face.print_face_on_frame(frame)

        face_tracker.update_faces_status()


def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: failed to capture frame")
            break
        process_faces(frame) if frame_n % 5 == 0 else None

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_img + b'\r\n')


@app.route('/emotion')
def get_next_desired_emotion():
    emotion = face_tracker.output_emotion
    return jsonify({'emotion': emotion})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
