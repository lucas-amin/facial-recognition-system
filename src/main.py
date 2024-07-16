import cv2
import numpy as np
from flask import Response, render_template, Flask

from face_tracker import FaceTracker
from facial_recognizer import FacialRecognizer

app = Flask(__name__)

facial_recognizer = FacialRecognizer()
face_tracker = FaceTracker()


def process_faces(frame: np.ndarray):
    faces = facial_recognizer.find_match_in_database(frame)

    if len(faces) > 1:
        message = f"Please provide only one face in the frame."
        cv2.putText(frame, message, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    elif len(faces) == 1 and not faces[0].empty:
        face_match = faces[0]
        best_face_match = face_match.loc[face_match['distance'].idxmin()]
        x1, y1, width, height = best_face_match[["source_x", "source_y", "source_w", "source_h"]].astype(int)
        face_name = best_face_match["identity"].split("/")[-1].split(".")[0]

        face_extraction = frame[y1:y1 + height, x1:x1 + width]
        analysis = facial_recognizer.facial_analysis(face_extraction)
        emotion = analysis[0]["dominant_emotion"]

        if width > 0 and height > 0:  # Check if valid bounding box
            face_tracker.track_face(face_name, x1, y1, width, height, emotion)

        # read stable faces from face_tracker and print them using function show_face
        stable_faces = face_tracker.get_stable_faces()

        for name, face in stable_faces.items():
            face.print_face_on_frame(frame)

        face_tracker.increment_face_absences()
        face_tracker.print_faces_status()


def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    frame_n = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture frame")
            break
        if frame_n % 5 == 0:
            process_faces(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_img + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
