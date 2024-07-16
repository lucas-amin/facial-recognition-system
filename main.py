import os

import cv2
import numpy as np
from deepface import DeepFace

from face_tracker import FaceTracker
from src.facial_recognizer import FacialRecognizer

facial_recognizer = FacialRecognizer()
face_tracker = FaceTracker()


def process_faces(frame: np.ndarray):
    faces = facial_recognizer.find_match_in_database(frame)

    for face_matches in faces:
        if face_matches.empty:
            continue

        best_face_match = face_matches.loc[face_matches['distance'].idxmin()]
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


if __name__ == "__main__":
    # Database Setup
    db_path = os.path.join(os.path.dirname(__file__)) + "../faces_database"
    db_path.mkdir(exist_ok=True)


    # Webcam Setup
    cap = cv2.VideoCapture(0)
    frame_n = 0
    try:
        while True:
            ret, frame = cap.read()

            if frame_n % 5 == 0:
                process_faces(facial_recognizer, frame)

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

    # Demonstrate face recognition: Find all faces in the database
    df = DeepFace.find(img_path="path/to/your/image.jpg", db_path=str(db_path))  # Replace with the path to your image
    print(df)