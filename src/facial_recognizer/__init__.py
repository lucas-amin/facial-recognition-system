import os
from typing import Dict, Any, List

import cv2
import numpy as np
from deepface import DeepFace
from pandas import DataFrame

from face_tracker import FaceTracker


class FacialRecognizer:
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace",
              "GhostFaceNet"]

    def __init__(self):
        self.model = FacialRecognizer.models[1]
        self.detector_backend = "yolov8"
        self.match_threshold = 0.75
        self.db_path = os.path.join(os.path.dirname(__file__)) + "/../../faces_database"
        os.makedirs(self.db_path, exist_ok=True)

    @staticmethod
    def verify_face_pair(image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        return DeepFace.verify(image1, image2)

    def find_match_in_database(self, image: np.ndarray) -> List[DataFrame]:
        try:
            return DeepFace.find(img_path=image, db_path=self.db_path, enforce_detection=False,
                                 detector_backend=self.detector_backend, threshold=self.match_threshold,
                                 silent=True)
        except Exception as e:
            print(f"Error finding face {e}")
            return []

    def facial_analysis(self, image: np.ndarray) -> List[Dict[str, Any]]:
        return DeepFace.analyze(img_path=image,
                                actions=['emotion'],
                                enforce_detection=False,
                                detector_backend=self.detector_backend)

    def extract_face(self, image: np.ndarray) -> np.ndarray:
        return DeepFace.detectFace(image, enforce_detection=False, detector_backend=self.detector_backend)[0]


def process_faces():
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
    facial_recognizer = FacialRecognizer()
    face_tracker = FaceTracker()

    cap = cv2.VideoCapture(0)
    frame_n = 0
    try:
        while True:
            ret, frame = cap.read()

            if frame_n % 2 == 0:  # Process every 5th frame
                faces = facial_recognizer.find_match_in_database(frame)
                process_faces()

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
