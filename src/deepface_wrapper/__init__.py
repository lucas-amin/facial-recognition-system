import os
from typing import Dict, Any, List

import numpy as np
from deepface import DeepFace
from pandas import DataFrame


class DeepFaceWrapper:
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepID", "ArcFace", "SFace", "GhostFaceNet"]

    def __init__(self):
        self.model = "Facenet512"
        self.detector_backend = "yolov8"
        self.match_threshold = 0.75
        self.db_path = os.path.join(os.path.dirname(__file__)) + "/../../faces_database"
        os.makedirs(self.db_path, exist_ok=True)

        self.frame_counter = 0

    @staticmethod
    def verify_face_pair(image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        return DeepFace.verify(image1, image2)

    def find_match_in_database(self, image: np.ndarray) -> List[DataFrame]:
        try:
            self.frame_counter += 1 if self.frame_counter <= 175 else self.frame_counter == 0

            # Change the model used every 20 frame_counter
            # model_index = int(self.frame_counter / 25)
            # print(f"Using model {self.models[model_index]}, {model_index}")
            return DeepFace.find(img_path=image, db_path=self.db_path, model_name=self.model,
                                 enforce_detection=False, detector_backend=self.detector_backend,
                                 threshold=self.match_threshold, silent=True)
        except Exception as e:
            print(f"Error finding face {e}")
            return []

    def facial_analysis(self, image: np.ndarray) -> List[Dict[str, Any]]:
        return DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False,
                                detector_backend=self.detector_backend)

    def extract_face(self, image: np.ndarray) -> np.ndarray:
        return DeepFace.detectFace(image, enforce_detection=False, detector_backend=self.detector_backend)[0]
