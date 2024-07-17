import logging
import os

import cv2
import numpy as np

from face_tracker.face import Face
from json_file_wrapper import JSONFileWrapper


class FaceTracker:
    tracked_faces = {}
    emotion_counter_threshold = 10
    absence_counter_threshold = 15
    presence_show_threshold = 5

    def __init__(self):
        self.database_folder = "../faces_database"
        os.makedirs(self.database_folder, exist_ok=True)
        print(os.listdir(self.database_folder))
        self.output_emotion = 0
        self.validated_faces_folder = f"../validation/"
        os.makedirs(self.validated_faces_folder, exist_ok=True)
        self.json_file_wrapper = JSONFileWrapper(f"{self.validated_faces_folder}validation_data.json")

    def track_face(self, face_name: str, face_image: np.ndarray, x1: int, y1: int, width: int, height: int,
                   emotion: str = None, tolerance: float = 0.2):
        """Tracks a face based on name and position, updating a global tracker dictionary.
           Creates a Face object if the face is not already tracked and then updates its data accordingly.

        Args:
            face_name (str): Name of the identified face.
            face_image (np.ndarray): Opencv crop of the detected face
            x1 (int): X-coordinate of the top-left corner of the face bounding box.
            y1 (int): Y-coordinate of the top-left corner of the face bounding box.
            width (int): Width of the face bounding box.
            height (int): Height of the face bounding box.
            tolerance (float): Percentage tolerance for allowing variations in position.
                               Default is 0.2 (20% tolerance).
            emotion: Emotion detected in the face, used by the Face object to validate the user.
        """

        # Calculate allowed position deviations
        x_tol = int(width * tolerance)
        y_tol = int(height * tolerance)

        # Check if this face is already tracked
        for tracked_name, tracked_face in self.tracked_faces.items():
            tx1, ty1, tw, th, count = tracked_face.get_face()
            # Compare positions within tolerance bounds
            if abs(x1 - tx1) <= x_tol and abs(y1 - ty1) <= y_tol and abs(width - tw) <= x_tol and abs(
                    height - th) <= y_tol:
                # Update the existing tracked face's position and count
                self.tracked_faces[tracked_name].increment_presence()
                if emotion is not None:
                    self.tracked_faces[tracked_name].increment_emotion_counter(emotion)

                return  # Exit the function after updating

        # If not tracked, add this face as a new entry
        self.tracked_faces[face_name] = Face(face_name, face_image, x1, y1, width, height, self.image_width)

    def update_stable_face(self):
        most_stable_face = self.get_stable_face()
        # Save the updated tracker dictionary to a json file when the face is validated
        if most_stable_face is not None:
            self.output_emotion = (1 + most_stable_face.first_happy_emotion_validation +
                                   most_stable_face.sad_emotion_validation +
                                   most_stable_face.second_happy_emotion_validation)
            if most_stable_face.validated and not most_stable_face.saved:
                self.save_face_image(most_stable_face)
                self.deletion_counter = 1
            if most_stable_face.deletion_counter >= 0:
                most_stable_face.deletion_counter += 1

                if most_stable_face.deletion_counter >= most_stable_face.DELETION_COUNTER_THRESHOLD:
                    del self.tracked_faces[most_stable_face.name]
                self.output_emotion = 0
        else:
            self.output_emotion = 0

    def save_face_image(self, most_stable_face: Face):
        self.json_file_wrapper.append_and_save((most_stable_face.name, most_stable_face.validation_time))
        file_name = (f"{most_stable_face.name}_{most_stable_face.validation_time}.jpg"
                     .replace("/", "-").replace(":", "_"))
        logging.info(f"Saving file {file_name} at {self.validated_faces_folder}")
        sucessfully_saved = cv2.imwrite(f"{self.validated_faces_folder}/{file_name}", most_stable_face.face_image)
        if not sucessfully_saved:
            logging.error(f"Could not save {file_name} at {self.validated_faces_folder}")
        else:
            most_stable_face.saved = True

    def get_stable_face(self):
        """Returns the tracked face that has highest presence_counter. if the presence_show_threshold is achieved"""
        highest_counter_face = max(self.tracked_faces, key=lambda x: self.tracked_faces[x].presence_counter)
        if self.tracked_faces[highest_counter_face].presence_counter >= self.presence_show_threshold:
            return self.tracked_faces[highest_counter_face]
        return None

    def update_faces_status(self):
        """Increments absence counters for all tracked faces and removes faces exceeding the absence threshold."""
        self.update_stable_face()

        self.tracked_faces = {name: face for name, face in self.tracked_faces.items() if
                              face.increment_absence() or face.absence_counter < self.absence_counter_threshold}

    def set_image_shape(self, frame: np.ndarray):
        self.image_height, self.image_width, _ = frame.shape

