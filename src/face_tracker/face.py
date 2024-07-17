import time

import cv2
import numpy as np


class Face:
    def __init__(self, name, x1, y1, width, height):
        self.EMOTION_CHANGE_THRESHOLD = 10
        self.EMOTION_VALIDATION_THRESHOLD = 20
        self.INCORRECT_DETECTION_THRESHOLD = 3

        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.width = width
        self.height = height
        self.presence_counter = 0
        self.absence_counter = 0

        self.emotion_counter = 0
        self.current_emotion = ""

        self.validated = False
        self.validation_time = None
        self.first_happy_emotion_validation = False
        self.sad_emotion_validation = False
        self.second_happy_emotion_validation = False

    def get_face(self):
        return self.x1, self.y1, self.width, self.height, self.presence_counter

    def update_emotion(self, emotion: str):
        if emotion == "fear" or emotion == "angry":
            emotion = "sad"

        if self.current_emotion != emotion:
            self.emotion_counter += 1  # Increment on a mismatch

            if self.emotion_counter >= self.INCORRECT_DETECTION_THRESHOLD:
                # Only reset if we've hit the threshold
                self.emotion_counter = 0
                self.current_emotion = emotion  # Update the current emotion
        else:
            self.emotion_counter += 1

    def increment_emotion_counter(self, emotion: str):
        self.update_emotion(emotion)

        # Validate emotions, if emotion is detected EMOTION_VALIDATION_THRESHOLD times, then it is validated.
        # It must be in the order happy -> Sad -> Neutral
        # If the order is broken, all are invalid and counters go to zero
        if self.emotion_counter >= self.EMOTION_VALIDATION_THRESHOLD:
            if self.current_emotion == "happy" and not self.sad_emotion_validation:
                self.first_happy_emotion_validation = True
                self.sad_emotion_validation = self.second_happy_emotion_validation = False

            elif self.current_emotion == "sad" and self.first_happy_emotion_validation:
                self.sad_emotion_validation = True

            elif self.current_emotion == "happy" and self.sad_emotion_validation:
                self.second_happy_emotion_validation = self.validated = True
        else:
            if (self.emotion_counter >= self.EMOTION_CHANGE_THRESHOLD and not (
                    self.current_emotion == "sad" and self.first_happy_emotion_validation) and not (
                    self.current_emotion == "happy" and self.sad_emotion_validation)):
                self.first_happy_emotion_validation = self.sad_emotion_validation =\
                    self.second_happy_emotion_validation = False

    def increment_presence(self):
        self.presence_counter += 1
        self.absence_counter = 0

    def increment_absence(self):
        self.absence_counter += 1

    def print_face_on_frame(self, frame: np.ndarray) -> None:
        """
        Print the face bounding box and name on the frame according to the emotion
        :param frame: frame to print on
        """
        colors = {"neutral": (100, 100, 100), "angry": (0, 0, 255), "happy": (0, 255, 0), "sad": (255, 0, 0),
                  "fear": (255, 0, 0)}
        color = colors.get(self.current_emotion, (100, 100, 100))
        cv2.rectangle(frame, (self.x1, self.y1), (self.x1 + self.width, self.y1 + self.height), color, 2)
        cv2.putText(frame, f"{self.name}", (self.x1, self.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36, 255, 12), 2)

        # Add color to the corners of the image to match the next emotion the person needs to show to be validated
        height, width, _ = frame.shape
        next_emotion_for_validation = ""
        if not self.first_happy_emotion_validation:
            next_emotion_for_validation = "happy"
        elif not self.sad_emotion_validation:
            next_emotion_for_validation = "sad"
        elif not self.second_happy_emotion_validation:
            next_emotion_for_validation = "happy"
        if next_emotion_for_validation:
            cv2.rectangle(frame, (0, 0), (width - 5, height - 5), colors[next_emotion_for_validation], 10)
        else:
            self.validation_time = self.validation_time or time.strftime("%H:%M:%S-%d/%m/%Y")
            cv2.putText(frame, f"{self.name} - Validated at {self.validation_time}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
