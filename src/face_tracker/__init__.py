import cv2


class Face:
    def __init__(self, name, x1, y1, width, height):
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

    def get_face(self):
        return self.x1, self.y1, self.width, self.height, self.presence_counter

    def increment_emotion_counter(self, emotion):
        if self.current_emotion != emotion:
            self.current_emotion = emotion
            self.emotion_counter = 0
        else:
            self.emotion_counter += 1

    def increment_presence(self):
        self.presence_counter += 1
        self.absence_counter = 0

    def increment_absence(self):
        self.absence_counter += 1

    def print_face_on_frame(self, frame):
        color = (255, 255, 255)
        if self.current_emotion == "happy":
            color = (0, 255, 0)
        elif self.current_emotion == "sad":
            color = (255, 0, 0)
        elif self.current_emotion == "angry":
            color = (0, 0, 255)
        cv2.rectangle(frame, (self.x1, self.y1), (self.x1 + self.width, self.y1 + self.height), color, 2)
        cv2.putText(frame, f"{self.name}",
                    (self.x1, self.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        print(f"Best match: {self.name}, Emotion: {self.current_emotion}, {self.emotion_counter}")


class FaceTracker:
    # Create a global dictionary to store face tracking data
    tracked_faces = {}

    def __init__(self):
        self.emotion_counter_threshold = 10
        self.absence_counter_threshold = 15
        self.presence_show_threshold = 5

    def track_face(self, face_name, x1, y1, width, height, emotion=None, tolerance=0.2):
        """Tracks a face based on name and position, updating a global tracker dictionary.

        Args:
            face_name (str): Name of the identified face.
            x1 (int): X-coordinate of the top-left corner of the face bounding box.
            y1 (int): Y-coordinate of the top-left corner of the face bounding box.
            width (int): Width of the face bounding box.
            height (int): Height of the face bounding box.
            tolerance (float): Percentage tolerance for allowing variations in position.
                               Default is 0.2 (20% tolerance).
        """

        # Calculate allowed position deviations
        x_tol = int(width * tolerance)
        y_tol = int(height * tolerance)

        # Check if this face is already tracked
        for tracked_name, tracked_face in self.tracked_faces.items():
            tx1, ty1, tw, th, count = tracked_face.get_face()
            # Compare positions within tolerance bounds
            if abs(x1 - tx1) <= x_tol and abs(y1 - ty1) <= y_tol and abs(width - tw) <= x_tol and abs(height - th) <= y_tol:
                # Update the existing tracked face's position and count
                self.tracked_faces[tracked_name].increment_presence()
                if emotion is not None:
                    self.tracked_faces[tracked_name].increment_emotion_counter(emotion)

                return  # Exit the function after updating

        # If not tracked, add this face as a new entry
        self.tracked_faces[face_name] = Face(face_name, x1, y1, width, height)

    def get_stable_faces(self):
        """Returns a dictionary of tracked faces that have been present for at least 5 frames."""
        return {name: face for name, face in self.tracked_faces.items()
                if face.presence_counter >= self.presence_show_threshold}

    def increment_face_absences(self):
        """Increments absence counters for all tracked faces and removes faces exceeding the absence threshold."""
        self.tracked_faces = {name: face for name, face in self.tracked_faces.items() if
                              face.increment_absence() or face.absence_counter < self.absence_counter_threshold}

    def print_faces_status(self):
        """Prints the status of all tracked faces."""
        for face in self.tracked_faces.values():
            emotion_message = f"Face validated!!! Emotion: {face.current_emotion}, {face.emotion_counter}" \
                                    if face.emotion_counter > self.emotion_counter_threshold else ""
            print(f"Face: {face.name}, Presence: {face.presence_counter}, Absence: {face.absence_counter} {emotion_message}")