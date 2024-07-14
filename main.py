import os

import cv2
from deepface import DeepFace

from src.facial_recognizer import FacialRecognizer


def main():
    # Database Setup
    db_path = os.path.join(os.path.dirname(__file__)) + "../faces_database"
    db_path.mkdir(exist_ok=True)

    facial_recognizer = FacialRecognizer()

    # Webcam Setup
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow("Webcam", frame)

        # Capture Image on Spacebar Press
        key = cv2.waitKey(1)
        if key == 32:  # 32 is the ASCII code for space
            face_path = db_path / f"face_{len(list(db_path.glob('*.jpg')))}.jpg"  # Create unique filename
            cv2.imwrite(str(face_path), frame)
            print("Face registered and saved!")

        # Quit on 'q' Press
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Demonstrate face recognition: Find all faces in the database
    df = DeepFace.find(img_path="path/to/your/image.jpg", db_path=str(db_path))  # Replace with the path to your image
    print(df)


if __name__ == "__main__":
    main()
