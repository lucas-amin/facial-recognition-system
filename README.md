# facial-recognition-system


#### Objective: 

* Developing a facial recognition system that not only accurately identifies users but also can distinguish between a real person and spoofing attempts using photos, videos, or masks.


## Facial Recognition System Documentation

# Overview

This Python application uses OpenCV for real-time face tracking and DeepFace for facial recognition and emotion analysis. It identifies faces in a video stream, tracks their presence and emotions over time, and validates users based on pre-defined emotional patterns. The validated users are saved with their images and metadata.

# Features
* Real-time Face Detection and Tracking: Utilizes a yolov8 face detector and a simple tracking algorithm to identify and track faces in a video stream.
* Facial Recognition: Employs [DeepFace](https://github.com/serengil/deepface) to match detected faces with a pre-built database of known individuals.
* Emotion Analysis: [DeepFace](https://github.com/serengil/deepface) also analyzes facial expressions to determine the dominant emotion (e.g., happy, sad, angry).
* Spoofing Detection: Detects and prevents spoofing attempts using photos, videos, or masks using [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master), encapsulated by Deepface .
* User Validation: Implements a validation process that requires users to display specific emotional patterns in sequence.
* Data Storage: Saves validated user name and validation timestamp in a JSON file and a crop of its face at the time of validation.

# Requirements
* Python 3.7.16
* OpenCV
* Flask
* DeepFace
* Ultralytics
* NumPy 

# Installation
Clone the Repository:

`git clone https://github.com/lucas-amin/facial-recognition-system.git`

Install Dependencies:

`pip install -r requirements.txt`

Make sure you have the relevant pre-trained face recognition models for DeepFace installed. If it is not automatically downloaded, you can find further information on how to download and use the models on their GitHub page: https://github.com/serengil/deepface

# Configuration
* Face Database: Ensure you have a directory named faces_database at the root level containing images of individuals you want the system to recognize.
    * Be sure to name the image files with the corresponding individual's name (e.g., john.jpg) and have a clear view of a single person's face in the image.


# Usage

### 1. Run the Application:

`python src/main.py`

### 2. Access the Video Stream:
Open a web browser and navigate to http://localhost:5000. You should see a live video feed with detected faces being tracked and analyzed.

### 3. User Validation:

Follow the instructions displayed on the video feed to display the required emotional patterns for validation.
Once validated, your information will be saved in the `validation` folder, and your image will be displayed.

# Code Structure
* app.py:
    * Main Flask application script.
    * Handles video streaming, face processing, and API endpoints.
* deepface_wrapper.py:
  * Wrapper class for DeepFace functions.
  * Encapsulates face recognition and emotion analysis.
* face_tracker.py:
  * Contains the FaceTracker class.
  * Manages face tracking, validation logic, and data storage.
* face_tracker/face.py:
  * Contains the Face class.
  * Represents a single face in the video feed.
  * Stores face metadata and validation status.
* templates/index.html:
  * HTML template for the video feed webpage.



# Customization
* Emotion Detection: Explore different emotion detection models within DeepFace or other libraries to enhance the system's accuracy.
* Validation Rules: Adjust the validation logic to fit your specific use case and the desired emotional patterns.
* User Interface: Modify the index.html template to customize the appearance of the video feed webpage.

