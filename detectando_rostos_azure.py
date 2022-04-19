import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import cv2
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition

# This key will serve all examples in this document.
KEY = "4e3878c805e34a96bb82ad72ce2694dd"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://phycustech.cognitiveservices.azure.com/"

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

img_file = open(r'foto1.jpg', 'rb')

# We use detection model 3 to get better performance.
response_detected_faces = face_client.face.detect_with_stream(image=img_file, detection_model='detection_03', recognition_model='recognition_04', return_face_landmarks=True)

if not response_detected_faces:
    raise Exception('No face detected')

print('Number of people deteted: {}'.format(len(response_detected_faces)))

print(vars(response_detected_faces[0]))
print(vars(response_detected_faces[0].face_landmarks).keys())
print(response_detected_faces[0].face_landmarks.mouth_left)

img = Image.open(img_file)
draw = ImageDraw.Draw(img)

for face in response_detected_faces:
    rect = face.face_rectangle
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top
    draw.rectangle(((left, top), (right, bottom)), outline='green', width=5)

    x = face.face_landmarks.nose_tip.x
    y = face.face_landmarks.nose_tip.y
    draw.rectangle(((x,y), (x,y)), outline='white', width=7)

    mouth_left = face.face_landmarks.mouth_left.x, face.face_landmarks.mouth_left.y
    mouth_right = face.face_landmarks.mouth_right.x, face.face_landmarks.mouth_right.y
    lip_bottom = face.face_landmarks.under_lip_bottom.x, face.face_landmarks.under_lip_bottom.y
    draw.rectangle((mouth_left, (mouth_right[0], lip_bottom[1])), outline='yellow', width=2)
    img.show()