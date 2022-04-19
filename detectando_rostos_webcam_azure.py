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

#Habilitando a WebCam conectada ao dispositivo
camera = cv2.VideoCapture(0)

contador = 0

#Mudar para True se quiser que as marcaçções de detecção de rosto, nariz e boca apareçam na foto
draw_choice = True

#Caminho para salvar as fotos
path = "C:/Users/gusta/PycharmProjects/PychusTechAI/Foto_Gustavo"

#Loop para a detecção tempo real
while 1:
    #Fazendo a leitura da imagem
    conectado, imagem = camera.read()

    #Salvando a imagem da webcam para posteriormente o serviço de detecção da azure ler o arquivo
    cv2.imwrite('teste.jpg', imagem)

    #Abrindo a imagem salva da webcam
    img_file = open(r'teste.jpg', 'rb')

    #Analisando se existe rostos na imagem
    response_detected_faces = face_client.face.detect_with_stream(image=img_file, detection_model='detection_03',
                                                                  recognition_model='recognition_04',
                                                                  return_face_landmarks=True)
    if not response_detected_faces:
        print('No face detected')
    else:
        print('Number of people deteted: {}'.format(len(response_detected_faces)))

    img = Image.open(img_file)
    draw = ImageDraw.Draw(img)

    #Para os rostos detectados
    for face in response_detected_faces:

        if draw_choice:
            rect = face.face_rectangle
            left = rect.left
            top = rect.top
            right = rect.width + left
            bottom = rect.height + top
            #Desenhando um retângulo na imagem
            #draw.rectangle(((left, top), (right, bottom)), outline='green', width=5)
            cv2.rectangle(imagem, (left, top), (right, bottom), (0, 0, 255), 2)

            #Desenhando um ponto no nariz
            x = int(face.face_landmarks.nose_tip.x)
            y = int(face.face_landmarks.nose_tip.y)
            #draw.rectangle(((x, y), (x, y)), outline='white', width=7)
            #print(F"x: {x}, y: {y}")
            cv2.rectangle(imagem, (x, y), (x, y), (255, 255, 255), 7)

            #Desenhando um retângulo em volta da boca
            mouth_left = int(face.face_landmarks.mouth_left.x), int(face.face_landmarks.mouth_left.y)
            mouth_right = int(face.face_landmarks.mouth_right.x), int(face.face_landmarks.mouth_right.y)
            lip_bottom = int(face.face_landmarks.under_lip_bottom.x), int(face.face_landmarks.under_lip_bottom.y)
            draw.rectangle((mouth_left, (mouth_right[0], lip_bottom[1])), outline='yellow', width=2)
            cv2.rectangle(imagem, (mouth_left), (mouth_right[0], lip_bottom[1]), (0, 0, 255), 2)
            #img.show()

    cv2.imshow('teste', imagem)
    nome = 'rosto_detectado_foto' + str(contador) + '.jpg'
    #Caso ainda não tenha tirado 10 fotos
    if contador <=10 and len(response_detected_faces)>0:
        print("taking photo...")
        #Salvar as fotos em um diretório específico
        cv2.imwrite(os.path.join(path , nome), imagem)
        print(f"photo {contador}, captured!")
        contador += 1

    #Encerrar o processo após tirar 10 fotos
    if contador == 10:
        print("taking photo...")
        print(f"photo {contador}, captured!")
        print("Finished!")
        break

    cv2.waitKey(1)