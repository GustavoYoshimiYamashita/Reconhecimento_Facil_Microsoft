#importando a biblioteca OpenCV
import cv2

#Criando um classificador
classificador = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#Definindo a fonte da letra que será imprimida na tela
fonte = cv2.FONT_HERSHEY_SIMPLEX

#Habilitando a WebCam conectada ao dispositivo
camera = cv2.VideoCapture(0)

img_counter = 0

#Loop para a detecção tempo real
while 1:
    #Fazendo a leitura da imagem
    conectado, imagem = camera.read()

    #Convertendo a imagem para a escala de cinza
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    #Atribuindo as classificações a variável facesDetectadas
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
    scaleFactor=1.5,
    minSize=(50, 50))
    #Nas faces dectadas, desenhar um retângulo e escrever Humano
    for (x, y, l, a) in facesDetectadas:
        #cv2.rectangle(imagem, (x,y), (x+l, y+a), (0,0,255), 2)
        face = imagem[y:y+a, x:x+l]
        #cv2.putText(imagem, 'Humano', (x, y + (a + 30)), fonte, 1, (0, 255, 255))
        cv2.imshow("test", face)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, face)
        print("{} written!".format(img_name))
        img_counter += 1

    if not conectado:
        print("failed to grab frame")
        break

    cv2.waitKey(1)


camera.release()
cv2.destroyAllWindows