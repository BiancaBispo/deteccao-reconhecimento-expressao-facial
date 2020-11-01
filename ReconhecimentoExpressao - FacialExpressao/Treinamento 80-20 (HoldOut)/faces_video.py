import numpy as np
import cv2 as cv
import pickle
import os

'''
Código fonte original disponível em: https://github.com/maelfabien/Machine_Learning_Tutorials/blob/master/1_Computer%20Vision/FaceDetection/FaceDetectionFull.ipynb

E em: https://github.com/codingforentrepreneurs/OpenCV-Python-Series/blob/master/src/faces.py


    A modificação do código fonte será representada da seguinte forma abaixo:
#------------------------------------------------------------------------------#
                            ALTERAÇÕES FEITAS
#------------------------------------------------------------------------------#
'''

#------------------------------------------------------------------------------#
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')
#------------------------------------------------------------------------------#


                        #Retirado do codingforentrepreneurs#
#------------------------------------------------------------------------------#
#Reconher as expressões
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainner_80-20_FacialExpressao.xml")

#Colocar nome 
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

#Fonte da letra
font = cv.FONT_HERSHEY_SIMPLEX

#Carregar video
video_capture = cv.VideoCapture("FelizBI 2.mp4")

#Carregando WebCam
#video_capture = cv.VideoCapture(0)

while(True):
    # capturar frame por frame
    ret, image = video_capture.read()

    #Colocar a imagem em cinza
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #Para cada rostos detectados
    faces = face_cascade.detectMultiScale(gray,1.3,5,flags=cv.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        image = cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)    
        roi_gray = gray[y:y+h, x:x+w]

        #Janelilha que aparece da captura só do rosto (em preto e branco)
        #cv.imshow("roi_gray", roi_gray)
        roi_color = image[y:y+h, x:x+w]

        #Colocar nome na borda do detector
        id_, conf = recognizer.predict(roi_gray)
        name = labels[id_]
        if conf >= 45 and conf <= 85:
            #Letra retangulo da fonte
            color = (255, 255, 255)
            stroke = 2
            cv.putText(image, name,(x,y), font, 1, color, stroke, cv.LINE_AA)
            print(labels[id_])

#------------------------------------------------------------------------------#
                                #retirado do maelfabien#
       #Para boca  
        smile = smile_cascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.16,
        minNeighbors=35,
        minSize=(25, 25),
        flags=cv.CASCADE_SCALE_IMAGE)

        for (sx, sy, sw, sh) in smile:
            cv.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (0,0,255), 2)
            cv.putText(image,'Boca',(x + sx,y + sy), 1, 1, (255, 255, 255), 2)

        #Para olhos
        eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.16,
        minNeighbors=35,
        minSize=(25, 25),
        flags=cv.CASCADE_SCALE_IMAGE)
        
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv.putText(image,'Olho',(x + ex,y + ey), 1, 1, (255, 255, 255), 2)

    #Visualizar
    cv.imshow("video",image)
    #Se apertar 'q' para de executar
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
#------------------------------------------------------------------------------#
cv.destroyAllWindows()
