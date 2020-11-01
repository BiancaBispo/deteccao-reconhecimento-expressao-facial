import cv2
import os
import numpy as np
from PIL import Image
import pickle

'''
Código fonte original disponível em: https://github.com/codingforentrepreneurs/OpenCV-Python-Series/tree/master/src

    A modificação do código fonte será representada da seguinte forma abaixo:
#------------------------------------------------------------------------------#
                            ALTERAÇÕES FEITAS
#------------------------------------------------------------------------------#
'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__ ))
image_dir = os.path.join(BASE_DIR, "images")

#--------------------------------------------------------------------------------#
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#--------------------------------------------------------------------------------#

#Utilizando o LBPHF
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

#------------------------------------------------------------------------------#
print('\n Começando o treinamento de cada imagem localiza na pasta: \n')
#------------------------------------------------------------------------------#
for root, dirs, files in os.walk(image_dir):
    for file in files:
#------------------------------------------------------------------------------#        
        if file.endswith("tiff") or file.endswith("jpg") or file.endswith("png") or file.endswith("gif"):
#------------------------------------------------------------------------------#
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" \n","-").lower()

            #informar/imprimir a pasta/diretório que as imagens estão armazenadas
           # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #imprime o nome das pessoas que estão armazenadas na pasta/diretorio
            print(label_ids)
                
            #y_label.append(label) #Some number
            #x_train.append(path) # verify this image, turn inro a NumPy array, gray
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            #imprime a detecção de cada face
            print(image_array)
#------------------------------------------------------------------------------#            
            faces = face_cascade.detectMultiScale(image_array,1.3,5,flags=cv2.CASCADE_SCALE_IMAGE)
#------------------------------------------------------------------------------#
            #achou uma face? recorte ela (crop)
            for(x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
#------------------------------------------------------------------------------#
                smile = smile_cascade.detectMultiScale(roi,scaleFactor= 1.16,minNeighbors=35,minSize=(25,25), flags=cv2.CASCADE_SCALE_IMAGE)
                eyes = eye_cascade.detectMultiScale(roi,scaleFactor= 1.16,minNeighbors=35,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)
#------------------------------------------------------------------------------#
#print(y_labels)    
#print(x_train)


with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner_70-30_FacialExpressao.xml")
