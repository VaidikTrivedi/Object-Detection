# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:16:17 2018

@author: Vaidik
"""

import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import pyttsx3
import os
from PIL import Image
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/trainningData.yml')
face_cascade_path = 'F:/Python_Projects/Face_Recognition/haarcascade_frontalface_default.xml'
eye_cascade_path = 'F:/Python_Projects/Face_Recognition/haarcascade_eye.xml'
face_path = 'F:/Python_Projects/Object Detection/darkflow-master/Faces/'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

names = ['', 'Vaidik', 'Dharik', 'Kafil', 'Sunil', 'Parth', 'Drumil','','','','Unknown']
persons = []
usedIds = []
numbers = [0]
data_path = 'F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/dataSet/'
imgcount = [os.path.join(data_path,f) for f in os.listdir(data_path)]
Id = 1
count = 0
for Id in range(1, 11):
    for lastCount in imgcount:
        temp = os.path.split(lastCount)[1].split('user.')[1]
        if(temp[1]=='.'):
            temp2 = temp[0]
        else:
            temp2 = temp[0:2]
        if(Id==int(temp2)):
            count = count + 1
    numbers.append(count)
    count = 0
    
def faceRecogition():
    global names
    print("Number Of Persons: ",persons)
    for face in persons:
        count = 1
        path = face_path + face
        print("\nReading from path: ", path, "\n")
        img = cv2.imread(path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        #print("Face: ", faces)
        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 5)
            Id, conf = recognizer.predict(gray_img[y:y+h, x:x+w])
            print("Sr. No: ", count, "   ||  Id: ", Id)
            conf = 100-conf
            if(conf>65):
                name = names[Id]
            elif(conf<0):
                name = "Noise"
                print("Undefined Image, Cannot defined.")
            else:
                name = "Unknown"
                Id="Unknown"
            print("Person: ", name, "   ||  Confidence: ", conf)
            cv2.putText(img, str(name), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            count+=1
            tempimg = cv2.resize(img, (500, 800))
            cv2.imshow(str(name), tempimg)
            engine.say(str(name))
            engine.runAndWait()
        print("\nDeleting", path , "image from directory\n")
        try:
            os.remove(path)
        except: pass
#    show_img = cv2.resize(img, (900,900))
#    cv2.imshow("Image", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
    return

def faceTrainer(path):
    imagePath = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for image in imagePath:
        faceImg = Image.open(image).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        Id = int(os.path.split(image)[-1].split('.')[1])
        faces.append(faceNp)
        print("Id: ", Id, "Person: ", names[Id])
        Ids.append(Id)
        cv2.imshow("Training...", faceNp)
        cv2.waitKey(10)
    return Ids, faces

def trainAgain(path):
    data_path = 'F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/dataSet/'
    #imgcount = [os.path.join(data_path,f) for f in os.listdir(data_path)]
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.5, 5)
    count = 0
    print("Person | ID")
    for i in names:
        print( i, count)
        count = count + 1
    for (x, y, w, h) in faces:
        count = 0
        crop_img = img[y:y+h, x:x+w]
        cv2.imshow("Id", crop_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if input("This was Correct? [y/n] ") == 'y':
            continue
        Id = int(input("Enter Id: "))
        count = numbers[Id]
        count+=1
        numbers[Id] = count
        print("Photo Saved by this count: ",count)
        temp = str(count)
        cv2.imwrite(os.path.join(data_path, 'user.'+str(Id)+'.'+ str(temp)+'.jpg'), crop_img)
    path = 'F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/dataSet/'
    Ids,faces=faceTrainer(path)
    recognizer.train(faces,np.array(Ids))
    recognizer.write('F:/Python_Projects/Face_Recognition/Face-Recognition-Train-YML-Python-master/trainningData.yml')
    cv2.destroyAllWindows()
    return

engine = pyttsx3.init()
option = {'model': 'cfg/yolo.cfg',
          'load': 'bin/yolo.weights',
          'thresold': 0.5,
          'gpu': 1.0
          }

tfnet = TFNet(option)
choice = 'y'
while(choice == 'y'):
    path = input("Enter Path: ")
    img = cv2.imread(path)
    image = img.copy()
    result = tfnet.return_predict(img)
    print(result)
    print(len(result))
    base = 0
    for j in result: 
        tl = (result[base]['topleft']['x'], result[base]['topleft']['y'])
        br = (result[base]['bottomright']['x'], result[base]['bottomright']['y'])
        bl = (result[base]['topleft']['x'], result[base]['bottomright']['y'])
        label = (result[base]['label'])
        conf = (result[base]['confidence'])
        print("\n\ntl: ",tl, " br: ", br, " label: ", label, "Confidence: ", conf)
    
        if(conf>0.25):
            img = cv2.rectangle(img, tl, br, (0, 255, 0), 5)
            scale = (result[base]['topleft']['y']) - (result[base]['bottomright']['y'])
            img = cv2.putText(img, label, bl, cv2.FONT_HERSHEY_COMPLEX, -(scale/200), (255, 255, 0), 2)
            plt.imshow(img)
            engine.say(str(label))
            engine.runAndWait()
            plt.show()
            tempImg = cv2.resize(img, (900, 900))
            cv2.imshow("Object Detected", tempImg)
            k = cv2.waitKey(5)
            if(label=="person" and conf>=0.5):
                print("Person Found at: Top left: ", tl, " Bottom Right: ", br)
                crop = image[result[base]['topleft']['y']: result[base]['bottomright']['y'], result[base]['topleft']['x']: result[base]['bottomright']['x']]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
                faces = face_cascade.detectMultiScale(gray, 1.5, 3)
                if(len(faces)>0):
                    print("Face Found!")
                    count = count+1
                    name = str(count)+'.jpg'
                    cv2.imwrite(os.path.join(face_path, name), crop)
                    persons.append(name)
                else:
                    print("Face not found")
                    
        base+=1
    img = cv2.resize(img, (1000, 800))
    cv2.imshow("Objects...", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if(len(persons)==0):
        print("No Known Face Found!")
    else:
        print("REcognitiong Face....\n")
        faceRecogition()
    learn = input("False Result? Want to train?[y/n]")
    if(learn=='y'):
        trainAgain(path)
    else:
        print("Yeah!")
    print("\n\nDo you want to detect more objects??[y/n]")
    choice = input("Entre Your choice: ")
    if(choice=='y'):
        persons = []
        count = 0