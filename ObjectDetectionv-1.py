# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:13:45 2018

@author: Vaidik
"""

import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import pyttsx3

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
    result = tfnet.return_predict(img)
    print(result)
    print(len(result))
    base = 0
    for j in result:   
        tl = (result[base]['topleft']['x'], result[base]['topleft']['y'])
        br = (result[base]['bottomright']['x'], result[base]['bottomright']['y'])
        label = (result[base]['label'])
        conf = (result[base]['confidence'])
        print("\n\ntl: ",tl, " br: ", br, " label: ", label, "Confidence: ", conf)
        base+=1
        if(conf>0.25):
            img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
            img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 0), 8)
            plt.imshow(img)
            #engine.say(str(label))
            #engine.runAndWait()
            plt.show()
            tempImg = cv2.resize(img, (900, 900))
            cv2.imshow("Object Detected", tempImg)
            k = cv2.waitKey(1)
    img = cv2.resize(img, (1000, 800))
    cv2.imshow("Objects...", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n\nDo you want to detect more objects??[y/n]")
    choice = input("Entre Your choice: ")