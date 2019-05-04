# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:58:29 2018

@author: Vaidik
"""

import cv2
from darkflow.net.build import TFNet
import time
import numpy as np

option = {'model': 'cfg/yolo.cfg',
          'load': 'bin/yolo.weights',
          'thresold': 0.5,
          'gpu': 1.0
          }
                                         
tfnet = TFNet(option)

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

while True:
    stime = time.time()
    rect, frame = capture.read()
    results = tfnet.return_predict(frame)
    if rect:
        for color, result in zip(colors, results):
             tl = (result['topleft']['x'], result['topleft']['y'])
             br = (result['bottomright']['x'], result['bottomright']['y'])
             label = (result['label'])
             conf = (result['confidence'])
             if(conf>0.5):
                 text = '{}: {:.0f}%'.format(label, conf * 100)
                 frame = cv2.rectangle(frame, tl, br, color, 5)
                 frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
             
        cv2.imshow("Frame", frame)
        print("FPS: {:.1f}".format(1 / (time.time() - stime)))
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()