# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:28:30 2019

@author: Sahil
"""

import numpy as np
import os
import cv2
import pyautogui
import imutils.paths as paths
import face_recognition
import pickle
import imutils


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
path = "E:\\Project\\dataset\\"# path were u want store the data set


def data():  
    id = pyautogui.prompt(text="""Enter Username >""", title='Recognition', default='none')
    
    os.mkdir(path+str(id))
    
    sampleN=0;
    while 1:
    
        ret, img = cap.read()
        frame = img.copy()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
    
            sampleN=sampleN+1;
    
            cv2.imwrite(path+str(id)+ "\\" +str(sampleN)+ ".jpg", gray[y:y+h, x:x+w])
    
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
            cv2.waitKey(100)
    
        cv2.imshow('img',img)
    
        cv2.waitKey(1)
    
        if sampleN >14 :
    
            break
    
    cap.release()
    
    cv2.destroyAllWindows()
    
    
    
def train():
    dataset = "E:\\Project\\dataset\\"# path of the data set 
    module = "E:\\Project\\encodings\\encoding1.pickle" # were u want to store the pickle file 
    
    imagepaths = list(paths.list_images(dataset))
    knownEncodings = []
    knownNames = []
    for (i, imagePath) in enumerate(imagepaths):
        print("[INFO] processing image {}/{}".format(i + 1,len(imagepaths)))
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
        boxes = face_recognition.face_locations(rgb, model= "hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
           knownEncodings.append(encoding)
           knownNames.append(name)
           print("[INFO] serializing encodings...")
           data = {"encodings": knownEncodings, "names": knownNames}
           output = open(module, "wb") 
           pickle.dump(data, output)
           output.close()
      
        
def main(): 
    encoding = "E:\\Project\\encodings\\encoding1.pickle"
    data = pickle.loads(open(encoding, "rb").read())
    print(data)
    cap = cv2.VideoCapture(0)
  
    if cap.isOpened :
        ret, frame = cap.read()
    else:
         ret = False
    while(ret):
      ret, frame = cap.read()
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      rgb = imutils.resize(frame, width=400)
      r = frame.shape[1] / float(rgb.shape[1])

      boxes = face_recognition.face_locations(rgb, model= "hog")
      encodings = face_recognition.face_encodings(rgb, boxes)
      names = []
   
      for encoding in encodings:
                matches = face_recognition.compare_faces(np.array(encoding),np.array(data["encodings"]))
                name = "Unknown"
               
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                   
                    
                    for i in matchedIdxs:
                                  name = data["names"][i]
                                  counts[name] = counts.get(name, 0) + 1
                                  name = max(counts, key=counts.get)
                names.append(name)
                
      for ((top, right, bottom, left), name) in zip(boxes, names):
          top = int(top * r)
          right = int(right * r)
          bottom = int(bottom * r) 
          left = int(left * r)
          cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
          y = top - 15 if top - 15 > 15 else top + 15
          cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
      cv2.imshow("Frame", frame)
      k = cv2.waitKey(30)
      stop = ord("S") # To Stop press capital S (While on the output image)
      if k == stop:
          break
      
    cv2.destroyAllWindows()

    cap.release()

    
#Options checking
opt =pyautogui.confirm(text= 'Chose an option', title='Recognition', buttons=['Detection','Recognize','Exit'])
if opt == 'Detection':
    opt = pyautogui.confirm(text="""
    Please look at the Webcam.\nTurn your head a little while capturing.""", title='Recognition', buttons=['Ready'])
        
    if opt == 'Ready':
        data()
        train()
        opt =pyautogui.confirm(text= 'Chose an option', title='Recognition', buttons=['Recognize','Exit'])
        if opt == 'Recognize':
            main()
        if opt == 'Exit':
            print("Quit the app")

if opt == 'Recognize':
    main()
    
if opt == 'Exit':
    print("Quit the app")
    
