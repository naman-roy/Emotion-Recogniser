#Emotion Recogmiser


import cv2
from deepface import DeepFace
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


#using live video feed to read emotions
cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('Cannot Open Webcam')

while True:
    ret,frame=cap.read()
    result=DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)

    #Gracode method to put an outline to the region covering face
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    #Using fonts to showcase the resulting emotion
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,result['dominant_emotion'],(50,50),font,3,2,cv2.LINE_4)
    cv2.imshow('Video',frame)
    
    #Press q to exit the Program and Colse The Live Cam
    if cv2.waitKey(2)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
