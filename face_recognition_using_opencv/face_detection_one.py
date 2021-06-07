import cv2

face = cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, videocapture = face.read()
    videocapture = cv2.cvtColor(videocapture,0)
    detections = cascade_classifier.detectMultiScale(videocapture)
    if len(detections) > 0:
        (x,y,a,b) = detections[0]
        videocapture = cv2.rectangle(videocapture,(x,y),(x+a,y+b),(255,0,0),3)
    cv2.imshow('Face Window',videocapture)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
face.release()

