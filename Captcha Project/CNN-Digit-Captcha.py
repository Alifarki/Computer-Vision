import cv2
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import load_model


net=load_model("digi_classifier.h5")
img=cv2.imread("pic.jpg")
img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
T,thresh=cv2.threshold(img1,120,255,cv2.THRESH_BINARY_INV) #convert to binary
cnts,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find contours
for i in range(len(cnts)):
    x,y,w,h=cv2.boundingRect(cnts[i])
    roi=img[y-5:y+h+5,x-5:x+w+5]
    roi=cv2.resize(roi, (32, 32))
    roi=roi/255.0
    roi=np.array([roi])
    output=net.predict(roi)[0]
    max_index=np.argmax(output)+1
    print(max_index)
    cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
    cv2.putText(img,str(max_index),(x-5,y+h+10),cv2.FONT_HERSHEY_SIMPLEX,0.95,(0,255,0),2)
cv2.imshow("Pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()