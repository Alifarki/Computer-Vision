# from cv2 import cv2
# import cv2
# img=cv2.imread("pic.jpg")
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert 3 channel to 1 channel(gray)
# T,thresh=cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV) #using thresh old inverse for change 0to255 and controverse
# # print(T)
# cv2.imshow("Image",thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
# from cv2  import cv2
img=cv2.imread("pic.jpg")

img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

T,thresh=cv2.threshold(img1,120,255,cv2.THRESH_BINARY_INV) #convert to binary

cnts,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find contours
# print(len(cnts))
# cv2.drawContours(img,cnts,-1,(0,255,0),2)
for i in range(len(cnts)):
    x,y,w,h=cv2.boundingRect(cnts[i])
    roi=img[y-5:y+h+5,x-5:x+w+5]
    # cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)

    cv2.imshow("Pic",roi)
    cv2.waitKey(0)
cv2.destroyAllWindows()
