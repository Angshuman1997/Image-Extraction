# Import relevant libraries
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
image = cv2.imread(r'C:\Users\Angshuman Bardhan\Desktop\PI.jpg', -1)
# convert to gray and binarize
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
# note: erosion and dilation works on white forground
binary_img = cv2.bitwise_not(binary_img)
# dilate the image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
dilated_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel,iterations=1)
# find contours, discard contours which do not belong to a rectangle
(cnts, _) = cv2.findContours(dilated_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
sq_cnts = []  # contours of interest to us
print(str(len(cnts)))
for cnt in cnts:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    #cv2.drawContours(image,[approx],0,(0,255,0),1)
    x=approx.ravel()[0]
    y=approx.ravel()[1]
    if len(approx)==4:
        x, y, w, h = cv2.boundingRect(cnt)
        aspectratio=float(w)/h
        if aspectratio >=1.0 and aspectratio<=1.20:
            if x>220 and x!=395:
                sq_cnts.append(approx)
                              
print(len(sq_cnts))#No. of Selected Squares which are the green boxes which is 12
f=""
directory = r'C:\Users\Angshuman Bardhan\Desktop\TEST FOLDER'#for new folder location
cv2.drawContours(image,sq_cnts,-1,(0,255,0),1)#outline in green color
cv2.imshow("GREEN BOXES",image)#bordering the green boxes and displaying
for i in range(len(sq_cnts)):
    (x, y, w, h) = cv2.boundingRect(sq_cnts[i])
    print(x,y,w,h)
    newimg=image[y:y+h,x:x+w] # crop the image
    newimg=cv2.resize(newimg,(128,128))#resizing the image 
    f=str(i)+".jpg"
    os.chdir(directory)
    cv2.imwrite(f, newimg)# writing the image in a different directory

