import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

def canny(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img,(7,7),0)
    img = cv2.Canny(img,120,200)
    return img

img = cv2.imread('res/swan/00000.jpg')
print(img[10,10,:])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([0,120,51])
upper_blue = np.array([130,255,150])
blue = img.copy()
blue[:,:] = lower_blue
##cv2.imshow('blue',blue)
mask = cv2.inRange(hsv, lower_blue, upper_blue)
##imgGreen = img.copy()
##imgGreen[:, :, 1] = 0
cv2.imshow('mask',mask)
##cv2.imshow('canny', cannyImg)
##cv2.imshow('imgGreen', imgGreen)
##img[:65,:]=0;
##img[400:,:]=0;
##img[:,:133]=0;
##img[:,512:]=0;
imgC= img[65:400,133:512]
##imgC = img.copy()
##imgNotGreen = -(mask-255)
##cv2.imshow('imgNotGreen',imgNotGreen)
##img[mask ==255] = np.array([0,0,0])

gray = cv2.cvtColor(imgC,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


cv2.imshow('img',img)

cv2.imshow('thresh',thresh)
##img[:133,:60]=0;

##cv2.imshow('original',thresh)
# noise removal
kernel = np.ones((4,4),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
##cv2.imshow('dilte',sure_bg)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(imgC,markers)

imgC[markers == -1] = [0,0,255]
cv2.imshow('1',imgC)
cv2.waitKey(0)
####cannyImg = canny(img)
##cv2.imshow('cannyImg',cannyImg)
##plt.imshow(markers)
##plt.imshow(img)
##plt.show()
##cv2.destroyAllWindows()
