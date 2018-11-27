import numpy as np
import cv2

def ColorFilter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([37,110,85])
    upper_green = np.array([90,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    obstacle_detected = cv2.bitwise_not(frame, frame, mask = mask)
    #obstacle_detected = cv2.cvtColor(obstacle_detected, cv2.COLOR_BGR2RGB)
    return obstacle_detected
    
def Canny(frame):
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray_img,(3,3),0)
    out = cv2.Canny(blured,120,200)
    return out


def AreaOfInterest(frame):
    roi = frame[85:430,257:395]
    return roi

input = cv2.imread("res/girl/00001.jpg",1)
#out = ColorFilter(taco)
roi = AreaOfInterest(input)
gray_img = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
sure_bg = cv2.dilate(thresh,kernel,iterations=3)
dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#color_filtered = ColorFilter(roi)

#cv2.imshow('out',color_filtered)
cv2.imshow('taco', sure_fg)
cv2.waitKey(0)