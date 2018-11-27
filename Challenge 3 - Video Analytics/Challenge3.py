import cv2
import numpy as np

def FieldFilter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_green = np.array([37,110,22])
    upper_green = np.array([87,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = 255 - mask
    obstacle_detected = cv2.bitwise_and(frame, frame, mask = mask)
    #obstacle_detected = cv2.cvtColor(obstacle_detected, cv2.COLOR_BGR2RGB)
    return obstacle_detected

def TimFinderBlu(frame):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    lower_green = np.array([90,128,50])
    upper_green = np.array([135,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # obstacle_detected = cv2.bitwise_and(frame, frame, mask = mask)
    # obstacle_detected[mask == 255]
    #obstacle_detected = cv2.cvtColor(obstacle_detected, cv2.COLOR_BGR2RGB)
    return mask

def TimFinderRed(frame):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    lower_red = np.array([0,100,0])
    upper_red = np.array([20,255,255])
    
    lower_white = np.array([0,0,100])
    upper_white = np.array([180,45,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    #obstacle_detected = cv2.bitwise_and(frame, frame, mask = mask)

    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    mask3 = cv2.bitwise_or(mask,mask2)
    # obstacle_detected = cv2.bitwise_and(frame, frame, mask = mask3)
    #obstacle_detected[mask == 255]
    #obstacle_detected = cv2.cvtColor(obstacle_detected, cv2.COLOR_BGR2RGB)
    return mask3


def pitagorazo(frame):
    triangle = np.array([[0, 0], [1080, 0], [0, 51]])
    color = [0, 0, 0] #white
    cv2.fillConvexPoly(frame, triangle, color)
    return frame
def pitagorazoIzquierdo(frame):
    frameP = frame.copy()
    triangle = np.array([[0, 0], [543, 23], [765, 655],[0,720]])
    color = [0, 0, 0] #white
    cv2.fillConvexPoly(frameP, triangle, color)
    return frameP

def pitagorazoDerecho(frame):
    frameP = frame.copy()
    triangle = np.array([[535, 23], [1080, 0], [1080, 655], [750, 655]])
    color = [0, 0, 0] #white
    cv2.fillConvexPoly(frameP, triangle, color)
    return frameP

# funcion que tome la imagen, aplique filtros de los equipos y cuente cuantos hay en total, se va a llamar en cada uno de los metodos para izq o der
# parametro: la imagen de la region de interes 
# regrese el total de objetos identificados
def Counter(frame):
    # cv2.imshow('fr',frame)
    count = 0
    maskRed = TimFinderRed(frame)
    maskBlu = TimFinderBlu(frame)
    # cv2.imshow('blu',maskBlu)
    # cv2.imshow('red',maskRed)
    count += Count(maskRed)
    count += Count(maskBlu)
    return count

def Count(mask):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000 and area > 25:
            # cv2.drawContours(filtered, contour, -1, (0, 255, 0), 3)
            count += 1
    return count


cap = cv2.VideoCapture('output.mp4')
kernelE = np.ones((2,2),np.uint8)
kernelD = np.ones((3,3),np.uint8)
totalIzq = 0
totalDer = 0
pcIzq = 0
pcDer = 0
while(1):
    numIzq = 0
    numDer = 0
    ret, frame = cap.read()
    frame = frame[70:540,:1080]
    frame = pitagorazo(frame)
    cv2.imshow("frame2",frame)
    filtered = FieldFilter(frame)
    frameDer = pitagorazoDerecho(filtered)
    frameIzq = pitagorazoIzquierdo(filtered)
    cv2.imshow("mask",filtered)
    eRojo = TimFinderRed(filtered)
    eAzul = TimFinderRed(filtered)  
    cv2.imshow("eRojo",eRojo)
    cv2.imshow("eAzul",eAzul)
    numIzq = Counter(frameIzq)
    numDer = Counter(frameDer)
    if(numIzq < numDer):
        totalIzq +=1
    elif(numDer < numIzq):
        totalDer+=1
    total = 1
    total = totalIzq + totalDer +1
    pcIzq = 100*totalIzq//total
    pcDer = 100*totalDer//total
    print("izq: " + str(pcIzq)+ "%   "+"\t"+"der: " + str(pcDer)+ "%")
    # print()
 
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
print("izq: " + str(totalIzq))
print("der: " + str(totalDer))
