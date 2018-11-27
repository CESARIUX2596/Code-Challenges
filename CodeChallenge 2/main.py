import numpy as np
import cv2
import time
# add database to res/ folder inside this one
def GreenRemover(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
	lower_green = np.array([0,10,75])
	upper_green = np.array([96,255,255])
	lower_white = np.array([0,0,100])
	upper_white = np.array([180,45,255])
	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask2 = cv2.inRange(hsv, lower_white, upper_white)
	mask3 = cv2.bitwise_or(mask,mask2)
	mask3 = 255 - mask3
	return mask3


def main():
	# img = cv2.imread('image.png')
	count = 1
	kernel = np.ones((3,3),np.uint8)
	while True:
		img_name = 'res/girl/'+ '{:05}'.format(count) + '.jpg'
		frame = cv2.imread(img_name, 1)
		mask = GreenRemover(frame)
		opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
		sure_bg = cv2.dilate(opening,kernel,iterations=3)
		dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
		ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
		sure_fg = np.uint8(sure_fg)
		unknown = cv2.subtract(sure_bg,sure_fg)
		ret, markers = cv2.connectedComponents(sure_fg)
		markers = markers+1
		markers[unknown==255] = 0
		markers = cv2.watershed(frame,markers)
		frame[markers == -1] = [0,0,255]
		cv2.imshow('frame', frame)
		cv2.imshow('mask', mask)
		#cv2.imshow('filtered', markers)
		time.sleep(0.2)
		count += 1
		
		#out = GreenRemover(frame)
		#cv2.imshow('Color Filter', out)
		

		
		#blured = cv2.GaussianBlur(roi,(5,5),0)
		#gray = cv2.cvtColor(blured,cv2.COLOR_BGR2GRAY)
		#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		
		

		k = cv2.waitKey(5) & 0xFF 
		if k == 27 or count == 69:
			break

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()