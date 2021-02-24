import cv2
import numpy as np

# Detect lane lines using Hough Transform
def hough_based_detect_line(image, threshold):
	# Using Hough Transform to detect eligible lanes
	houghLines = cv2.HoughLinesP(image, 1, np.pi/180, threshold, minLineLength=30, maxLineGap=10)
	return houghLines


# Find interception of the line
def find_interception(imageHeight, x1, y1, x2, y2):
	Y1 = y1
	X1 = x1
	Y2 = y2
	X2 = x2
	Y = imageHeight
	slope = float(Y2 - Y1)/float(X2 - X1)
	intercept = 0
	if slope != 0:
		intercept = float(Y-Y1)/slope + float(X1)
	return intercept

# Extract left or right lanes from lines
def find_slope(x1, y1, x2, y2):
	return float(y2-y1)/float(x2-x1)

def lane_decision(imageHeight, lines):
	leftLane = []
	rightLane = []
	for x1,y1,x2,y2 in lines:
		if x2-x1!=0:
			slope = find_slope(x1,y1,x2,y2)
			if slope >= 0:
				rightLane.append([x1,y1,x2,y2,find_interception(imageHeight,x1,y1,x2,y2)])
			else:
				leftLane.append([x1,y1,x2,y2,find_interception(imageHeight,x1,y1,x2,y2)])
	return leftLane, rightLane
