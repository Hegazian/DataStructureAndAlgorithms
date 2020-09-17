# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:03:24 2020

@author: lenovo
"""

import cv2
import numpy as np

def Canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    bluer = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(bluer,50,150)
    return canny

def region_of_interset(image):
    poly = np.array([(200,700),(1100,700),(550,250)])
    mask = np.zeros_like(image)
    cv2.fillConvexPoly(mask,poly,255)
    masked_image = cv2.bitwise_and(canny,mask)
    return masked_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image,(x1 ,y1),(x2, y2),(0,255,0),10)
    return line_image

def make_coordinates(image,line_parameters):
    slope, intersect = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intersect)/slope)
    x2 = int((y2-intersect)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intersect(image,lines):
    left_line = []
    right_line = []
    for line in lines :
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intersect = parameters[1]
        if slope < 0:
            left_line.append((slope,intersect))
        else:
            right_line.append((slope,intersect))
    lift_slope_average = np.average(left_line,axis=0)
    right_slope_average = np.average(right_line,axis=0)
    lift__line = make_coordinates(image,lift_slope_average)
    right__line = make_coordinates(image,right_slope_average)
    return np.array([lift__line,right__line])

    
#image = cv2.imread('test_image.jpg')
#copyImage = np.copy(image)
#
#canny = Canny(copyImage)
#mask = region_of_interset(canny)
#lines = cv2.HoughLinesP(mask,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
#average_lines = average_slope_intersect(copyImage,lines)
#line_image = display_lines(copyImage,average_lines)
#cobo_image = cv2.addWeighted(copyImage,0.7,line_image,1,1)

#cv2.imshow("result1",cobo_image)
#cv2.waitKey(0)

video = cv2.VideoCapture("test2.mp4")

while(video.isOpened()):
    _, frame = video.read()
    canny = Canny(frame)
    mask = region_of_interset(canny)
    lines = cv2.HoughLinesP(mask,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    average_lines = average_slope_intersect(frame,lines)
    line_image = display_lines(frame,average_lines)
    cobo_image = cv2.addWeighted(frame,0.7,line_image,1,1)
    cv2.imshow("result1",cobo_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

