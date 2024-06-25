
"""
Created on Tue Jun 25 10:50:30 2024

@author: Roohi
"""
#%% Basic Operations on Images
import numpy as np
import cv2 as cv
img1 = cv.imread('C:/Roohi/Git/Tutorial/Tutorial_ImageProcessing/Blank.jpg',1)
cv.line(img1, (50,50), (250,250), (150,100,0),20)
cv.rectangle(img1, (50,50),(250,250),(0,50,100), 10)
cv.circle(img1, (150,150),100, (0,200,170),5)
points=np.array([[100,100],[400,400],[400,300],[100,300]],np.int32)
cv.polylines(img1, [points], True, (200,170,20),15)
cv.imshow('Shapes', img1)
cv.waitKey(0)
cv.destroyAllWindows()

#%% Text
img2 = cv.imread('C:/Roohi/Git/Tutorial/Tutorial_ImageProcessing/Nature.jpg',1)
font = cv.FONT_HERSHEY_COMPLEX
cv.putText(img2, 'A perfect landscape', (50,50), font, 1, (200,200,200),2, cv.LINE_4)
cv.imshow('nature', img2)
cv.waitKey(0)
cv.destroyAllWindows()