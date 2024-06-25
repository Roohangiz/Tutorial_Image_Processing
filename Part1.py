
"""
Created on Mon Jun  3 17:14:19 2024

@author: Roohi
"""
#%% Read and Display an image (OpenCV_Library)
import cv2 as cv
img = cv.imread('C:/Roohi/Git/Tutorial/Tutorial_ImageProcessing/Nature.jpg', cv.IMREAD_COLOR)
cv.imshow('Nature_Poster', img)
cv.waitKey(0)
cv.destroyAllWindows()

#cv.IMREAD_GRAYSCALE      0
#cv.IMREAD_COLOR          1
#cv.IMREAD_UNCHANGED     -1

#%% Save an image (OpenCV_Library)
img = cv.imread('C:/Roohi/Git/Tutorial/Tutorial_ImageProcessing/Nature.jpg', 0)
cv.imshow('Color_poster', img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('C:/Roohi/Git/Tutorial/Tutorial_ImageProcessing/Nature_gray.jpg', img)

#%% Display an image (Matplotli_Library)
from matplotlib import pyplot as plt
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.show() 
