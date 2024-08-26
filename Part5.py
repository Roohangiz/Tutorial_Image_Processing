# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:51:13 2024

@author: Roohi
"""
#%% Image Thresholding
import numpy as np
import cv2 as cv

# Load the input image in color
img = cv.imread('BrainMRI.jpeg')

# Convert the color image to grayscale
imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Basic Binary Thresholding
ret1, thresh1 = cv.threshold(imggray, 70, 255, cv.THRESH_BINARY)

# Otsu's Thresholding
# Automatically determines the optimal threshold value 
ret2, thresh2 = cv.threshold(imggray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Adaptive Thresholding with a 35x35 Gaussian-weighted neighborhood, subtracting 1 from the result
thresh3 = cv.adaptiveThreshold(imggray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, 1)

# Display the original image
cv.imshow('Original', img)

# Display the result of the basic binary thresholding
cv.imshow('Threshold', thresh1)

# Display the result of Otsu's thresholding
cv.imshow('Threshold OTSU', thresh2)

# Display the result of adaptive thresholding
cv.imshow('Threshold adaptive', thresh3)

# Wait for any key to be pressed, then close all displayed windows
cv.waitKey(0)
cv.destroyAllWindows()



