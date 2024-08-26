# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:32:34 2024

@author: Roohi
"""
#%% Logical Operations on Images
import numpy as np
import cv2 as cv
img1 = cv.imread('image1.jpg')
img2 = cv.imread('image2.jpg')
#added = img1 + img2
added = cv.add(img1,img2)
#added = cv.addWeighted(img1, 0.8, img2, 0.2, 0)

img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 210, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

img1_m = cv.bitwise_and(img1, img1, mask = mask)
img2_m = cv.bitwise_and(img2, img2, mask = mask_inv)
imgadded = img1_m + img2_m

cv.imshow('Image1', img1)
cv.imshow('Image2', img2)
cv.imshow('Mask', mask)
cv.imshow('Mask_INV', mask_inv)
cv.imshow('Mask_img1', img1_m)
cv.imshow('Mask_img2', img2_m)
cv.imshow('Added 1', added)
cv.imshow('Added 2', imgadded)

cv.waitKey(0)
cv.destroyAllWindows()
