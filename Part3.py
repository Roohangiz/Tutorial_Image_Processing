# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:19:15 2024

@author: Roohi
"""
#%% Basic Operations on Images
import cv2 as cv
img = cv.imread('Nature.jpg',1)
imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgcrop = imggray[100:300, 250:350]

cv.imshow('image', img)
cv.imshow('image_gray', imggray)
cv.imshow('image_crop', imgcrop)
cv.waitKey()
cv.destroyAllWindows()

A_piece_img = img[500:600,500:600]
img[0:100,0:100] = A_piece_img
cv.imshow('New-image', img)
cv.waitKey()
cv.destroyAllWindows()

#BGR
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
cv.imshow('blue', b)
cv.imshow('green', g)
cv.imshow('red', r)
cv.waitKey()
cv.destroyAllWindows()

img[:,:,0] = 0
cv.imshow('image', img)
cv.waitKey()
cv.destroyAllWindows()

b,g,r = cv.split(img)
img1 = cv.merge((b,g,r))

