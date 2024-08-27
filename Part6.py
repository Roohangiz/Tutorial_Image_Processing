# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:57:53 2024

@author: Roohi
"""
#Morphological operations
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#%% Erosion
Original_img = cv.imread('Fingerprint.png', 0)
_, mask =cv.threshold(Original_img, 90, 255, cv.THRESH_BINARY)

kernel1 = np.ones((2,2), np.uint8)
Eroded_img = cv.erode(mask, kernel1, iterations=1)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 40))
cmap_val = 'gray'

ax1.axis('off')
ax1.title.set_text('Original_img')

ax2.axis('off')
ax2.title.set_text('mask')

ax3.axis('off')
ax3.title.set_text('Eroded_img')

ax1.imshow(Original_img, cmap=cmap_val)
ax2.imshow(mask, cmap=cmap_val)
ax3.imshow(Eroded_img, cmap=cmap_val)

plt.show()
#%% Dilation
kernel2 = np.ones((3,3), np.uint8)
Dilated_img = cv.dilate(mask, kernel2, iterations=1)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 40))
cmap_val = 'gray'

ax1.axis('off')
ax1.title.set_text('Original_img')

ax2.axis('off')
ax2.title.set_text('mask')

ax3.axis('off')
ax3.title.set_text('Dilated_img')

ax1.imshow(Original_img, cmap=cmap_val)
ax2.imshow(mask, cmap=cmap_val)
ax3.imshow(Dilated_img, cmap=cmap_val)

plt.show()
#%% Closing
Original_image = cv.imread('Cells.jpg', 0)
_, Mask =cv.threshold(Original_image, 20, 255, cv.THRESH_BINARY)

kernel3 = np.ones((9,9), np.uint8)
Closed_img = cv.morphologyEx(Mask, cv.MORPH_CLOSE, kernel3)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 40))
cmap_val = 'gray'

ax1.axis('off')
ax1.title.set_text('Original_image')

ax2.axis('off')
ax2.title.set_text('Mask')

ax3.axis('off')
ax3.title.set_text('Closed_img')

ax1.imshow(Original_image, cmap=cmap_val)
ax2.imshow(Mask, cmap=cmap_val)
ax3.imshow(Closed_img, cmap=cmap_val)

plt.show()
#%% Opening
kernel4 = np.ones((7,7), np.uint8)
Opened_img = cv.morphologyEx(Mask, cv.MORPH_OPEN, kernel4)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 40))
cmap_val = 'gray'

ax1.axis('off')
ax1.title.set_text('Original_image')

ax2.axis('off')
ax2.title.set_text('Mask')

ax3.axis('off')
ax3.title.set_text('Opened_img') 

ax1.imshow(Original_image, cmap=cmap_val)
ax2.imshow(Mask, cmap=cmap_val)
ax3.imshow(Opened_img, cmap=cmap_val)

plt.show()
#%% Gradient
kernel5 = np.ones((5,5), np.uint8)
Gradient = cv.morphologyEx(Mask, cv.MORPH_GRADIENT, kernel4)

Dilated_img = cv.dilate(Mask, kernel5, iterations=1)
Subtract_img = Dilated_img - Mask 

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(30, 40))
cmap_val = 'gray'

ax1.axis('off')
ax1.title.set_text('Original_image')

ax2.axis('off')
ax2.title.set_text('Mask')

ax3.axis('off')
ax3.title.set_text('Gradient') 

ax4.axis('off')
ax4.title.set_text('Subtract_img')

ax1.imshow(Original_image, cmap=cmap_val)
ax2.imshow(Mask, cmap=cmap_val)
ax3.imshow(Gradient, cmap=cmap_val)
ax4.imshow(Subtract_img, cmap=cmap_val)

plt.show()

