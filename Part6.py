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

#Load the original fingerprint image in grayscale
Original_img = cv.imread('Fingerprint.png', 0)

# Apply a binary threshold to convert the image to a binary mask 
_, mask = cv.threshold(Original_img, 90, 255, cv.THRESH_BINARY)

# Define a 2x2 kernel for erosion 
kernel1 = np.ones((2,2), np.uint8)

# Perform erosion operation on the binary mask with 1 iteration
Eroded_img = cv.erode(mask, kernel1, iterations=1)

# Set up subplots to display the original image, mask, and eroded image
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 40))
cmap_val = 'gray'    # Set colormap to grayscale for all images

# Turn off axis and set title for the original image
ax1.axis('off')
ax1.title.set_text('Original_img')

# Turn off axis and set title for the mask
ax2.axis('off')
ax2.title.set_text('mask')

# Turn off axis and set title for the eroded image
ax3.axis('off')
ax3.title.set_text('Eroded_img')

# Display the original image, mask, and eroded image in the subplots
ax1.imshow(Original_img, cmap=cmap_val)
ax2.imshow(mask, cmap=cmap_val)
ax3.imshow(Eroded_img, cmap=cmap_val)

# Show the plots
plt.show()
#%% Dilation

# Define a 3x3 kernel for dilation
kernel2 = np.ones((3,3), np.uint8)

# Perform dilation operation on the binary mask with 1 iteration
Dilated_img = cv.dilate(mask, kernel2, iterations=1)

# Set up subplots to display the original image, mask, and dilated image
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 40))
cmap_val = 'gray'    # Set colormap to grayscale for all images

# Turn off axis and set title for the original image
ax1.axis('off')
ax1.title.set_text('Original_img')

# Turn off axis and set title for the mask
ax2.axis('off')
ax2.title.set_text('mask')

# Turn off axis and set title for the dilated image
ax3.axis('off')
ax3.title.set_text('Dilated_img')

# Display the original image, mask, and dilated image in the subplots
ax1.imshow(Original_img, cmap=cmap_val)
ax2.imshow(mask, cmap=cmap_val)
ax3.imshow(Dilated_img, cmap=cmap_val)

# Show the plots
plt.show()
#%% Closing (dilation followed by erosion)

# Load the original cell image in grayscale
Original_image = cv.imread('Cells.jpg', 0)

# Apply a binary threshold to convert the image to a binary mask
_, Mask = cv.threshold(Original_image, 20, 255, cv.THRESH_BINARY)

# Define a 9x9 kernel for closing 
kernel3 = np.ones((9,9), np.uint8)

# Perform closing operation on the binary mask
Closed_img = cv.morphologyEx(Mask, cv.MORPH_CLOSE, kernel3)

# Set up subplots to display the original image, mask, and closed image
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 40))
cmap_val = 'gray'    # Set colormap to grayscale for all images

# Turn off axis and set title for the original image
ax1.axis('off')
ax1.title.set_text('Original_image')

# Turn off axis and set title for the mask
ax2.axis('off')
ax2.title.set_text('Mask')

# Turn off axis and set title for the closed image
ax3.axis('off')
ax3.title.set_text('Closed_img')

# Display the original image, mask, and closed image in the subplots
ax1.imshow(Original_image, cmap=cmap_val)
ax2.imshow(Mask, cmap=cmap_val)
ax3.imshow(Closed_img, cmap=cmap_val)

# Show the plots
plt.show()
#%% Opening (erosion followed by dilation)

# Define a 7x7 kernel for opening 
kernel4 = np.ones((7,7), np.uint8)

# Perform opening operation on the binary mask
Opened_img = cv.morphologyEx(Mask, cv.MORPH_OPEN, kernel4)

# Set up subplots to display the original image, mask, and opened image
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 40))
cmap_val = 'gray'    # Set colormap to grayscale for all images

# Turn off axis and set title for the original image
ax1.axis('off')
ax1.title.set_text('Original_image')

# Turn off axis and set title for the mask
ax2.axis('off')
ax2.title.set_text('Mask')

# Turn off axis and set title for the opened image
ax3.axis('off')
ax3.title.set_text('Opened_img') 

# Display the original image, mask, and opened image in the subplots
ax1.imshow(Original_image, cmap=cmap_val)
ax2.imshow(Mask, cmap=cmap_val)
ax3.imshow(Opened_img, cmap=cmap_val)

# Show the plots
plt.show()
#%% Gradient

# Define a 5x5 kernel for the morphological operations 
kernel5 = np.ones((5,5), np.uint8)

# Apply the morphological gradient operation using the 5x5 kernel
# The gradient is the difference between dilation and erosion, highlighting the edges of objects
Gradient = cv.morphologyEx(Mask, cv.MORPH_GRADIENT, kernel5)

# Perform dilation on the binary mask using the 5x5 kernel with 1 iteration
Dilated_img = cv.dilate(Mask, kernel5, iterations=1)

# Perform erosion on the binary mask using the 5x5 kernel with 1 iteration
Eroded_img = cv.erode(Mask, kernel5, iterations=1)

# Subtract the eroded image from the dilated image to highlight the edge-like structures
Subtract_img = Dilated_img - Eroded_img

# Set up subplots to display the original image, mask, gradient, and the result of the subtraction
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(30, 40))
cmap_val = 'gray'         # Set colormap to grayscale for all images

# Turn off axis and set title for the original image
ax1.axis('off')
ax1.title.set_text('Original_image')

# Turn off axis and set title for the mask 
ax2.axis('off')
ax2.title.set_text('Mask')

# Turn off axis and set title for the gradient image (edge detection result)
ax3.axis('off')
ax3.title.set_text('Gradient')

# Turn off axis and set title for the image obtained by subtracting erosion from dilation
ax4.axis('off')
ax4.title.set_text('Subtract_img')

# Display the original image, mask, gradient, and subtraction result in the subplots
ax1.imshow(Original_image, cmap=cmap_val)
ax2.imshow(Mask, cmap=cmap_val)
ax3.imshow(Gradient, cmap=cmap_val)
ax4.imshow(Subtract_img, cmap=cmap_val)

# Show the plots
plt.show()

