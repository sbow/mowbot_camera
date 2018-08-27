# birds_eye.py
# Shaun Bowman
# August 27 2018
# Proof of concept birds eye view calculation
import cv2
import numpy as np
import matplotlib.pyplot as plt


x  = 0
y  = 1
br_crop= [348, 981]
bl_crop = [1551,981]
tl_crop = [1163,647]
tr_crop = [747, 647]
br = [0   , br_crop[y] - tr_crop[y] - 1]
bl = [1203 - 1, br_crop[y] - tr_crop[y] - 1]
tl = [400 ,  0]
tr = [820 ,  0]
widthA = np.sqrt(((br_crop[x] - bl_crop[x]) ** 2) + ((br_crop[y] - bl_crop[y]) ** 2))
widthB = np.sqrt(((tr_crop[x] - tl_crop[x]) ** 2) + ((tr_crop[y] - tl_crop[y]) ** 2))
maxWidth = max(int(widthA), int(widthB))
heightA = np.sqrt(((tr_crop[x] - br_crop[x]) ** 2) + ((tr_crop[y] - br_crop[y]) ** 2))
heightB = np.sqrt(((tl_crop[x] - bl_crop[x]) ** 2) + ((tl_crop[y] - bl_crop[y]) ** 2))
maxHeight = max(int(heightA), int(heightB))

src = np.array([[tl_crop[x], tl_crop[y]], [tr_crop[x], tr_crop[y]], [br_crop[x], br_crop[y]], [bl_crop[x], bl_crop[y]]], dtype = "float32" )
dst = np.array([
	[0, 0],
	[maxWidth -1, 0],
	[maxWidth -1, maxHeight - 1],
	[0, maxHeight -1]], dtype = "float32")

img = cv2.imread("1ft_grid_12inch setback_1920x1080_1.png")
#img = img[tr_crop[y]:br_crop[y],br_crop[x]:bl_crop[x] ]

M = cv2.getPerspectiveTransform( src, dst )
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

#plt.imshow(img)
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.show()

