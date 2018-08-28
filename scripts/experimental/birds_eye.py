# birds_eye.py
# Shaun Bowman
# August 27 2018
# Proof of concept birds eye view calculation
import cv2
import numpy as np
import matplotlib.pyplot as plt


x  = 0
y  = 1

bl_crop= [348, 981]
br_crop = [1551,981]
tr_crop = [1163,647]
tl_crop = [747, 647]

y_scale = 1
br = [0   , (br_crop[y] - tr_crop[y] - 1)*y_scale]
bl = [1203 - 1, (br_crop[y] - tr_crop[y] - 1)*y_scale]
tl = [400 ,  0]
tr = [820 ,  0]

widthA = np.sqrt(((br_crop[x] - bl_crop[x]) ** 2) + ((br_crop[y] - bl_crop[y]) ** 2))
widthB = np.sqrt(((tr_crop[x] - tl_crop[x]) ** 2) + ((tr_crop[y] - tl_crop[y]) ** 2))
maxWidth = max(int(widthA + 2*offset_x), int(widthB + 2*offset_x))
heightA = np.sqrt(((tr_crop[x] - br_crop[x]) ** 2) + ((tr_crop[y] - br_crop[y]) ** 2))
heightB = np.sqrt(((tl_crop[x] - bl_crop[x]) ** 2) + ((tl_crop[y] - bl_crop[y]) ** 2))

offset_x = 1850
offset_yt = 1000
offset_yb = 50
maxHeight = max(int(heightA + offset_yt + offset_yb), int(heightB + offset_yt + offset_yb))*y_scale

src = np.array([[tl_crop[x], tl_crop[y]], [tr_crop[x], tr_crop[y]], [br_crop[x], br_crop[y]], [bl_crop[x], bl_crop[y]]], dtype = "float32" )
dst = np.array([
	[offset_x, offset_yt],
	[maxWidth -1 - offset_x, offset_yt],
	[maxWidth -1 - offset_x, maxHeight - 1 - offset_yb],
	[offset_x, maxHeight -1 - offset_yb]], dtype = "float32")

img = cv2.imread("1ft_grid_12inch setback_1920x1080_1.png")
#img = img[tr_crop[y]:br_crop[y],br_crop[x]:bl_crop[x] ]

M = cv2.getPerspectiveTransform( src, dst )
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

#plt.imshow(img)
fig, axs = plt.subplots(1,1)
axs.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
axs.axis('tight')
#axs.axis('equal')
plt.show()

