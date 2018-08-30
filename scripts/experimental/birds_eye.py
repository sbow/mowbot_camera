# birds_eye.py
# Shaun Bowman
# August 27 2018
# Proof of concept birds eye view calculation
import cv2
import numpy as np
import matplotlib.pyplot as plt


x  = 0
y  = 1

w_source = 1920.
h_source = 1080.
wh_source = np.array([w_source, h_source])

bl_crop= [348, 981]
br_crop = [1551,981]
tr_crop = [1163,647]
tl_crop = [747, 647]
bl_ratio = np.array([0.18125, 0.90833])
br_ratio = np.array([0.80781, 0.90833])
tr_ratio = np.array([0.60573, 0.59907])
tl_ratio = np.array([0.38906, 0.59907])
bl_pix = (bl_ratio*wh_source).astype(int)
br_pix = (br_ratio*wh_source).astype(int)
tr_pix = (tr_ratio*wh_source).astype(int)
tl_pix = (tl_ratio*wh_source).astype(int)

y_crop_dsrd = 500 # for birdseye @ 1920 x 1080; crop first 500 rows
y_crop_dsrd_ratio = 0.46296 # ratio of 500 / 1080.; tobe subtracted from
bl_crop_ratio = np.array([0.18125, 0.44537])
br_crop_ratio = np.array([0.80781, 0.44537])
tr_crop_ratio = np.array([0.60573, 0.13611])
tl_crop_ratio = np.array([0.38906, 0.13611])
bl_crop_pix = (bl_crop_ratio*wh_source).astype(int)
br_crop_pix = (br_crop_ratio*wh_source).astype(int)
tr_crop_pix = (tr_crop_ratio*wh_source).astype(int)
tl_crop_pix = (tl_crop_ratio*wh_source).astype(int)

br = [0   , (br_crop[y] - tr_crop[y] - 1)]
bl = [1203 - 1, (br_crop[y] - tr_crop[y] - 1)]
tl = [400 ,  0]
tr = [820 ,  0]
br_dst_ratio = [0.00000, 0.30833]
bl_dst_ratio = [0.62604, 0.30833]
tl_dst_ratio = [0.20833, 0.00000]
tr_dst_ratio = [0.42708, 0.00000]
br_dst_pix = (br_dst_ratio*wh_source).astype(int)
bl_dst_pix = (bl_dst_ratio*wh_source).astype(int)
tl_dst_pix = (tl_dst_ratio*wh_source).astype(int)
tr_dst_pix = (tr_dst_ratio*wh_source).astype(int)

offset_x = 1750
offset_yt = 1000
offset_yb = 41

widthA = np.sqrt(((br_crop[x] - bl_crop[x]) ** 2) + ((br_crop[y] - bl_crop[y]) ** 2))
widthB = np.sqrt(((tr_crop[x] - tl_crop[x]) ** 2) + ((tr_crop[y] - tl_crop[y]) ** 2))
maxWidth = max(int(widthA + 2*offset_x), int(widthB + 2*offset_x))

heightA = np.sqrt(((tr_crop[x] - br_crop[x]) ** 2) + ((tr_crop[y] - br_crop[y]) ** 2))
heightB = np.sqrt(((tl_crop[x] - bl_crop[x]) ** 2) + ((tl_crop[y] - bl_crop[y]) ** 2))
maxHeight = max(int(heightA + offset_yt + offset_yb), int(heightB + offset_yt + offset_yb))

src = np.array([[tl_crop[x], tl_crop[y]], [tr_crop[x], tr_crop[y]],
				[br_crop[x], br_crop[y]], [bl_crop[x], bl_crop[y]]],
			   dtype = "float32" )

dst = np.array([
	[offset_x, offset_yt],
	[maxWidth -1 - offset_x, offset_yt],
	[maxWidth -1 - offset_x, maxHeight - 1 - offset_yb],
	[offset_x, maxHeight -1 - offset_yb]], dtype = "float32")

img = cv2.imread("1ft_grid_12inch setback_1920x1080_1.png")

M = cv2.getPerspectiveTransform( src, dst )
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

fig, axs = plt.subplots(1,1)
axs.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
axs.axis('tight')
plt.show()

