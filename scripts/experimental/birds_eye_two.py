# birds_eye_two.py
# Shaun Bowman
# August 29 2018
#
# Proof of concept birds eye view calculation
#
# Uses target grid (1ft x 1ft) layed out on ground plane
# See image 1ft_grid_12_inch setback_1920x1080_1.png
#
# Notes:
#
#       The objective is to find transform M using a calibration (target) image. This
#       transform only needs to be found once. For a video stream, it would simply be
#       applied to each frame, not recalculated each time.
#
#       The code below allows the target image to be of a different resolution (ie: higher)
#       than the eventual video stream; the output transform (M) being correct for the
#       video stream resolution.
#
#       The first step is to determine the source (src) and destination (dst) pixel
#       locations of targets in the calibration image. These are stored interms of the ratio
#       of pixel location to w/h of the calibration image (XX_ratio, XX_crop_ratio, XX_dst_ratio
#
#       The output image does not have equal scale in x & y directions. This is OK because by
#       the birds-eye perspective of the ground plane requires a lot of interpolation in the Y
#       diretion, so the reduced Y scale relative to X is not reducing "information"

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define array ordering, x==horizontal, y==vertical
x  = 0
y  = 1

# Define properties of source image
img_file = '1ft_grid_12inch setback_1920x1080_1.png'
w_vid_strm = 640.
h_vid_strm = 480.
wh_vid_strm = np.array([w_vid_strm, h_vid_strm])

# Target locations in source image - expressed in ratio of px location to width / height
bl_ratio = np.array([0.18125, 0.90833])
br_ratio = np.array([0.80781, 0.90833])
tr_ratio = np.array([0.60573, 0.59907])
tl_ratio = np.array([0.38906, 0.59907])
bl_pix = (bl_ratio*wh_vid_strm).astype(int)
br_pix = (br_ratio*wh_vid_strm).astype(int)
tr_pix = (tr_ratio*wh_vid_strm).astype(int)
tl_pix = (tl_ratio*wh_vid_strm).astype(int)

# Define crop first portion of source image, redefine target locations in terms of cropped image
y_crop_dsrd_ratio = 0.46296 # ratio of 500 / 1080.; tobe subtracted XX_ratio[y] variables
y_crop_dsrd_pix = int( y_crop_dsrd_ratio*h_vid_strm)
bl_crop_ratio = np.array([0.18125, 0.44537])
br_crop_ratio = np.array([0.80781, 0.44537])
tr_crop_ratio = np.array([0.60573, 0.13611])
tl_crop_ratio = np.array([0.38906, 0.13611])
bl_crop_pix = (bl_crop_ratio*wh_vid_strm).astype(int)
br_crop_pix = (br_crop_ratio*wh_vid_strm).astype(int)
tr_crop_pix = (tr_crop_ratio*wh_vid_strm).astype(int)
tl_crop_pix = (tl_crop_ratio*wh_vid_strm).astype(int)

# Define target locations after birds eye projection
br_dst_ratio = [0.00000, 0.30833]
bl_dst_ratio = [0.62604, 0.30833]
tl_dst_ratio = [0.20833, 0.00000]
tr_dst_ratio = [0.42708, 0.00000]
br_dst_pix = (br_dst_ratio*wh_vid_strm).astype(int)
bl_dst_pix = (bl_dst_ratio*wh_vid_strm).astype(int)
tl_dst_pix = (tl_dst_ratio*wh_vid_strm).astype(int)
tr_dst_pix = (tr_dst_ratio*wh_vid_strm).astype(int)

# Define extra area in dst image outside polygon formed by target points.
# This space in the image will be the source image rectified like the area in the target points.
offset_x = int(1750 / 1920.0 * w_vid_strm)
offset_yt = int(1000 / 1080.0 * (h_vid_strm - y_crop_dsrd_pix))
offset_yb = int(41 / 1080.0 * h_vid_strm)

# Figure out the width & height of the dst image. Uses distance between corners
widthA = np.sqrt(((br_crop_pix[x] - bl_crop_pix[x]) ** 2) + ((br_crop_pix[y] - bl_crop_pix[y]) ** 2))
widthB = np.sqrt(((tr_crop_pix[x] - tl_crop_pix[x]) ** 2) + ((tr_crop_pix[y] - tl_crop_pix[y]) ** 2))
heightA = np.sqrt(((tr_crop_pix[x] - br_crop_pix[x]) ** 2) + ((tr_crop_pix[y] - br_crop_pix[y]) ** 2))
heightB = np.sqrt(((tl_crop_pix[x] - bl_crop_pix[x]) ** 2) + ((tl_crop_pix[y] - bl_crop_pix[y]) ** 2))
maxWidth = max(int(widthA + 2*offset_x), int(widthB + 2*offset_x))
maxHeight = max(int(heightA + offset_yt + offset_yb), int(heightB + offset_yt + offset_yb))

# Define properties of source (src) image
src = np.array([[tl_crop_pix[x], tl_crop_pix[y]], [tr_crop_pix[x], tr_crop_pix[y]],
                [br_crop_pix[x], br_crop_pix[y]], [bl_crop_pix[x], bl_crop_pix[y]]],
               dtype = "float32" )

# Define properties of destination (dst) image
dst = np.array([
    [offset_x,                  offset_yt],
    [maxWidth -1 - offset_x,    offset_yt],
    [maxWidth -1 - offset_x,    maxHeight - 1 - offset_yb],
    [offset_x,                  maxHeight -1 - offset_yb]], dtype = "float32")

# Get source image with targets and scale to desired w & h
img = cv2.imread(img_file)
img_scaled = cv2.resize(img, (int(w_vid_strm), int(h_vid_strm)), interpolation=cv2.INTER_AREA)
img_crop = img_scaled[y_crop_dsrd_pix:, :]

# Figure out transformation between source (target) and destination (birds eye) image
M = cv2.getPerspectiveTransform( src, dst )
img_warped = cv2.warpPerspective(img_crop, M, (maxWidth, maxHeight))

# Plot results
fig, axs = plt.subplots(1,1)
axs.imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
axs.axis('tight')
plt.show()
