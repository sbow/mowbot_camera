# snap_image.py

import numpy as np
import cv2


cap = cv2.VideoCapture(0)

ret, frame2 = cap.read()

# MANIPULATION
#
#

# Convert to Grayscale
# gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

success = cv2.imwrite('snap.jpg',frame2)
