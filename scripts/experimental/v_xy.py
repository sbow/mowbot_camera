import numpy as np
import cv2
import matplotlib.pyplot as plt
execfile("points.py")

img = cv2.imread("1ft_grid_12inch setback_1920x1080_1.png")
n_r, n_c, n_ch = img.shape


pts1 = np.float32([\
	[u_r[v_x == -4][-1], u_c[v_x == -4][-1]],
	[u_r[v_x == -2][0], u_c[v_x == -2][0]],
	[u_r[v_x == 4][-1], u_c[v_x == 4][-1]], 
	[u_r[v_x == 2][0], u_c[v_x == 2][0]], 
	])
x_4 = (4 + 5) / 10.
x_m4 = (-4 + 5) / 10.
 
v_x = (v_x + 5) / 10.
v_y = (-1*v_y + 5) / 5.

pts2 = np.float32([\
	[v_y[v_x == x_m4][-1]*600,  v_x[v_x == x_m4][-1]*300],
	[v_y[v_x == .3][0]*600, v_x[v_x == .3][0]*300],
	[v_y[v_x == x_4][-1]*600,  v_x[v_x == x_4][-1]*300], 
	[v_y[v_x == .7][0]*600,   v_x[v_x == .7][0]*300], 
	])


M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (600,300))

plt.imshow(dst)
plt.show()

orig_r = 472
orig_c = 961


	
