import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("1ft_grid_12inch setback_1920x1080_1.png")
n_r, n_c, n_ch = img.shape

v_xy = np.array([
[0, 1],\
[1, 1],\
[2, 1],\
[3, 1],\
[4, 1],\
[-1, 1],\
[-2, 1],\
[-3, 1],\
[-4, 1],\
[0, 2],\
[1, 2],\
[2, 2],\
[3, 2],\
[4, 2],\
[-1, 2],\
[-2, 2],\
[-3, 2],\
[0, 3],\
[1, 3],\
[2, 3],\
[3, 3],\
[4, 3],\
[-1, 3],\
[-2, 3],\
[-3, 3],\
[-4, 3],\
[-5, 3],\
[0, 4],\
[1, 4],\
[2, 4],\
[3, 4],\
[4, 4],\
[-1, 4],\
[-2, 4],\
[-3, 4],\
[-4, 4],\
[-5, 4],\
[0, 5],\
[1, 5],\
[2, 5],\
[3, 5],\
[4, 5],\
[-1, 5],\
[-2, 5],\
[-3, 5],\
[-4, 5],\
[-5, 5],\
])
#i_1 = 46 #0
#i_2 = 41 #12
#i_3 = 4 #25
#i_4 = 0 #43

n_x = 5.
n_y = 5.
v_rc_norm = np.copy(v_xy) / (n_x + n_y) + .5
v_rc_norm[:,1] = v_rc_norm[:,1]*-1 + 1

u_rc = np.array([\
[1061, 954],\
[1033, 1293],\
[980, 1552],\
[915, 1753],\
[862, 1857],\
[1032, 615],\
[977, 356],\
[917, 176],\
[864, 53],\
[819, 961],\
[797, 1344],\
[761, 1618],\
[720, 1789],\
[693, 1903],\
[798, 572],\
[760, 292],\
[723, 119],\
[960, 709],\
[700, 1233],\
[687, 1463],\
[670, 1633],\
[655, 1753],\
[702, 686],\
[690, 456],\
[673, 287],\
[675, 160],\
[642, 70],\
[652, 958],\
[648, 1170],\
[644, 1366],\
[634, 1514],\
[626, 1640],\
[650, 747],\
[643, 556],\
[637, 400],\
[630, 276],\
[621, 179],\
[618, 960],\
[617, 1132],\
[615, 1291],\
[610, 1431],\
[605, 1545],\
[618, 790],\
[615, 629],\
[610, 490],\
[607, 371],\
[601, 279],\
])

u_rc_norm = np.copy(u_rc)
u_rc_norm = np.float32(u_rc_norm)
u_rc_norm[:,0] = u_rc_norm[:,0] / float( n_r )
u_rc_norm[:,1] = u_rc_norm[:,1] / float( n_c )

i_1 = 46 #0
i_2 = 41 #12
i_3 = 4 #25
i_4 = 0 #43
u_1 = u_rc[i_1,:]
u_2 = u_rc[i_2,:]
u_3 = u_rc[i_3,:]
u_4 = u_rc[i_4,:]

v_1 = v_rc_norm[i_1,:]
v_2 = v_rc_norm[i_2,:]
v_3 = v_rc_norm[i_3,:]
v_4 = v_rc_norm[i_4,:]

r_crop = np.min([u_1[0], u_2[0], u_3[0], u_4[0]])
c_crop = np.min([u_1[1], u_2[1], u_3[1], u_4[1]])
u_1_crop = u_1 - [r_crop, c_crop]
u_2_crop = u_2 - [r_crop, c_crop]
u_3_crop = u_3 - [r_crop, c_crop]
u_4_crop = u_4 - [r_crop, c_crop]
img_crop = img[r_crop:, c_crop:]

pts1 = np.float32([u_1_crop, u_2_crop, u_3_crop, u_4_crop])
pts2 = np.float32([v_1, v_2, v_3, v_4])*300 + 1

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img_crop, M, (300,300))

plt.imshow(dst)
plt.show()

orig_r = 472
orig_c = 961

# ray: lines of constant lateral offset from far to near
#ray_1_i = np.array([38, 29, 20, 13, 5]) # line of 1,1 offset from 0,0 to 4,5
ray_1_i = 38,28, 18, 10, 1
ray_1_rc_norm = np.array([[u_rc_norm[38,:]], [u_rc_norm[28,:]], [u_rc_norm[18,:]],\
[u_rc_norm[10,:]],[u_rc_norm[1,:]]])
ray_1_rc_u = np.array([[u_rc[38,:]], [u_rc[28,:]], [u_rc[18,:]],\
[u_rc[10,:]],[u_rc[1,:]]])
ray_1_rc = ray_1_rc_u - [orig_r, orig_c]

ray_1_mag_u_rc = np.zeros(len(ray_1_rc_norm))
for i in range( len(ray_1_mag_u_rc) ):
	ray_1_mag_u_rc[i] = np.sqrt( ray_1_rc_norm[i, 0]**2 + ray_1_rc_norm[i,1]**2)

	
