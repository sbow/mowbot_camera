# adapted from Matlab fileexchange file adaptcluster_kmeans.m by Ankit Dixit

IS_DEMO = True
VERBOSE = False
img = []

import cv2
import numpy as np
import math

if IS_DEMO:
    VERBOSE = True
    img = cv2.imread('snap_1.jpg')

[h_src, w_src, n_src_clrs] = img.shape

if n_src_clrs != 3:
    print('Img needs 3 color chanels')
    quit()

[red, green, blue] = [img[:,:,i] for i in range(n_src_clrs)]

array = np.array([np.reshape(red, -1), np.reshape(green, -1), np.reshape(blue, -1) ] ) # 3x307200
[n_src_clrs, n_pixels] = array.shape

i = 0
j = 0

test_n = 1
while test_n == 1:

    seed = np.zeros([3, 1]) # 3x1 array
    seed[0] = array[0, :].mean()
    seed[1] = array[1, :].mean()
    seed[2] = array[2, :].mean()

    i = i + 1

    while test_n == 1:

        j = j + 1

        seedvec = np.tile( seed, (1, n_pixels) ) # 3 x n_pixels(307200)
        diff = array - seedvec
        abs_diff = np.power( np.power( diff, 2), 0.5) # 3 x n_pixes(307200)
        dist = np.sum( abs_diff, 0) # sub differance over all colors for each pixel, 1xn_pixels

        distth = 0.25 * dist.max() # thershold for like colors 0.25 * max diff
        qualified = np.where( dist < distth )

        newred = array[0,:]
        newgreen = array[1,:]
        newblue = array[2, :]

        newseed = np.zeros([3, 1])
        newseed[0] = newred[ qualified ].mean()
        newseed[1] = newgreen[ qualified ].mean()
        newseed[2] = newblue[ qualified ].mean()

        if np.isnan(newseed).any(): # dont think this works...
            break

        if (seed == newseed).any() or j > 10:
            j = 0
            
            break

        print(dist)
        print(dist.shape)
        print(newseed)
        test_n = 0

