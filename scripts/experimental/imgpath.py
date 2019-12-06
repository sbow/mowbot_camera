#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:33:37 2019

@author: shaun
"""

import os
os.chdir("/Users/shaun/mowbot/track_slantcsi")
files = os.listdir(".")
files.sort()

from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time


def do_kmeans(sample_img):
    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    sample_img = np.array(sample_img, dtype=np.float64) / 255
    
    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(sample_img.shape)
    assert d == 3
    image_array = np.reshape(sample_img, (w * h, d))
    
    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    print("done in %0.3fs." % (time() - t0))
    
    # Get labels for all points
    print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    print("done in %0.3fs." % (time() - t0))
    
    return kmeans, labels, w, h


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def find_adjacent(kmeans_img, start_row, start_col, width, height):
    """Start start_row, start_col, search LR then up for matching values,  
    repeat. Return end col, return list of matching row,col ie x,y. Simple
    version of the "fill" command in Photoshop. These matching row,col will
    be used later to curve-fit a polynomial to each kmeans track to determine
    blue line / lane edge / open road ect by comparing strength of curve fit"""
    end_col = 0
    match_r = []
    match_c = []
    width_r = [] # width of matching zone at current row
    target = kmeans_img[start_row, start_col, 0]
    i_r = start_row
    i_c = start_col
    result = True
    finished = False
    while finished == False:
        row_match_c = [] # matching coluns on current row
        i_c_start_right = i_c # save initial i_c to save time on search left
        width = 0 # width on current row, initialize to zero
        while result:
            # build list of matching columns on current row
            test = kmeans_img[i_r, i_c, 0]
            if test == target:
                # search right & origin
                match_r.append(i_r)
                match_c.append(i_c)
                row_match_c.append(i_c)
                width = width+1
                if i_c < h-1:
                    i_c = i_c + 1 # one right
                    if i_r == start_row:
                        end_col = i_c # return right most col of first row
                else:
                    # as far right as can go, exit
                    i_c = i_c_start_right #search left at origin of right srch
                    break
            else:
                # found first non matching pixel right, exit
                i_c = i_c_start_right #search left at origin of right search
                break
        
        while result:
            # build list of matching columns on current row, search left            
            i_c = i_c - 1 # one left of origin at current row
            if i_c < 0:
                # as far left as can go, exit
                i_c = 0
                break
            test = kmeans_img[i_r, i_c, 0]
            if test == target:
                match_r.append(i_r)
                match_c.append(i_c)
                row_match_c.append(i_c)
                width = width+1
            else:
                # found first non matching pixel left, exit
                break
        
        width_r.append(width) # save width for current row        
        c_vtest_match = []
        i_r = i_r - 1 # up by 1
        if i_r < 0:
            finished = True # at first i_r, no further row to search
            break
        for c_vtest in row_match_c: # does not run if len(row_match_c)==0
            # build list of columns with matching elements vertically adjacent
            test = kmeans_img[i_r, c_vtest, 0]
            if test == target:
                c_vtest_match.append(c_vtest)
        if len(c_vtest_match) == 0:
            # didn't find any matching adjacent elements vertically, end
            finished = True
        else:
            # there was at least one matching element adjacent vertically
            # from a matching r,c. Find the mean column of all matching
            # elements from i_r & use this as the initial search value of
            # i_r-1,c. Note this algorithm does not handle "forks" in the road.
            mean_c = int(sum(c_vtest_match)/len(c_vtest_match))
            #i_r = i_r - 1 # protected against i_r < 0 above
            i_c = mean_c #avoid edges by starting v search at center of old row
    # finished search, return results
    return end_col, match_r, match_c, width_r
        

n_colors = 5

n_img = 300
# first real image: 474
# no blue line, only lane endge: 1690..1764
# noise - another vehicle: 1942
# discontinuity: 2002
# discontinuity, no blue line: 4002
# blank image: 778

sample_img = plt.imread(files[n_img])
sample_img = cv2.imread(files[n_img])
scale_percent = 20 # percent of original size
width = int(sample_img.shape[1] * scale_percent / 100)
height = int(sample_img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
sample_img = cv2.resize(sample_img, dim, interpolation = cv2.INTER_AREA)

kmeans, labels, w, h = do_kmeans(sample_img)
kmeans_img = recreate_image(kmeans.cluster_centers_, labels, w, h)*255

#end_col, match_r, match_c = find_adjacent(kmeans_img=kmeans_img, \
#                    start_row=height-1, start_col=0, width=width, height=height)     

#end_col, match_r, match_c = find_adjacent(kmeans_img=kmeans_img, \
#                   start_row=height-1, start_col=37, width=width, height=height) 

#end_col, match_r, match_c, width_r = find_adjacent(kmeans_img=kmeans_img, \
#                    start_row=height-1, start_col=37, width=width, height=height) 
    
#end_col, match_r, match_c = find_adjacent(kmeans_img=kmeans_img, \
#                    start_row=height-1, start_col=90, width=width, height=height) 

#end_col, match_r, match_c = find_adjacent(kmeans_img=kmeans_img, \
#                    start_row=height-1, start_col=100, width=width, height=height) 
    
    
# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(sample_img)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image ('+str(n_colors)+' colors, K-Means)')
plt.imshow(kmeans_img)

searched = False
lane_c = []
test_c = 0
width_rows = []
lane_r = [] 
lane_c = []
n_lanes = 0
while searched == False:
    end_col, match_r, match_c, width_r = find_adjacent(kmeans_img=kmeans_img, \
                    start_row=height-1, start_col=test_c, width=width, height=height) 
    width_rows.append(width_r)
    if end_col == width - 1:
        searched = True
    fit_good = False
    fit_laneline = False
    fit_unknown = False # shadow, bright spot
    fit_lane = False
    fit_free = False
    fit = np.polyfit(match_c, match_r, 2, full=True) #fit[0]=[c2,c1,c0], fit[1]=residuals, 
    if len(match_c) > 100:
        fit_good = True
        fit_var = fit[1]/(len(match_c) - 2) #variance of the fit, used to detect narrow line
        if np.mean(width_r) < 7:
            fit_laneline = True
            lane_r.append(match_r)
            lane_c.append(match_c)
            n_lanes += 1
        elif fit_var < 150:
            fit_unknown = True
        elif fit_var < 200:
            fit_unknown = True
        else:
            fit_unknown = True
    else:
        fit_good = False
#    if fit_laneline:
#        lane_c.append(test_c)
    test_c = end_col
        
print("len: "+str(len(match_c))+"\nvar: "+str(fit_var)+"\nfit_good: "+\
      str(fit_good)+"\nfit_laneline: "+str(fit_laneline)+"\nfit_lane: "+\
      str(fit_lane)+"\nfit_tree: "+str(fit_free)+"\nfit_unknown: "+\
      str(fit_unknown))        

#end_col, match_r, match_c, width_r = find_adjacent(kmeans_img=kmeans_img, \
#                    start_row=height-1, start_col=37, width=width, height=height) 


segment_img = kmeans_img
for i in range(n_lanes):
    for j in range(len(lane_r[i])):
        segment_img[lane_r[i][j],lane_c[i][j],:] = [0,0,0]

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('Segment image ('+str(n_colors)+' colors, K-Means)')
plt.imshow(segment_img)
