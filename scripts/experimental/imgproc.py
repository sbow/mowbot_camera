#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:15:31 2019

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
image = plt.imread(files[0])
image.shape
plt.imshow(image)

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
    
    
    codebook_random = shuffle(image_array, random_state=0)[:n_colors]
    print("Predicting color indices on the full image (random)")
    t0 = time()
    labels_random = pairwise_distances_argmin(codebook_random,
                                              image_array,
                                              axis=0)
    print("done in %0.3fs." % (time() - t0))
    
    return kmeans, labels, labels_random, w, h


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

n_colors = 5

# Load the Summer Palace photo
n_img = 4500
for i in range(len(files)):
    n_img = i
    sample_img = plt.imread(files[n_img])
    kmeans, labels, labels_random, w, h = do_kmeans(sample_img)
    kmeans_img = recreate_image(kmeans.cluster_centers_, labels, w, h)*255
    f_name = "kmeans_N"+str(n_colors)+"_"+files[n_img]
    print(f_name)
    cv2.imwrite(f_name, kmeans_img)
    plt.imshow(kmeans_img)

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
#plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('Quantized image ('+str(n_colors)+' colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()