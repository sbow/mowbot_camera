#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Dec 7 16:24:25 2019

@author: shaun
"""



import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time


def do_kmeans(sample_img, n_colors):
    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    sample_img = np.array(sample_img, dtype=np.float64) / 255
    
    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(sample_img.shape)
    assert d == 3
    image_array = np.reshape(sample_img, (w * h, d))
    
    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:150]
    print("shuffle done in %0.3fs"%(time()-t0))
    kmeans = KMeans(n_clusters=n_colors,  n_init=10, max_iter=1, tol=.01, random_state=0).fit(image_array_sample)
    print("Kmeans + shuffle done in %0.3fs." % (time() - t0))
    
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
    t0 = time()
    end_col = 0
    match_r = []
    match_c = []
    width_r = [] # width of matching zone at current row
    row_match_c_last = [] # last full row match c
    target = kmeans_img[start_row, start_col, 0]
    h = width 
    i_r = start_row
    i_c = start_col
    result = True
    finished = False
    while finished == False:
        TEST_R = 3 # number of rows to test to get past discontinuit        
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
                row_match_c_last = row_match_c
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
                row_match_c_last = row_match_c
                width = width+1
            else:
                # found first non matching pixel left, exit
                break
        
        width_r.append(width) # save width for current row        
        c_vtest_match = []
        i_r = i_r - 1 # up by 1
        #TEST_R = TEST_R - 1 # update max search rows
        if i_r < 0:
            finished = True # at first i_r, no further row to search
            break
        for c_vtest in row_match_c: # does not run if len(row_match_c)==0
            # build list of columns with matching elements vertically adjacent
            test = kmeans_img[i_r, c_vtest, 0]
            if test == target:
                c_vtest_match.append(c_vtest)
        if len(c_vtest_match) == 0:
            # didn't find any matching adjacent elements vertically, continue
            # searching up incase of small discontinuity, then exit
            while TEST_R > 0:
                i_r = i_r - 1 # up by 1
                #TEST_R = TEST_R - 1 # update max search rows
                if i_r < 0:
                    finished = True # at first i_r, no further row to search
                    break
                for c_vtest in row_match_c_last: # does not run if len(row_match_c)==0
                    # build list of columns with matching elements vertically adjacent
                    test = kmeans_img[i_r, c_vtest, 0]
                    if test == target:
                        c_vtest_match.append(c_vtest)
                        TEST_R = 0
                        finished = False
                        i_c = c_vtest
                        break
                TEST_R = TEST_R - 1
                if len(c_vtest_match) == 0:
                    # didn't find matching row, stop searching
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
        
def get_bl(raw_img, cntrl_row=35, n_colors=5, scale_w=120, scale_h=96, lane_len_min=5,
           lane_max_width=10, lane_min_spacing=5, n_bl_points=20,
           lane_max_widthstd=99, lane_min_widthmean=0):
    sample_img = raw_img 
    width = scale_w
    height = scale_h
    dim = (width, height)
    # resize image:
    sample_img = cv2.resize(sample_img, dim, interpolation = cv2.INTER_AREA)#scale
    t0 = time() 
    kmeans, labels, w, h = do_kmeans(sample_img, n_colors) # do kmeans clustering
    print("kmeans done in %0.3fs." % (time() - t0))
    kmeans_img = recreate_image(kmeans.cluster_centers_, labels, w, h)*255 #scale
    
    searched = False
    lane_c = []
    test_c = 0
    width_rows = []
    lane_r = [] 
    lane_c = []
    n_lanes = 0
    col_space = [] # column spacing
    start_col_prev = []
    
    # determine lane markers from kmeans_img
    fit_good = False
    fit_laneline = False
    fit_unknown = False # shadow, bright spot
    fit_lane = False
    fit_free = False
    MIN_LEN_R = lane_len_min # min row length to be consitered lane marker
    MAX_WIDTH = lane_max_width # max average column width to be considered lane marker
    mean_width = []
    max_width = []
    while searched == False:
        end_col, match_r, match_c, width_r = find_adjacent(kmeans_img=kmeans_img, \
                        start_row=height-1, start_col=test_c, width=width, height=height) 
        width_rows.append(width_r)
        if end_col == width - 1:
            searched = True
        if (max(match_r) - min(match_r)) > MIN_LEN_R:
            width_std = np.std(width_r)
            if width_std < lane_max_widthstd and np.mean(width_r) > \
            lane_min_widthmean:
                print("Width mean: %0.3f" % np.mean(width_r))
                print("Width STD: %0.3f" % width_std)
                fit_good = True
                mean_width.append(np.mean(width_r))
                max_width.append(np.max(width_r))
                #if mean_width[-1] < MAX_WIDTH:
                if max_width[-1] < MAX_WIDTH:
                    fit_laneline = True
                    lane_r.append(match_r)
                    lane_c.append(match_c)
                    n_lanes += 1
                    # determine column spacing between this blueline and previous
                    if start_col_prev == []: # on first loop, col_prev is undefined
                        col_space.append(999)
                        start_col_prev = end_col # sav end_col to col_prev
                    else:
                        col_space.append(end_col - start_col_prev) # determine spacing between startcol
                        start_col_prev = end_col # update col_prev            
                else:
                    fit_unknown = True
        else:
            fit_good = False        
        test_c = end_col
    
    bluelines = []
    bl_cmin_at_la = []
    del_row = []
    segment_img = copy.copy(kmeans_img)
    MIN_COL_SPACE = lane_min_spacing # min spacing between blue lines below which considered single line
    if fit_laneline:
        # find blue lines that are so closely spaced in X as likely single line
        for i in range(len(col_space)):
            space = col_space[i]
            if space < MIN_COL_SPACE:
                del_row.append(i)
        # combine blue lines that are so closely spaced in X as likely single line
        del_row = np.array(del_row) # convert to numpy array so can do int subtraction
        for i in range(len(del_row)):
            row = del_row[i]
            match_r = lane_r[row] # save row data to append to previous BL
            match_c = lane_c[row] # save column data to append to previous BL
            del lane_r[row] # delete duplicate BL
            del lane_c[row] # delete duplicate BL
            lane_r[row-1] = lane_r[row-1] + match_r # save duplicate BL points to previous BL
            lane_c[row-1] = lane_c[row-1] + match_c # save duplicate BL points to previous BL
            n_lanes += -1 # decrement number of BL by one
            del_row += -1 # update delete row
        
        # fit 2nd order polynomial, define blue lines
        fit_r = np.linspace(0,scale_h-1,n_bl_points) # fixed spacing for building blue lines
        n_blpoint = n_bl_points # number of coordinates in blue line
        for i in range(n_lanes):
            match_c = lane_c[i]
            match_r = lane_r[i]
            cmax = max(match_c)
            cmin = min(match_c)
            c_bl_i = np.linspace(cmin,cmax,n_blpoint)
            r_bl_i = []
            # do polynomial fit: 
            [c2,c1,c0],res,rank,sval,rcond = np.polyfit(match_c, match_r, 2, full=True) 
            for j in range(n_blpoint):
                r_bl_i.append(c2*c_bl_i[j]**2 + c1*c_bl_i[j] + c0)
            c_bl_i = map(int,np.round(c_bl_i)) # convert to integer, image space
            r_bl_i = map(int,np.round(r_bl_i)) # convert to integer, image space               
            bluelines.append([r_bl_i,c_bl_i,c2,c1,c0,res,rcond])
        
        # color lane marker segments black
        for i in range(n_lanes):
            for j in range(len(lane_r[i])):
                segment_img[lane_r[i][j],lane_c[i][j],:] = [0,0,0]
        
        # color points along bluelines red
        for i in range(len(bluelines)):
            for j in range(len(bluelines[i][0])):
                if bluelines[i][0][j] < height and bluelines[i][1][j] < width:
                    segment_img[bluelines[i][0][j],bluelines[i][1][j],:] = [50,0,0]
                
        # define lookahead point for control
        la_r = height - cntrl_row # 76 = 96 - 20
        la_c = width/2 - 1 # 63 = 128/2 - 1
        
        # find x / column distance from lookahead point to blue lines
        segment_img = cv2.drawMarker(segment_img, (la_c,la_r), (90,128,30),
                markerType=cv2.MARKER_TILTED_CROSS, markerSize=5, 
                thickness=1, line_type=cv2.LINE_AA)
        # X@Y..C@R = -b +- (b^2 - 4ac)^.5/(2a
        # R = C2*C^2 + C1*C + (C0 - la_r)
        # C@R = -C1 +- (C1^2 - 4C2(C0-la_r))^.5/(2C2)
        bl_c_at_la = []
        bad_bl = []
        for i in range(len(bluelines)):
            line = bluelines[i]
            c2 = line[2]
            c1 = line[3]
            c0 = line[4]
            disc = c1**2 - 4*c2*(c0-la_r)
        
            if disc > 0:
                bl_c_at_la.append([(-c1 + disc**.5)/(2*c2), (-c1 - disc**.5)/(2*c2) ])
                # choose bl_c_at_la that is contained in the range of datapoints
                c_min = min(line[1])
                c_max = max(line[1])
                i_bl_c = -1
                if c_min < bl_c_at_la[-1][0] <= c_max:
                    i_bl_c = 0
                elif c_min < bl_c_at_la[-1][1] <= c_max:
                    i_bl_c = 1
                else:
                    #look-ahead point is not contained in range of datapoints,choose
                    #point closest to mean of columns in datapoints
                    c_mean = np.mean(line[1])
                    i_bl_c = np.argmin(np.abs(c_mean-bl_c_at_la[-1]))
                
                bl_cmin_at_la.append(bl_c_at_la[-1][i_bl_c])    
                segment_img=cv2.arrowedLine(segment_img, (la_c,la_r), \
                                            (int(np.round(bl_cmin_at_la[-1])),la_r),\
                                            (50,128,0),1)
            else:
                # negative discriminant, the line is not a valid blueline, pop
                bad_bl.append(i)
        # delete bad blue-lines (negative discriminate)
        for i in reversed(bad_bl):
            bluelines.pop(i)
    
    # guess at which BL is the one to control to
    # if 1 BL present, control to that one
    # if 3 BL present, control to most central one
    # if 2 BL present, control to least central one
    # if 4 BL present.... control to most central one ?
    n_bl = len(bluelines)
    i_bl = -1 # bluelines index to control to
    i_ll = -1 # leftlane index
    i_rl = -1 # rightlane index
    conf_bl = 0 # blueline, center lane, measurement confidence
    conf_ll = 0 # leftlane, measurement confidence
    conf_rl = 0 # rightlane, measurement confidence
    b_cntrl_bl = True
    if n_bl == 1:
        b_cntrl_bl = False
        dist_l_bl = la_c - bl_cmin_at_la[0] # sign determines if guess LL or RL
        if dist_l_bl > 0:
            i_ll = 0
            conf_ll = 75
        else:
            i_rl = 0
            conf_rl = 75
            
    elif n_bl == 2:
        dist_l_bl = abs(la_c - bl_cmin_at_la[0])
        dist_r_bl = abs(la_c - bl_cmin_at_la[1])
        if dist_l_bl > dist_r_bl:
            i_bl = 0
            conf_bl = 75
            i_rl = 1
            conf_rl = 75
        else:
            i_bl = 1
            conf_bl = 75
            i_ll = 0
            conf_ll = 75
    elif n_bl == 3:
        #dist_la = abs(la_c - np.array(bl_cmin_at_la))
        #i_bl = np.argmin(dist_la)
        i_bl = 1
        conf_bl = 90
        i_ll = 0
        conf_ll = 75
        i_rl = 2
        conf_rl = 75
    elif n_bl >= 4:
        # if theirs 4 or more lane lines, some must be duplicate.. chose
        # most central of lines
        mean_c = np.mean(bl_cmin_at_la)
        i_bl = np.argmin(np.abs(mean_c - bl_cmin_at_la))
        conf_bl = 50
        i_ll = i_bl - 1
        conf_ll = 25
        i_rl = i_bl + 1
        conf_rl = 25
    else:
        # n_bl = 0, no bl for control
        b_cntrl_bl = False
    
    if b_cntrl_bl:    
        segment_img = cv2.drawMarker(segment_img, (int(round(bl_cmin_at_la[i_bl])),la_r), (90,128,30),
            markerType=cv2.MARKER_SQUARE, markerSize=9, \
            thickness=1, line_type=cv2.LINE_AA)    

    # Prepare return datastring, csv, for ROS String message from nano to TX1 
    #
    bl_col_str = ""
    bl_string = ""
    if len(bl_cmin_at_la) > 0:
        for col in bl_cmin_at_la:
            bl_col_str = bl_col_str + "," + np.str(np.round(col,2))

        bl_string = \
            "i_bl,"+np.str(i_bl)+",i_ll,"+np.str(i_ll)+",i_rl,"+np.str(i_rl)+ \
            ",conf_bl,"+np.str(conf_bl)+",conf_ll,"+np.str(conf_ll)+",conf_rl,"+ \
            np.str(conf_rl)+",n_bl,"+np.str(len(bl_cmin_at_la))+","+bl_col_str
    print(bl_string) 
    return bl_string, segment_img

    # plt.figure(0)
    # plt.clf()
    # plt.axis('off')
    # plt.title('Quantized image ('+str(n_colors)+' colors, K-Means)')
    # plt.imshow(kmeans_img)
    
            
    # plt.figure(1)
    # plt.clf()
    # plt.axis('off')
    # plt.title('Segment image ('+str(n_colors)+' colors, K-Means)')
    # plt.imshow(segment_img)
