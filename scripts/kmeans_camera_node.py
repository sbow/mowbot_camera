#!/usr/bin/python

'''
    kmeans_camera_node.py
    Shaun Bowman
    December 6 2019
'''

# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
import kmeans_lanes as kl
from time import time
import os

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=60,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            img[1] = img[1]/255.0
            print(img.shape)
            time0 = time()
            bl_string, seg_img = kl.get_bl(img)

            print("Delta t for complete kmeans function: %0.3f"%( time() - time0))
            cat_img = np.concatenate((img, cv2.resize(seg_img, (640,360), \
                                                     interpolation=cv2.INTER_AREA)), axis=1)
            cv2.imshow("CSI Camera", cat_img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

def kmeansCam():
    show_cam = True
    if show_cam:
        show_camera()
    ret_val = []
    img = []
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    pub = rospy.Publisher('KmeansMsg', String, queue_size=10)
    rospy.init_node('KmeansCam', anonymous=True)
    rate = rospy.Rate(30) # 30 hz
    i = 0
    f_name_raw = "Raw_Track_"
    f_name_seg = "Seg_Track_"
    img_n = 0
    os.chdir("/home/nvidia")
    while not rospy.is_shutdown():
        kmeans_msg = "Kmeans Running"
        if cap.isOpened():
            ret_val, img = cap.read()
            img_name = 'REC_'+str(img_n)+'.jpg'
            cv2.imwrite(img_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            img_n = img_n + 1
            bl_string, seg_img = kl.get_bl(img, scale_w=96, scale_h=54, lane_len_min=5, n_colors=5)
            kmeans_msg = kmeans_msg + "," + bl_string + "," + img_name
            #cv2.imwrite(f_name_raw+str(i)+".png",img)
            #cv2.imwrite(f_name_seg+str(i)+".png",seg_img)
            i = i + 1
        rospy.loginfo(kmeans_msg)
        pub.publish(kmeans_msg)
        rate.sleep()

if __name__ == "__main__":
    #show_camera()
    try:
        kmeansCam()
    except rospy.ROSInterruptException:
        cap.release()
        if show_cam:
            cv2.destroyAllWindows()
        pass
