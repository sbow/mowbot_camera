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
            bl_string, seg_img = kl.get_bl(img)
            cv2.imshow("CSI Camera", seg_img)
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
    show_cam = False
    if show_cam:
        show_camera()
    ret_val = []
    img = []
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    pub = rospy.Publisher('KmeansMsg', String, queue_size=10)
    rospy.init_node('KmeansCam', anonymous=True)
    rate = rospy.Rate(30) # 30 hz
    while not rospy.is_shutdown():
        kmeans_msg = "Kmeans Running"
        if cap.isOpened():
            ret_val, img = cap.read()
            bl_string, seg_img = kl.get_bl(img)
            kmeans_msg = kmeans_msg + "," + bl_string
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
