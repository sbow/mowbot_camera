#!/usr/bin/python

'''
    mowbot_camera_node.py
    Shaun Bowman
    Aug 9 2018

    openCV / cvbridge ROS implementatoin for monocular vision
    mowbot project
    based on RACECAR/J & MIT RACECAR
    '''

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class MowbotCameraNode:
    myVar = 0

    def __init__(self):
        myVar = 10
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv2.imshow("Image", cv_image)
        cv2.waitKey(1)




if __name__ == "__main__":
    rospy.init_node("mowbot_camera")
    node = MowbotCameraNode()
    rospy.spin()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
