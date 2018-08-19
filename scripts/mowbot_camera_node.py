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
from mowbot_line_find.mowbot_line_find import MowbotLineFind

class MowbotCameraNode:
    myVar = 0
    debug_find_line = False
    debug = False
    debug_one_frame = True
    debug_show_grid = True
    _frame_count = 0
    _image = []
    _frame_stop = 3
    #_image_path = "/home/shaun/catkin_ws/src/mowbot_camera/scripts/experimental/"
    _image_path = "/home/nvidia/racecar-ws/src/mowbot_camera/scripts/experimental/"
    _file_name = "one_image.png"

    line_find = MowbotLineFind()

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

    def callback(self,data):

        if self._frame_count < self._frame_stop:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

            self.line_find.updade_img_cur(cv_image)

            if self.debug_find_line:
                self.line_find.find_lines()

            if self.debug:
                cv2.imshow("Image", cv_image)
                cv2.waitKey(1)

            if self.debug_one_frame:
                self._frame_count = self._frame_count + 1
                self._image = cv_image

            if self.debug_show_grid:
                self.line_find.draw_debug_grid()

        else:

            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

            if self._frame_count == self._frame_stop:
                cv2.imwrite(self._image_path + self._file_name, self._image)
                self._frame_count = self._frame_count + 1

            if self.debug_show_grid:
                self.line_find.draw_debug_grid()

            else:
                self._image = cv_image
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 960, 540)
                cv2.imshow('image', self._image)
                cv2.waitKey(1)
            pass




if __name__ == "__main__":
    rospy.init_node("mowbot_camera")
    node = MowbotCameraNode()
    rospy.spin()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
