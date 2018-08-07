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

class MowbotCameraNode:
    myVar = 0

    def __init__(self):
        myVar = 10



if __name__ == "__main__":
    rospy.init_node("mowbot_camera")
    node = MowbotCameraNode()
    rospy.spin()