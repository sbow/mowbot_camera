#!/usr/bin/python
'''
noros_mowbot_camera_nod.py
Shaun Bowman
Aug 16 2018

For debugging image processing using straight python.
No ROS dependancy

'''

import cv2
from mowbot_line_find.mowbot_line_find import MowbotLineFind

_image_path = "/home/shaun/catkin_ws/src/mowbot_camera/scripts/experimental/"
_file_name = "one_image.png"

mowbot_line_find = MowbotLineFind()
img = cv2.imread(_image_path + _file_name)

mowbot_line_find.updade_img_cur(img)
mowbot_line_find.draw_debug_grid()

