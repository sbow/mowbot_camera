#!/usr/bin/python
'''
line_fit.py
2018 / 09 / 03
Shaun Bowman

line_fit.py is an experimental script to come up with a method of interpreting the outpuf of Hough Lines
detection. The output of Hough Lines are lines in the format (x1), (y1), (x2), (y2)
This function will come up with a 2nd order polynomial best-fit to the set of these lines.
The lines will be broken down into a set of XY points.
A best fit 2nd order polynomial will be fit to the set of these points.
Lines on the edge of the image, by products of image processing, should be ignored.

Later, another 3rd order polynomial will be generated in potentially another script which will come
be ploted from the center of the lowest row, to a point tangent to the first 2nd order polynomial
discussed above. This second 3rd order line will be the desired trajectory of the robot.

The input to the Hough Lines calculation is a forward facing camera image which has been converted
to a "birds eye" perspective such that it is linear in real world x&y coordinates & suitable for
path planning without much additional processing.
'''

import re
import matplotlib.pyplot as plt
import numpy as np

_image_path = "/home/nvidia/racecar-ws/src/mowbot_camera/scripts/experimental/"
_file_name = "lines.txt" # individual frames are seperated by "END\n", elements seperated by ","

#f = open(_image_path + _file_name, "r")
f = open(_file_name, "r")

file_contents = f.read() # x1,y1,x2,y2\n....END\n

match = re.search(r"[0-9,\n]*", file_contents) # Get lines, seperate on "END"

lines_frame_temp = match.group() # get contents of match; x1i,y1i,x2i,y2i\nx1j,y1j,x2j,y2j\n...
lines_frame_temp2 = lines_frame_temp.split("\n")
lines_frame_temp2.pop() # remove extra blank line
nLines = len( lines_frame_temp2 )
lines = np.empty((nLines, 4))

i = 0
while lines_frame_temp2:
    line = lines_frame_temp2.pop()
    lines[i,:] = line.split(",") # x1,y1,x2,y2 -> [[x1],[y1],[x2],[y2]]
    i = i + 1

nSplitLine = 3 # split line segment into 3; get 4 xy points per line
xScale = 0.01 # m per pixel, x direction, image coords
yScale = 0.02 # m per pixel, y direction, image coords
pointsX = np.empty( (nLines*4) ) # [[x1, x2, x3,...]
pointsY = np.empty( (nLines*4) ) # [[y1, y2, y3,...]

for i in range(nLines):
    x1 = lines[i, 0] #
    y1 = lines[i, 1]
    x4 = lines[i, 2]
    y4 = lines[i, 3]

    xDelta = x4 - x1
    yDelta = y4 - y1

    xStep = xDelta / (nSplitLine - 1)
    yStep = yDelta / (nSplitLine - 1)

    x2 = x1 + xStep
    y2 = y1 + yStep
    x3 = x2 + xStep
    y3 = y2 + yStep

    pointsX[i*4:(i*4 + 4)] = [x1, x2, x3, x4]
    pointsY[i*4:(i*4 + 4)] = [y1, y2, y3, y4]
