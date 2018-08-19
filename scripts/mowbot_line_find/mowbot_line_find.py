#!/usr/bin/python

'''
mowbot_line_find.py
Shaun Bowman - shaun_bowman@hotmail.com - https://github.com/sbow/mowbot_camera
Aug 14 2018

mowbot_line_find
Class which handles CV images & finds a "line" on the ground.

ros agnostic
'''

import cv2
import numpy as np

class MowbotLineFind():
    meow = []
    debug = True
    img_cur = []
    img_cur_hsv = []
    img_cur_markup = []
    _crop_region_x = 640
    _crop_region_y = 290

    def __init__(self):
        meow = 0
    def updade_img_cur(self, img):
        self.img_cur = img

    def set_meow(self, mow_val):
        meow = mow_val

    def find_lines(self):

        # set non ROI to zeros
        img_cur_crop = self.img_cur.copy()
        img_cur_crop = img_cur_crop[self._crop_region_y:, :]

        # Convert BGR to HSV
        self.img_cur_hsv = cv2.cvtColor(img_cur_crop, cv2.COLOR_BGR2HSV)

        # Color mask - look for blue line - in hsv https://www.rapidtables.com/convert/color/rgb-to-hsv.html
        lower = np.array([230, 58, 35]) #230 67 35
        upper = np.array([231, 67, 47]) #231 58 47
        blue_mask = cv2.inRange(self.img_cur_hsv, lower, upper)
        self.img_cur_markup = cv2.bitwise_and( self.img_cur_hsv, self.img_cur_hsv, blue_mask)

        # Gray image & do guassian blur
        self.img_cur_markup = cv2.cvtColor(self.img_cur_markup, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(self.img_cur_markup, cv2.COLOR_BGR2GRAY)

        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # edge detection using canny
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        # get lines using HoughLinesP
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(self.img_cur_markup) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        # Draw the lines on the  image
        # self.img_cur_markup = cv2.addWeighted(self.img_cur, 0.8, line_image, 1, 0)
        self.img_cur_markup[self._crop_region_y:, self._crop_region_x:] =   line_image #+ self.img_cur_markup[
                                                                            #self._crop_region_y:, self._crop_region_x:]

        # Plot if desired
        if self.debug:
            self.plot_image()

        pass

    def draw_debug_grid(self):
        # show an image w grid-lines
        n_row, n_col, n_colors = self.img_cur.shape
        line_width = 1

        if n_row % 2 == 0:
            line_width = 2

        center_row = int( np.floor( n_row / 2) )
        center_col = int( np.floor( n_col / 2) )
        self.img_cur_markup = self.img_cur
        self.img_cur_markup[ center_row:(center_row + line_width), :] = [0, 0, 0]
        self.img_cur_markup[ :, center_col:(center_col + line_width)] = [0, 0, 0]

        self.plot_image()

    def plot_image(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 960, 540)
        cv2.imshow('image', self.img_cur_markup)
        cv2.waitKey(1)
        pass