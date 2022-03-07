#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Visualizes the undistortion of straight blue lines
@File      : check_camera_matrix.py
@Project   : BrickScanner
@Time      : 07.03.22 11:47
@Author    : flowmeadow
"""
import cv2
import numpy as np
from definitions import *
from lib.data_management import params_from_json


def main():
    cam_idx = 2  # [2, 0]
    cam = cv2.VideoCapture(cam_idx)
    K, dist = params_from_json(f"{CAM_DIR}/params.json")

    undistort = True
    while True:
        ret, frame = cam.read()
        if undistort:
            frame = cv2.undistort(frame, K, dist)

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 50, 100], dtype="uint8")
        upper = np.array([40, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
        # frame = cv2.bitwise_and(frame, frame, mask=mask)
        # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
        edges = cv2.Canny(mask, 100, 200)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        for line in lines:
            for x1, y1, x2, y2 in line:
                if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 200:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        if key & 0xFF == ord("s"):  # swap between distorted and undistorted image
            undistort = not undistort

        cv2.imshow("frame", frame)


if __name__ == "__main__":
    main()
