#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : detect_color.py
@Project   : BrickScanner
@Time      : 23.08.22 11:25
@Author    : flowmeadow
"""
import os
import sys

sys.path.append(os.getcwd())  # required to run script from console

import cv2
from lib.camera.stereo_cam import Cam
from lib.helper.data_management import append_img_pair, new_stereo_img_dir


def main():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # init cameras
    cam = Cam(0)
    cam.start()
    print("Start main loop")
    while True:
        # Capture frame-by-frame
        frame = cam.read()
        if frame is None:
            continue

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned
                # in top-left, top-right, bottom-right, and bottom-left
                # order)
                corners = markerCorner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the
                # ArUco marker
                cX = int((top_left[0] + bottom_right[0]) / 2.0)
                cY = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the frame
                cv2.putText(frame, str(markerID), (top_left[0], top_left[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("frame", frame)
    cam.stop()


if __name__ == "__main__":
    main()
