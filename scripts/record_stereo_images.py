#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Records several stereo images with a stereo camera setup
@File      : record_stereo_images.py
@Project   : BrickScanner
@Time      : 11.02.22 20:22
@Author    : flowmeadow
"""
import os
import sys

sys.path.append(os.getcwd())  # required to run script from console

import cv2
from lib.camera.stereo_cam import StereoCam
from lib.data_management import append_img_pair, new_stereo_img_dir


def main():
    # create a new directory based on date and time
    img_dir = new_stereo_img_dir()

    # init cameras
    cam = StereoCam()

    print("Start main loop")
    while True:
        # Capture frame-by-frame
        frame_1, frame_2 = cam.read()

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        elif key & 0xFF == ord("s"):
            print("Saving both frames")
            append_img_pair(img_dir, frame_1, frame_2)

        # Display the resulting frame
        frame = cv2.hconcat([frame_1, frame_2])
        frame = cv2.resize(frame, (1760, 720))
        cv2.imshow("frame", frame)

        if key & 0xFF == ord("s"):
            cv2.waitKey(500)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
