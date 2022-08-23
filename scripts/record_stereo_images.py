#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Records several stereo images with a stereo camera real_setup
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
from lib.helper.data_management import append_img_pair, new_stereo_img_dir


def main():
    stream = True

    # create a new directory based on date and time
    img_dir = new_stereo_img_dir()

    # init cameras
    cam = StereoCam(frame_rate=60, resolution=(160, 120))

    print("Start main loop")
    stream_is_running = False
    buffer = []
    while True:
        # Capture frame-by-frame
        frame_1, frame_2 = cam.read()

        frame_2 = cv2.rotate(frame_2, cv2.ROTATE_180)  # rotate second frame about 180°

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        elif key & 0xFF == ord("s"):
            if stream:
                if stream_is_running:
                    print("Stopped stream")
                    # save buffer
                    for pair in buffer:
                        append_img_pair(img_dir, *pair)
                else:
                    print("Started stream")

                stream_is_running = not stream_is_running

            else:
                print("Saving both frames")
                append_img_pair(img_dir, frame_1, frame_2)

        if stream_is_running and cam.updated():
            buffer.append((frame_1, frame_2))

        # Display the resulting frame
        frame = cv2.hconcat([frame_1, frame_2])
        frame = cv2.resize(frame, (1760, 720))
        cv2.imshow("frame", frame)

        if key & 0xFF == ord("s") and not stream:
            cv2.waitKey(500)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
