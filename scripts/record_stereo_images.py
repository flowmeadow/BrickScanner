#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : record_stereo_images.py
@Project   : LECVO
@Time      : 11.02.22 20:22
@Author    : flowmeadow
"""
import os
import time
from datetime import datetime

import cv2
from definitions import *

VIDEO_RES = (352, 288)
IMAGE_RES = (1280, 960)


def change_resolutions(cam, resolution):
    print(f"Change camera resolution to {resolution[0]}x{resolution[1]}")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])


def main():
    date_time = datetime.now().strftime("%y%m%d-%H%M%S")
    image_path = f"{IMG_DIR}/{date_time}"

    dir_left = f"{image_path}/left"
    dir_right = f"{image_path}/right"
    for directory in [image_path, dir_left, dir_right]:
        os.mkdir(directory)
    print(f"Created directory {image_path}")

    print("Initialize cameras")
    cam_ids = [2, 0]  # left, right
    cam_1 = cv2.VideoCapture(cam_ids[0])
    cam_2 = cv2.VideoCapture(cam_ids[1])
    for cam in [cam_1, cam_2]:
        change_resolutions(cam, VIDEO_RES)

    print("Start main loop")
    img_count = 0
    focus = 0
    while True:
        # Capture frame-by-frame
        ret_1, frame_1 = cam_1.read()
        ret_2, frame_2 = cam_2.read()

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        elif key & 0xFF == ord("s"):
            cam_2.release()
            change_resolutions(cam_1, IMAGE_RES)
            ret_1, frame_1 = cam_1.read()
            cam_1.release()
            cam_2 = cv2.VideoCapture(cam_ids[1])
            change_resolutions(cam_2, IMAGE_RES)
            ret_2, frame_2 = cam_2.read()
            cam_1 = cv2.VideoCapture(cam_ids[0])
            for cam in [cam_1, cam_2]:
                change_resolutions(cam, VIDEO_RES)

            if ret_1 and ret_2:
                print("Saving both frames")
                file_name = f"image_{img_count}.png"
                cv2.imwrite(f"{dir_left}/{file_name}", frame_1)
                cv2.imwrite(f"{dir_right}/{file_name}", frame_2)

                frame = cv2.hconcat([frame_1, frame_2])
                frame = cv2.resize(frame, (1760, 720))

                wait_time = 1
                start_time = time.time()
                while start_time + wait_time > time.time():
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                img_count += 1
            else:
                print("WARNING: Failed to take picture")
            for cam in [cam_1, cam_2]:
                change_resolutions(cam, VIDEO_RES)
            continue

        # Display the resulting frame
        frame = cv2.hconcat([frame_1, frame_2])
        frame = cv2.resize(frame, (1760, 720))
        cv2.imshow("frame", frame)

    cam_1.release()
    cam_2.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
