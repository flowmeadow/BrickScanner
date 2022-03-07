#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Checks the fundamental matrix
@File      : check_f_mat.py
@Project   : BrickScanner
@Time      : 07.03.22 15:39
@Author    : flowmeadow
"""
import cv2
import numpy as np
from definitions import *
from lib.cam_mangement import change_resolutions

global mouseX, mouseY


DISPLAY_RES = (1760, 720)


class InteractiveWindow(object):
    def __init__(self, name):
        self.name = name
        self.mouse_pos_x, self.mouse_pos_y = None, None
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.get_mouse_pos)

    def get_mouse_pos(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.mouse_pos_x, self.mouse_pos_y = x, y

    def imshow(self, frame):
        cv2.imshow(self.name, frame)


def main():
    use_image = True  # TODO: due to wrong format, video is not working

    cam_1, cam_2 = None, None
    imgLeft, imgRight = None, None
    if use_image:
        directory_name = "test_01"  # "epipolar"
        image_path = f"{IMG_DIR}/{directory_name}"
        for file_name in sorted(os.listdir(f"{image_path}/left")):
            imgLeft = cv2.imread(f"{image_path}/left/{file_name}")
            imgRight = cv2.imread(f"{image_path}/right/{file_name}")
            break
        img_res = IMAGE_RES
    else:
        print("Initialize cameras")
        cam_ids = [2, 0]  # left, right
        cam_1 = cv2.VideoCapture(cam_ids[0])
        cam_2 = cv2.VideoCapture(cam_ids[1])
        for cam in [cam_1, cam_2]:
            change_resolutions(cam, VIDEO_RES)

    win = InteractiveWindow("frame")

    F = np.load(f"{SETUP_DIR}/F_mat.npy")
    print("Start main loop")
    while True:
        # Capture frame-by-frame
        if use_image:
            frame_1 = imgLeft.copy()
            frame_2 = imgRight.copy()
        else:
            ret_1, frame_1 = cam_1.read()
            ret_2, frame_2 = cam_2.read()

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        # Display the resulting frame

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        x, y = win.mouse_pos_x, win.mouse_pos_y
        if x is not None and y is not None:
            if x < DISPLAY_RES[0] / 2:
                x = int(x * 2 * IMAGE_RES[0] / DISPLAY_RES[0])
                y = int(y * IMAGE_RES[1] / DISPLAY_RES[1])
                frame_1 = cv2.circle(frame_1, (x, y), 10, (0, 0, 255), 2)

                r, c = frame_1.shape[:2]
                line = cv2.computeCorrespondEpilines(np.array([x, y]).reshape(-1, 1, 2), 1, F)
                line = line.flatten()

                x0, y0 = map(int, [0, -line[2] / line[1]])
                x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])

                frame_2 = cv2.line(frame_2, (x0, y0), (x1, y1), (0, 0, 255), 1)
            else:
                x = int((x - DISPLAY_RES[0] / 2) * 2 * IMAGE_RES[0] / DISPLAY_RES[0])
                y = int(y * IMAGE_RES[1] / DISPLAY_RES[1])
                frame_2 = cv2.circle(frame_2, (x, y), 10, (0, 0, 255), 2)

                r, c = frame_1.shape[:2]
                line = cv2.computeCorrespondEpilines(np.array([x, y]).reshape(-1, 1, 2), 2, F)
                line = line.flatten()

                x0, y0 = map(int, [0, -line[2] / line[1]])
                x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])

                frame_1 = cv2.line(frame_1, (x0, y0), (x1, y1), (0, 0, 255), 1)

        frame = cv2.hconcat([frame_1, frame_2])
        frame = cv2.resize(frame, DISPLAY_RES)

        win.imshow(frame)


if __name__ == "__main__":
    main()
