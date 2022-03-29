#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Checks the fundamental matrix
@File      : check_f_mat.py
@Project   : BrickScanner
@Time      : 07.03.22 15:39
@Author    : flowmeadow
"""
import os
import sys

sys.path.append(os.getcwd())  # required to run script from console

from typing import Tuple

import cv2
import numpy as np
from definitions import *
from lib.windows.interactive_window import InteractiveWindow

DISPLAY_RES = (int(2560 * 0.75), int(720 * 0.75))


def draw_line_for_point(frame: np.array, p: Tuple[int, int], F: np.array, img_idx: int) -> np.array:
    """
    Computes the epiline for a 2D point p and draws it in the given frame
    :param frame: image frame to draw the line in
    :param p: 2D point (x, y)
    :param F: fundamental matrix
    :param img_idx: defines if the epiline in the left (1) or right (2) image is computed
    :return:
    """
    if img_idx not in [1, 2]:
        raise NotImplementedError

    r, c = frame.shape[:2]
    # compute line params
    line = cv2.computeCorrespondEpilines(np.array(p).reshape(1, 2), img_idx, F)
    line = line.flatten()

    # draw line to frame
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])

    return cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)


def main(directory_name, img_idx=0):
    """
    Program that displays the corresponding epiline for a clicked point.
    TODO: check F not only from stored images but real time camera captures
    :param directory_name: name of the directory to load the image pair from
    :param img_idx: index of the image pair to chose
    """

    # load image pair from directory
    img_path = f"{IMG_DIR}/{directory_name}"
    file_name = sorted(os.listdir(f"{img_path}/left"))[img_idx]
    imgLeft = cv2.imread(f"{img_path}/left/{file_name}")
    imgRight = cv2.imread(f"{img_path}/right/{file_name}")

    # initialize window
    win = InteractiveWindow("frame")

    # load fundamental matrix
    F = np.load(f"{SETUP_DIR}/F.npy")

    frame_l = imgLeft.copy()
    frame_r = imgRight.copy()

    # program loop
    while True:

        # check for quit (q)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        # check for mouse button press
        x, y = win.mouse_pos_x, win.mouse_pos_y
        if x is not None and y is not None:
            frame_l = imgLeft.copy()
            frame_r = imgRight.copy()
            if x < DISPLAY_RES[0] / 2:  # left image half was clicked
                x = int(x * 2 * frame_l.shape[1] / DISPLAY_RES[0])
                y = int(y * frame_l.shape[0] / DISPLAY_RES[1])
                frame_l = cv2.circle(frame_l, (x, y), 10, (0, 0, 255), 2)  # draw circle at mouse pos
                frame_r = draw_line_for_point(frame_r, (x, y), F, 1)  # draw epiline in other image
            else:  # right image half was clicked
                x = int((x - DISPLAY_RES[0] / 2) * 2 * frame_l.shape[1] / DISPLAY_RES[0])
                y = int(y * frame_l.shape[0] / DISPLAY_RES[1])  # draw circle at mouse pos
                frame_r = cv2.circle(frame_r, (x, y), 10, (0, 0, 255), 2)
                frame_l = draw_line_for_point(frame_l, (x, y), F, 2)  # draw epiline in other image
            win.reset_mouse()

        # concatenate images and show final frame
        frame = cv2.hconcat([frame_l, frame_r])
        frame = cv2.resize(frame, DISPLAY_RES)
        win.imshow(frame)


if __name__ == "__main__":
    main("test_06", 1)
