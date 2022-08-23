#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : calibration methods
@File      : calibration.py
@Project   : BrickScanner
@Time      : 06.08.22 16:57
@Author    : flowmeadow
"""
from typing import Tuple

import cv2
import numpy as np


def find_chessboard(
    frame_1: np.array,
    frame_2: np.array,
    c_size: Tuple[int, int],
    cell_width: float,
    criteria=None,
    center_point=(0, 0),
    z_value=0,
    flip_view=False,
) -> Tuple[bool, np.array, np.array, np.array]:
    """
    Looks for checkerboard corners in images and computes the 3D coordinates of these corners
    :param frame_1: left image frame
    :param frame_2: right image frame
    :param c_size: dimension of the chess corner grid
    :param cell_width: width of a chess cell square in mm
    :param criteria: calibration criteria
    :param center_point: define origin
    :param z_value: z-position of the chessboard
    :param flip_view: flip coordinate axes
    :return: Tuple of 4 elements containing:
        a boolean return value (only true if checkerboard was detected in both frames)
        the 2D corner coordinates for frame_1
        the 2D corner coordinates for frame_2
        the 3D corner coordinates
    """
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    ret = False

    # Convert images into grayscale
    gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    # Find 2D corners
    c_ret_1, corners_1 = cv2.findChessboardCorners(gray_1, c_size, None)
    c_ret_2, corners_2 = cv2.findChessboardCorners(gray_2, c_size, None)

    # If corners found in both images ...
    if c_ret_1 and c_ret_2:
        ret = True

        # refine corner positions
        corners_1 = cv2.cornerSubPix(gray_1, corners_1, (11, 11), (-1, -1), criteria)
        corners_2 = cv2.cornerSubPix(gray_2, corners_2, (11, 11), (-1, -1), criteria)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((c_size[0] * c_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : c_size[0], 0 : c_size[1]].T.reshape(-1, 2)

    objp -= np.array([*center_point, z_value])  # shift center point
    objp = cell_width * objp  # adjust dimension
    objp[:, :-1] = np.flip(objp[:, :-1], axis=1)  # flip x and y axis
    if flip_view:
        objp = -objp

    return ret, corners_1, corners_2, objp
