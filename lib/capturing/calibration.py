#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : calibration methods
@File      : calibration.py
@Project   : BrickScanner
@Time      : 06.08.22 16:57
@Author    : flowmeadow
"""
import math
import sys
from typing import Tuple

import cv2
import numpy as np
from lib.helper.cloud_operations import construct_T


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

    objp -= np.array([*center_point, 0])  # shift center point
    objp = cell_width * objp  # adjust dimension
    objp[:, :-1] = np.flip(objp[:, :-1], axis=1)  # flip x and y axis
    if flip_view:
        objp = -objp
    objp[:, 2] += z_value

    return ret, corners_1, corners_2, objp


def rodrigues_vec_to_rotation_mat(rodrigues_vec: np.ndarray) -> np.ndarray:
    """
    Computes a 3x3 rotation matrix from a given rodrigues vector
    :param rodrigues_vec: rodrigues vector (3,)
    :return: rotation matrix (3, 3)
    """
    theta = np.linalg.norm(rodrigues_vec)
    init_rot = np.eye(3, dtype=float)
    if theta < sys.float_info.epsilon:
        return init_rot
    else:
        r = rodrigues_vec / theta
        r_rT = np.array(
            [
                [r[0] * r[0], r[0] * r[1], r[0] * r[2]],
                [r[1] * r[0], r[1] * r[1], r[1] * r[2]],
                [r[2] * r[0], r[2] * r[1], r[2] * r[2]],
            ]
        )
        r_cross = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        rotation_mat = math.cos(theta) * init_rot + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat


def get_world_space_poses(
    frame_1: np.ndarray,
    frame_2: np.ndarray,
    c_size: Tuple[int, int],
    cell_width: float,
    K_1: np.ndarray,
    K_2: np.ndarray,
    dist_1: np.ndarray,
    dist_2: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes and returns the camera's world poses based on given calibration images,
    which show the checkerboard pattern, and the corresponding calibration data
    :param frame_1: calibration image of camera 1 (h, w)
    :param frame_2: calibration image of camera 2 (h, w)
    :param cell_width: length of a checkerboard cell edge in cm
    :param c_size: number of the checkerboards rows and columns
    :param K_1: camera matrix of camera 1 (3, 3)
    :param K_2: camera matrix of camera 2 (3, 3)
    :param dist_1: distortion coefficients of camera 1 (1, 5)
    :param dist_2: distortion coefficients of camera 1 (1, 5)
    :param kwargs: forwarded keyboard arguments for 'find_chessboard' method
    :return: camera poses in world space [(4, 4), (4, 4)]
    """
    ret, corners_1, corners_2, objp = find_chessboard(frame_1, frame_2, c_size, cell_width, **kwargs)

    if not ret:
        raise ValueError("Could not find chessboard corners")

    ret_1, r_1, t_1 = cv2.solvePnP(objp, corners_1, K_1, dist_1)
    ret_2, r_2, t_2 = cv2.solvePnP(objp, corners_2, K_2, dist_2)
    if not (ret_1 and ret_2):
        raise ValueError("Could not find relative point poses")
    # compute transformation matrices
    R_1, t_1 = rodrigues_vec_to_rotation_mat(r_1.flatten()), t_1.flatten()
    R_2, t_2 = rodrigues_vec_to_rotation_mat(r_2.flatten()), t_2.flatten()

    T_1W = construct_T(R_1, t_1)
    T_2W = construct_T(R_2, t_2)
    return T_1W, T_2W
