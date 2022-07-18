#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains triangulation methods
@File      : triangulation.py
@Project   : BrickScanner
@Time      : 29.03.22 17:47
@Author    : flowmeadow
"""

import numpy as np
from scipy import linalg
from cv2 import triangulatePoints


def dlt(P1: np.array, P2: np.array, point_1: np.array, point_2: np.array) -> np.array:
    """
    Performs a direct line transformation
    :param P1: projection matrix of camera 1
    :param P2: projection matrix of camera 2
    :param point_1: 2D point of image 1
    :param point_2: 2D point of image 2
    :return: triangulated 3D point
    """
    A = [
        point_1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point_1[0] * P1[2, :],
        point_2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point_2[0] * P2[2, :],
    ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]


def triangulate_points(
    kpts_1: np.ndarray,
    kpts_2: np.ndarray,
    K_1: np.ndarray,
    K_2: np.ndarray,
    T_W1: np.ndarray,
    T_W2: np.ndarray,
) -> np.ndarray:
    """
    Performs triangulation for given keypoints from 2 camera image sets based on camera matrices and camera poses
    :param kpts_1: array of keypoints from camera 1 (2, n)
    :param kpts_2: array of keypoints from camera 2 (2, n)
    :param K_1: camera matrix of camera 1 (3, 3)
    :param K_2: camera matrix of camera 2 (3, 3)
    :param T_W1: transformation matrix for pose of camera 1 (4, 4)
    :param T_W2: transformation matrix for pose of camera 2 (4, 4)
    :return: reconstructed 3d points in world space (n, 3)
    """
    T_1W = np.linalg.inv(T_W1)  # transformation matrix from 3d world space to 3d cam 1 space
    T_2W = np.linalg.inv(T_W2)  # transformation matrix from 3d world space to 3d cam 2 space
    P1 = K_1 @ T_1W[:3]  # projection matrix from 3d world space to cam 1 image space
    P2 = K_2 @ T_2W[:3]  # projection matrix from 3d world space to cam 2 image space

    # triangulate keypoints
    pts_recon = triangulatePoints(P1, P2, kpts_1, kpts_2).T
    # returns are homogeneous, so they need to be transformed back to 3D
    pts_recon = (pts_recon.T / pts_recon[:, 3]).T[:, :3]
    return pts_recon


if __name__ == "__main__":
    pass
