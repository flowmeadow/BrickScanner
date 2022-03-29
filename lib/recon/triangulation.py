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


if __name__ == "__main__":
    pass
