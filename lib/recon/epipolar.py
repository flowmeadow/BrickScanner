#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains methods dealing with epipolar geometry
@File      : epipolar.py
@Project   : BrickScanner
@Time      : 29.03.22 18:04
@Author    : flowmeadow
"""
import numpy as np


def epipolar_distance(p_l: np.ndarray, p_r: np.ndarray, F: np.ndarray) -> float:
    """
    error value to measure the correspondence between two points x (p_l) and x' (p_r) wrt. the fundamental matrix
    :param p_l: 2D point (x, y)
    :param p_r: 2D point (x, y)
    :param F: fundamental matrix
    :return: error measure
    """
    p_l, p_r = np.append(p_l, 1), np.append(p_r, 1)  # make points homogeneous
    a = np.power(F @ p_l.T, 2)[1]
    b = np.power(F @ p_l.T, 2)[2]
    return np.abs(p_l.T @ F @ p_r) / np.sqrt(a + b)


def compute_F(T_W1: np.ndarray, T_W2: np.ndarray, K_1: np.ndarray, K_2: np.ndarray) -> np.ndarray:
    """
    Computes the fundamental matrix given two camera poses and camera matrices
    (MVG page 244)
    TODO reference
    F = K_1^{-T} [t]_x R K^{-1} where R and t result from the transformation from camera 1 to camera 2
    :param T_W1: Transformation matrix for cam 1 (4, 4)
    :param T_W2: Transformation matrix for cam 2 (4, 4)
    :param K_1: camera matrix for cam 1 (3, 3)
    :param K_2: camera matrix for cam 2 (3, 3)
    :return: Fundamental matrix (3, 3)
    """
    T_12 = np.linalg.inv(T_W1) @ T_W2  # transformation matrix from cam 1 to cam 2

    t = T_12[:3, 3]  # translation
    R = T_12[:3, :3]  # rotation
    S = np.mat([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])  # skew symmetric matrix [t]_x
    E = S @ R  # essential matrix
    K_1_inv = np.linalg.inv(K_1)  # camera matrix inverse 1
    K_2_inv = np.linalg.inv(K_2)  # camera matrix inverse 1
    F = K_2_inv.T * (E * K_1_inv)
    return F


if __name__ == "__main__":
    pass
