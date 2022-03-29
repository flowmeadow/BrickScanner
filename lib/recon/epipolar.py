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
    error value to measure the correspondance between two points x (p_l) and x' (p_r) wrt. the fundamental matrix
    :param p_l: 2D point (x, y)
    :param p_r: 2D point (x, y)
    :param F: fundamental matrix
    :return: error measure
    """
    p_l, p_r = np.append(p_l, 1), np.append(p_r, 1)  # make points homogeneous
    a = np.power(F @ p_l.T, 2)[1]
    b = np.power(F @ p_l.T, 2)[2]
    return np.abs(p_l.T @ F @ p_r) / np.sqrt(a + b)


if __name__ == "__main__":
    pass
