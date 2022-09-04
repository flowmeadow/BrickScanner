#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Color operations
@File      : color.py
@Project   : BrickScanner
@Time      : 04.09.22 16:28
@Author    : flowmeadow
"""
from typing import Tuple

import cv2
import numpy as np


def hsv_to_3d(color: np.ndarray, weights=np.ones(3)) -> np.ndarray:
    """
    Computes Euclidian 3D position of a given hsv color
    :param color: hsv color (3,)
    :param weights: channel weights (3,)
    :return: 3D position (3,)
    """
    h, s, v = color
    s, v = s * weights[1], v * weights[1]  # weight saturation and value channel
    h, s, v = h / 180, s / 255, v / 255  # normalize
    angle = 2 * np.pi * h
    x = s * np.cos(angle)
    y = s * np.sin(angle)
    return v * np.array([x, y, 1])


def compare_hsv_colors(
    source: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray = np.ones(3),
    d_thresh: float = np.inf,
    r_thresh: float = np.inf,
) -> Tuple[int, float, float]:
    """
    compares a source color with a list of target colors and returns the best match
    :param source: hsv source color (3,)
    :param targets: list of hsv target colors (n, 3)
    :param weights: hsv channel weights (3,)
    :param d_thresh: distance threshold. The Euclidian distance between source and best target has to be lower than this
    :param r_thresh: ratio threshold. The Euclidian distance ratio between best and
    second-best target has to be lower than this
    :return: Index of the best match, distance error and ratio error
    """

    source = hsv_to_3d(source, weights)
    targets = np.array([hsv_to_3d(t, weights) for t in targets])
    err = np.sum((targets - source) ** 2, axis=1)
    first, second = np.sort(err)[:2]

    best = None
    if first < d_thresh and first / second < r_thresh:
        best = np.argsort(err)[0]
    return best, first, first / second


def hsv_from_region(img: np.ndarray, p_1: np.ndarray, p_2: np.ndarray) -> np.ndarray:
    """
    obtains the mean hsv color from a given sub-region defined by p_1 and p_2
    TODO: Average of HSV has to be computed differently. E.g. for red, the hue value (h) is
          approximately 0-5 or 175-180. Averaging pixels for red could lead to different hue values around 87-93.
    :param img: source image (h, w, 3)
    :param p_1: 2D point defining the top left corner of the sub-region (2,)
    :param p_2: 2D point defining the bottom right corner of the sub-region (2,)
    :return: average hsv color (3,)
    """
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(tmp[p_1[1] : p_2[1], p_1[0] : p_2[0], :], axis=(0, 1))
