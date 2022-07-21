#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : OpenCV image operations
@File      : image_operations.py
@Project   : BrickScanner
@Time      : 19.07.22 22:35
@Author    : flowmeadow
"""
import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates a given image around an angle
    :param image: image
    :param angle: rotation angle
    :return: image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
