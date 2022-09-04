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


def get_sub_region(img: np.ndarray, p_1: np.ndarray, p_2: np.ndarray) -> np.ndarray:
    """
    Returns an image cutout of a given source image
    :param img: source image (h, w, 3)
    :param p_1: 2D point defining the top left corner of the sub-region (2,)
    :param p_2: 2D point defining the bottom right corner of the sub-region (2,)
    :return: image cutout (y_max - y_min, x_max - x_min, 3)
    """
    arr = np.array([p_1, p_2])
    x_min, x_max = np.min(arr[:, 0]), np.max(arr[:, 0])
    y_min, y_max = np.min(arr[:, 1]), np.max(arr[:, 1])
    return img[y_min:y_max, x_min:x_max, :]


def fill_sub_region(img: np.ndarray, sub: np.ndarray, p_1: np.ndarray, p_2: np.ndarray) -> np.ndarray:
    """
    Overwrites a subregion in a source image
    :param img: source image (h, w, 3)
    :param sub: image data to fill the subregion with (y_max - y_min, x_max - x_min, 3)
    :param p_1: 2D point defining the top left corner of the sub-region (2,)
    :param p_2: 2D point defining the bottom right corner of the sub-region (2,)
    :return: updated source image (h, w, 3)
    """
    arr = np.array([p_1, p_2])
    x_min, x_max = np.min(arr[:, 0]), np.max(arr[:, 0])
    y_min, y_max = np.min(arr[:, 1]), np.max(arr[:, 1])
    img[y_min:y_max, x_min:x_max, :] = sub
    return img
