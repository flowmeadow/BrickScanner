#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains image point description methods
@File      : description.py
@Project   : BrickScanner
@Time      : 29.03.22 17:55
@Author    : flowmeadow
"""
from typing import Optional

import cv2
import numpy as np


def get_descriptors(img: np.ndarray, mask: np.ndarray, descriptor: Optional[object] = None) -> np.ndarray:
    """

    :param img: image to compute descriptors for
    :param mask: image mask defining the region where descriptors are computed
    :param descriptor: descriptor object to use (optional)
    :return: list of descriptors
    """
    # Initiate ORB detector if none is given
    descriptor = cv2.ORB_create() if descriptor is None else descriptor

    # find keypoints in image mask
    kp = np.argwhere(mask > 0)
    kp = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in kp]

    # compute the descriptors with ORB
    des = descriptor.compute(img, kp)
    return des


if __name__ == "__main__":
    pass
