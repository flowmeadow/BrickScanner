#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains methods for image masking
@File      : masking.py
@Project   : BrickScanner
@Time      : 29.03.22 17:55
@Author    : flowmeadow
"""
import cv2
import numpy as np


def mask_from_ref(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Computes a binary mask of the brick region using a reference image without brick
    :param img: image with brick
    :param ref: image without brick
    :return: mask
    """
    # prepare mask
    mask = cv2.subtract(ref, img)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    for k_size in range(30):
        kernel = np.ones((k_size * 2 + 1, k_size * 2 + 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


if __name__ == "__main__":
    pass
