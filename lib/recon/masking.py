#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains methods for image masking
@File      : masking.py
@Project   : BrickScanner
@Time      : 29.03.22 17:55
@Author    : flowmeadow
"""
from typing import Optional

import cv2
import numpy as np

from lib.helper.image_operations import rotate_image
from lib.recon.keypoints import find_hcenters


def mask_from_ref(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Computes a binary mask of the brick region using a reference image without brick
    :param img: image with brick
    :param ref: image without brick
    :return: mask
    """
    # prepare mask
    mask = cv2.absdiff(ref, img)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)

    # for k_size in range(1):
    #     kernel = np.ones((k_size * 2 + 1, k_size * 2 + 1), np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def mask_segments(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
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


def red_light_mask(img: np.ndarray, gap_window: int = 10, rotation_angle: Optional[float] = None) -> np.ndarray:
    """
    Generate weighted mask for area of red laser line. Used in simulation
    :param img: BGR image (h, w, 3)
    :param gap_window: defines a window for how many rows to delete around 'gap' rows
    :param rotation_angle: rotate image before masking (Optional)
    :return: grayscale mask (h, w)
    """
    # rotate image
    if rotation_angle:
        img = rotate_image(img, rotation_angle)

    # filter image for red color and generate binary mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, (0, 1, 0), (20, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 1, 0), (180, 255, 255))
    mask = (mask_red1 + mask_red2) / 255

    # subtract a line from binary mask, representing the laser light projection on the belt
    # TODO: currently first and last point is used.
    #       better would be to compute the outer 2d points for 3d points on the belt
    pts = np.round(find_hcenters(mask)).astype(np.uint)
    p_1, p_2 = pts[0, :], pts[-1, :]
    # TODO: might be inaccurate due to integer rounding
    line_mask = cv2.line(np.zeros(mask.shape), p_1, p_2, 255, 3)
    mask = np.clip(cv2.subtract(mask, line_mask), 0, 1.0)

    # for subpixel refinement, a dilation is performed to increase the region of interest
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel)

    # remove rows with gap between white pixels
    pos_mask = mask * np.arange(mask.shape[1])  # give each 1 value its position in x direction
    non_zero_idcs = np.where(pos_mask.any(axis=1))[0]  # get indices of rows that are not all zero
    pos_mask[pos_mask == 0] = np.nan  # replace all zeros with nan
    min_vals = np.nanmin(pos_mask[non_zero_idcs, :], axis=1)  # find the lowest non nan value
    max_vals = np.nanmax(pos_mask[non_zero_idcs, :], axis=1)  # find the highest non nan value
    diff = max_vals - min_vals + 1  # get the difference between min and max
    count = np.count_nonzero(mask[non_zero_idcs, :], axis=1)  # get the number of nonzero values per row
    gap_idcs = non_zero_idcs[np.where(diff != count)[0]]  # with no gap, diff should be equal to count

    # remove not only rows with gap, but also neighboring rows to decrease reconstruction failures
    window_range = np.linspace(-gap_window // 2, gap_window // 2, num=gap_window + 1).astype(int)
    gap_idcs_tmp = np.full((gap_idcs.shape[0], gap_window + 1), np.array([gap_idcs]).T)
    gap_idcs = np.unique((gap_idcs_tmp + window_range).flatten())
    gap_idcs = gap_idcs[gap_idcs >= 0]
    mask[gap_idcs, :] = 0.0  # fill this rows with zero

    # Value is multiplied with the mask to have higher values at the center of the laser line
    hsv = hsv / 255
    mask = mask * hsv[:, :, 2]

    # rotate image back
    if rotation_angle:
        mask = rotate_image(mask, -rotation_angle)

    # cv2.imshow("test", mask)
    # cv2.waitKey()
    return mask


if __name__ == "__main__":
    pass
