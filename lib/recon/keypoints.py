#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : methods for finding keypoints and there correspondences
@File      : keypoints.py
@Project   : BrickScanner
@Time      : 19.07.22 00:16
@Author    : flowmeadow
"""
from typing import Tuple

import cv2
import numpy as np


def get_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    :param a1: [x, y] a point on the first line (2,)
    :param a2: [x, y] another point on the first line (2,)
    :param b1: [x, y] a point on the second line (2,)
    :param b2: [x, y] another point on the second line (2,)
    :return 2D intersection point (2,)
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    assert z != 0, "Lines are not allowed to be parallel"
    return np.array([x / z, y / z])


def get_line_points(P: np.ndarray, img_shape: Tuple[int, int], z_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a world to image projection matrix, the upper and lower position of a line along
    the x-axis is returned in image coordinates (for a z_value != 0 this line is shifted in z-direction in world space)
    :param P: camera projection matrix (3, 4)
    :param img_shape: width and height of the image
    :param z_value: shift value of the line in world coordinates
    :return: Two 2D points [(2,), (2,)]
    """
    h, w = img_shape
    l_1 = P @ np.array([10.0, 0.0, z_value, 1.0])
    l_2 = P @ np.array([-10.0, 0.0, z_value, 1.0])
    l_1 = l_1[:-1] / l_1[-1]
    l_2 = l_2[:-1] / l_2[-1]
    p_1 = get_intersect(l_1, l_2, np.array([0, 0]), np.array([w, 0])).astype(np.int)
    p_2 = get_intersect(l_1, l_2, np.array([0, h]), np.array([w, h])).astype(np.int)
    return p_1, p_2


def find_hcenters(mask: np.ndarray) -> np.ndarray:
    """
    Find the horizontal centers for each line in a grayscale image. This is based on the 'center of mass' computation,
    where for each line: center = SUM(x_pos * value) / SUM(value)
    :param mask: grayscale image (w, h)
    :return: center points (for each row) (n, 2)  ; n <= h
    """
    # positions in x direction in pixel dimension ( first pixel in row has position 0, last one has position w - 1)
    x_pos = np.arange(mask.shape[1])

    pixel_pos = x_pos * mask  # multiply with values (mask)
    pixel_counts = np.sum(mask, axis=1)  # some all values in y direction
    y_idcs = np.nonzero(pixel_counts)[0]  # sort out rows with no values
    x_centers = np.sum(pixel_pos[y_idcs], axis=1) / pixel_counts[y_idcs]  # compute centers

    centers = np.array([x_centers, y_idcs]).T
    return centers


def point_from_epiline(lines: np.ndarray, key_pts: np.ndarray, y_extension: float = 2.0) -> np.ndarray:
    """
    find keypoint correspondence from set of epilines. for each line, the closest point above (upper_pt)
    and below (lower_pt) is searched. The corresponding point results from the intersection between the epiline
    and the line from upper_pt to lower_pt. If no point was found, it will be [nan, nan]
    :param lines: set of epilines in form ax + by + c = 0 (n, 3)
    :param key_pts: set of keypoints (m, 2)
    :param y_extension: extents the search area in y direction (in pixel dimension)
    :return: corresponding points array (n, 3)
    """
    lines = lines.copy()
    key_pts = key_pts.copy()

    # get window bounds in x_direction
    min_x, max_x = np.min(key_pts[:, 0]), np.max(key_pts[:, 0])

    key_pts_rel = np.zeros((lines.shape[0], 2))  # empty array for corresponding points
    for idx, line in enumerate(lines):
        # compute end points for epiline
        y_max = -(line[0] * min_x + line[2]) / line[1]
        y_min = -(line[0] * max_x + line[2]) / line[1]
        p_1, p_2 = np.array([min_x, y_max]), np.array([max_x, y_min])

        # compute search area. Only key points within the rectangle from ll to ur are used
        ll = np.array([min_x, y_max - y_extension])  # lower-left
        ur = np.array([max_x, y_min + y_extension])  # upper-right

        # use only keypoints within search area
        in_idx = np.all(np.logical_and(ll <= key_pts, ur >= key_pts), axis=1)
        in_pts = key_pts[in_idx, :]
        if in_pts.shape[0] < 2:  # if no point in search area
            key_pts_rel[idx] = np.array([np.nan, np.nan])  # point not found
            continue

        # compute closest lower and upper point to epiline
        line_dists = np.cross(p_2 - p_1, in_pts - p_1) / np.linalg.norm(p_2 - p_1)
        neg_dists = np.where(line_dists < 0, line_dists, np.nan)  # replace positive values with nan
        pos_dists = np.where(line_dists > 0, line_dists, np.nan)  # replace negative values with nan
        if np.all(np.isnan(neg_dists)) or np.all(np.isnan(pos_dists)):  # check if a point on both sides exist
            key_pts_rel[idx] = np.array([np.nan, np.nan])  # point not found
            continue
        lower_pt = in_pts[neg_dists.argmax(), :]  # get upper pt
        upper_pt = in_pts[pos_dists.argmin(), :]

        # compute intersection between line from lower to upper point and the epiline
        key_pts_rel[idx] = get_intersect(p_1, p_2, lower_pt, upper_pt)

    return key_pts_rel


def find_keypoint_pairs(
    kpts_1: np.ndarray,
    kpts_2: np.ndarray,
    F: np.ndarray,
    y_extension: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Indices notate the image the points appear in. For example, ppts_2 are the corresponding points in
    image 2 for the keypoints kpts_1 from image 1
    :param kpts_1: keypoints of image 1 (n, 2)
    :param kpts_2: keypoints of image 2 (m, 2)
    :param F: fundamental matrix (3, 3)
    :param y_extension: extents the search area in y direction (in pixel dimension)
    :return: point set for image 1 (k, 2) and image 2 (k, 2) with k <= n + m
    """
    # compute epilines for each keypoint. keypoints of image 1 generate epilines for image 2 and vice versa
    lines_1, lines_2 = [cv2.computeCorrespondEpilines(kpts, i + 1, F) for i, kpts in enumerate([kpts_1, kpts_2])]
    lines_1, lines_2 = np.squeeze(lines_1), np.squeeze(lines_2)

    # compute paired points for each keypoint set
    ppts_1 = point_from_epiline(lines_2, kpts_1, y_extension=y_extension)
    ppts_2 = point_from_epiline(lines_1, kpts_2, y_extension=y_extension)

    # remove points no paired point was found for
    kpts_1 = kpts_1[~np.isnan(ppts_2[:, 0])]
    ppts_2 = ppts_2[~np.isnan(ppts_2[:, 0])]
    kpts_2 = kpts_2[~np.isnan(ppts_1[:, 0])]
    ppts_1 = ppts_1[~np.isnan(ppts_1[:, 0])]

    # build a new set of points for image 1 and 2
    pts_1 = np.append(kpts_1, ppts_1, axis=0)
    pts_2 = np.append(ppts_2, kpts_2, axis=0)

    return pts_1, pts_2
