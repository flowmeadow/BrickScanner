#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : features.py
@Project   : BrickScanner
@Time      : 07.03.22 21:42
@Author    : flowmeadow
"""
import cv2
import numpy as np


def get_point_pairs(img_left, img_right):
    orb = cv2.ORB_create()
    keyPointsLeft, descriptorsLeft = orb.detectAndCompute(img_left, None)
    keyPointsRight, descriptorsRight = orb.detectAndCompute(img_right, None)

    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptorsLeft, descriptorsRight)

    # Apply ratio test
    ptsLeft = []
    ptsRight = []
    for match in matches:
        ptsLeft.append(keyPointsLeft[match.queryIdx].pt)
        ptsRight.append(keyPointsRight[match.trainIdx].pt)

    return np.array(ptsLeft).astype(int), np.array(ptsRight).astype(int)
