#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains methods for camera configuration
@File      : cam_mangement.py
@Project   : BrickScanner
@Time      : 07.03.22 15:42
@Author    : flowmeadow
"""

import cv2


def change_resolutions(cam, resolution):
    print(f"Change camera resolution to {resolution[0]}x{resolution[1]}")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
