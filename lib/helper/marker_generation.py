#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : marker_generation.py
@Project   : BrickScanner
@Time      : 29.05.22 16:02
@Author    : flowmeadow
"""
import cv2
import numpy as np
import pandas as pd
from cv2 import aruco
from definitions import *


from lib.colors.base import hex2rgb


def direction_checker():
    patch_size = 200
    size = 800
    h = w = size
    canvas = np.full((h, w, 3), 1.0).astype(np.float32)  # white canvas

    # load aruco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # draw markers
    for idx in range(9):
        marker = aruco.drawMarker(aruco_dict, idx, patch_size)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB) / 255

        x = (idx % 3) * (size // 2 - patch_size // 2)
        y = (idx // 3) * (size // 2 - patch_size // 2)
        print(x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (
            x + patch_size + 10 - (patch_size + 40) * (idx % 3) // 2,
            y - 10 if idx >= 3 else y + patch_size + 30,
        )
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (0, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        cv2.putText(canvas, str(idx), org, font, fontScale, color, thickness, cv2.LINE_AA)

        canvas[y : y + patch_size, x : x + patch_size] = marker

    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", canvas)
    cv2.waitKey()


def color_checker():

    patch_size = 100
    gap = 10
    num_rows, num_cols = 11, 17

    w = (num_cols + 2) * patch_size + (num_cols + 1) * gap
    h = (num_rows + 2) * patch_size + (num_rows + 1) * gap
    canvas = np.full((h, w, 3), 1.0).astype(np.float32)  # white canvas

    color_df = pd.read_csv(f"{DATA_DIR}/colors.csv")
    color_df = color_df[color_df["is_trans"] == "f"]  # only not transparent colors
    color_df = color_df[1:-1]  # remove last and first item
    print(color_df)
    labels = np.array(color_df["name"])
    colors = np.array(color_df["rgb"])
    colors = np.array(list(map(hex2rgb, colors))) / 255

    if len(colors) > num_rows * num_cols:
        raise ValueError("Fewer patches than colors")

    # load aruco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # draw markers
    for idx in range(4):
        marker = aruco.drawMarker(aruco_dict, idx, patch_size)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB) / 255
        x = (idx % 2) * (w - patch_size)
        y = (idx // 2) * (h - patch_size)
        canvas[y : y + patch_size, x : x + patch_size] = marker

        # draw black background
        canvas[patch_size:-patch_size, patch_size:-patch_size] = (0.0, 0.0, 0.0)

    for idx, c in enumerate(colors):
        x = (idx % num_cols + 1) * (gap + patch_size)
        y = (idx // num_cols + 1) * (gap + patch_size)
        canvas[y : y + patch_size, x : x + patch_size] = c

    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", canvas)
    cv2.waitKey()


if __name__ == "__main__":
    direction_checker()
