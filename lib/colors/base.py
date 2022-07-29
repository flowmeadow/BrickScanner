#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : base.py
@Project   : BrickScanner
@Time      : 30.05.22 22:39
@Author    : flowmeadow
"""

import numpy as np
import pandas as pd
import skimage.color

def hex2rgb(hex_str):
    n = 2
    str_lst = [hex_str[i : i + n] for i in range(0, len(hex_str), n)]
    return np.array([int(s, 16) for s in str_lst])


def colors_from_csv(path):
    # use original RGB values from data base
    color_df = pd.read_csv(path)
    color_df = color_df[color_df["is_trans"] == "f"]  # only not transparent colors
    color_df = color_df[1:-1]  # remove last and first item
    labels = np.array(color_df["name"])
    colors = np.array(color_df["rgb"])
    colors = np.array(list(map(hex2rgb, colors))).astype(np.float64)
    return colors, labels


def dist_LAB(color_1, color_2):
    """
    https://en.wikipedia.org/wiki/Color_difference
    :param color_1:
    :param color_2:
    :return:
    """
    color_1 = skimage.color.rgb2lab(color_1 / 255)
    color_2 = skimage.color.rgb2lab(color_2 / 255)
    # CIE76
    dist = np.sqrt(np.sum((color_1 - color_2) ** 2, axis=-1))
    return dist


def dist_RGB(color_1, color_2):
    """
    https://en.wikipedia.org/wiki/Color_difference
    :param color_1:
    :param color_2:
    :return:
    """
    color_1 = skimage.color.rgb2xyz(color_1 / 255)
    color_2 = skimage.color.rgb2xyz(color_2 / 255)
    # sRGB
    dist = np.sqrt(np.sum((color_1 - color_2) ** 2, axis=-1))
    return dist



