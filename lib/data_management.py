#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains methods and functions for data loading and saving
@File      : data_management.py
@Project   : BrickScanner
@Time      : 06.03.22 14:45
@Author    : flowmeadow
"""
import json
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from definitions import *


def params_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"])
    dist = np.array(data["distortion_coefficients"])
    return K, dist


def new_stereo_img_dir(
    dir_name: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    create new image directory for stereo images based on current date and time (if no name is given)
    :param dir_name: name of the directory (optional)
    :param prefix: add a prefix string to directory name (optional)
    :param suffix: add a suffix string to directory name (optional)
    :return: directory
    """
    if dir_name is None:
        dir_name = datetime.now().strftime("%y%m%d-%H%M%S")

    prefix = "" if prefix is None else f"{prefix}_"
    suffix = "" if suffix is None else f"_{suffix}"
    image_path = f"{IMG_DIR}/{prefix}{dir_name}{suffix}"

    dir_left = f"{image_path}/left"
    dir_right = f"{image_path}/right"
    for directory in [image_path, dir_left, dir_right]:
        os.mkdir(directory)
    print(f"Created directory {image_path}")
    return image_path


def append_img_pair(path: str, img_l: np.array, img_r: np.array):
    """
    saves a new image pair in the given directory
    :param path: image directory
    :param img_l: left image frame
    :param img_r: right image frame
    """
    base = "images_"
    ext = "png"

    # define the file name index by checking the other files in the directory
    idx = 0
    for f in os.listdir(f"{path}/left/"):
        f_idx = int(f.split(".")[0][len(base) :])
        if f_idx >= idx:
            idx = f_idx + 1

    # save images
    file_name = f"{base}{str(idx).zfill(3)}.{ext}"
    cv2.imwrite(f"{path}/left/{file_name}", img_l)
    cv2.imwrite(f"{path}/right/{file_name}", img_r)
