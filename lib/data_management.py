#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : data_management.py
@Project   : BrickScanner
@Time      : 06.03.22 14:45
@Author    : flowmeadow
"""
import json

import numpy as np


def params_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"])
    dist = np.array(data["distortion_coefficients"])
    return K, dist
