#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains project constants
@File      : definitions.py
@Project   : BrickScanner
@Time      : 06.03.22 14:52
@Author    : flowmeadow
"""

import os

# paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT_DIR, 'images')
CAM_DIR = os.path.join(ROOT_DIR, 'data/Logitech C270')
SETUP_DIR = os.path.join(ROOT_DIR, 'data/setup')
MODEL_DIR = os.path.join(ROOT_DIR, '../pygletPlayground/rendering/models/model_files')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
BRICK_DIR = os.path.join(ROOT_DIR, '/home/florian/ldraw/parts')
# camera settings
# VIDEO_RES = (1920, 1080)  # (800, 600)  # (320, 240)
# IMAGE_RES = (1920, 1080)
