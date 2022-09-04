#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Threaded stereo camera object
@File      : stereo_cam.py
@Project   : BrickScanner
@Time      : 16.03.22 17:27
@Author    : flowmeadow
"""
from typing import Tuple

import numpy as np
from lib.camera.cam import Cam


class StereoCam:
    """
    Description: Threaded stereo camera object
    """

    def __init__(
        self,
        idcs: Tuple[int, int] = (2, 0),
        name="StereoCam",
        *args,
        **kwargs,
    ):
        """
        Initialize Cam
        :param idcs: system ID of the cameras
        :param name: object name
        """

        self._name = name

        # Initialize two camera objects
        print(f"Initialize {self._name}")
        self._cam_1 = Cam(idcs[0], *args, **kwargs)
        self._cam_2 = Cam(idcs[1], *args, **kwargs)
        self._cam_1.start()
        self._cam_2.start()

        self._updated = False
        # wait until a frame from each cam was received
        self._frame_1 = None
        self._frame_2 = None
        while self._frame_1 is None or self._frame_2 is None:  # TODO: Timeout
            self._update()
        print(f"{self._name} is running")

    def _update(self):
        """
        get the latest images from cam objects
        """
        f_1 = self._cam_1.read()
        if f_1 is not None:
            self._frame_1 = f_1
        f_2 = self._cam_2.read()
        if f_2 is not None:
            self._frame_2 = f_2
        self._updated = f_1 is not None and f_2 is not None

    def read(self) -> Tuple[np.array, np.array]:
        """
        return the latest images
        :return: Tuple of numpy arrays
        """
        self._update()
        return self._frame_1.copy(), self._frame_2.copy()

    def updated(self):
        return self._updated

    def __del__(self):
        """
        Stop camera threads and cleanup
        """
        self._cam_1.stop()
        self._cam_2.stop()
