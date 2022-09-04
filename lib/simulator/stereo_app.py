#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Parent class to handle a stereo cam scene, mainly for image generation
@File      : stereo_app.py
@Project   : BrickScanner
@Time      : 18.07.22 20:21
@Author    : flowmeadow
"""
import shutil

from glpg_flowmeadow.display.gl_screen import GLScreen
from glpg_flowmeadow.camera.camera import Camera
from glpg_flowmeadow.transformations.methods import get_K, get_P
from pyglet.image import get_buffer_manager
import os
from pyglet.window import key as gl_key
import numpy as np


class PoseCam(Camera):
    """ Camera class that gets its pose from a transformation matrix """

    def __init__(self, T: np.ndarray, name="cam"):
        """
        :param T: transformation matrix to set the cameras pose (4, 4)
        :param name: name of the camera
        """
        super().__init__()
        self.T = T
        self.name = name
        self.update_from_T(T)

    def update_from_T(self, T: np.ndarray):
        """
        Update camera pose based on transformation matrix
        :param T: transformation matrix (4, 4)
        """
        self.camera_pos = T[:3, 3]
        self.camera_view = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        self.camera_up = T[:3, :3] @ np.array([0.0, -1.0, 0.0])


class StereoApp(GLScreen):
    """ Parent class to handle a stereo cam scene, mainly for image generation """
    # Camera objects
    cam_1: PoseCam = None  # has to be defined! has to be named 'cam_1'
    cam_2: PoseCam = None  # has to be defined! has to be named 'cam_2'
    cam: Camera = None  # has to be defined!

    K: np.array = None  # camera matrix (overwritten after image saving)
    T_W1: np.array = None  # camera pose 1 (overwritten after image saving)
    T_W2: np.array = None  # camera pose 2 (overwritten after image saving)

    image_count = 0  # current number of saved image pairs
    new_images = False  # is set to True after image generation
    generating_images = False  # is True while image generation is running

    _save_count = 0  # internal count to switch between cases

    def __init__(self, image_dir, max_images=1, automated=False, **kwargs):
        """
        Initialize app
        :param image_dir: directory to store images
        :param max_images: number of image pairs to generate
        :param automated: if True, image generation and app closing is done automatically
        :param kwargs: forwarded keyword arguments
        """
        super().__init__(**kwargs)
        self._image_dir = image_dir
        self._max_images = max_images
        self._automated = automated

    def handle_events(self) -> None:
        """
        start image generation with SPACE or if 'automated is True'
        :return: None
        """
        if gl_key.SPACE in self.keys or self._automated:
            if not self.generating_images:
                print(f"Cleared directory {self._image_dir}")
                shutil.rmtree(self._image_dir)  # delete images
                os.makedirs(self._image_dir)  # create directory
            self.generating_images = True

    def save_image(self):
        """
        Saves the current color buffer if all conditions required are satisfied
        """
        if not self.generating_images:
            return

        def _save_frame(path):
            """saves current color buffer to path"""
            print(f"\rSave image {path} ({self.image_count}|{self._max_images})", end="")
            get_buffer_manager().get_color_buffer().save(path)
            self._save_count += 1

        # define file_path and generate directory if neccessary
        file_name = f"image_{str(self.image_count).zfill(3)}.png"
        file_path = ""
        if self._save_count == 0:  # prepare path for left images
            file_path = f"{self._image_dir}/left"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        elif self._save_count == 1:  # prepare path for right images
            file_path = f"{self._image_dir}/right"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        file_path = f"{file_path}/{file_name}"

        # save color buffer or reset depending on save_count
        if self.cam.name == "cam_1" and self._save_count == 0:
            # save view of cam 1 and switch to cam 2
            _save_frame(file_path)
            self.cam = self.cam_2
        elif self.cam.name == "cam_2" and self._save_count == 1:
            # save view of cam 2 and switch to fly cam
            _save_frame(file_path)
            self.cam = self.cam_fly
        elif self._save_count >= 2:
            # reset save_count and save_state
            self._save_count = 0
            self.image_count += 1
        else:
            self.cam = self.cam_1  # switch to cam 1

        # close app if process is automated
        if self.image_count == self._max_images:
            print("\nFinished image generation")
            # save camera matrix and camera poses
            self.K = get_K(get_P(), self.size)
            self.T_W1, self.T_W2 = self.cam_1.T, self.cam_2.T

            self.new_images = True
            self.image_count = 0
            self.generating_images = False
            if self._automated:
                self.close()
