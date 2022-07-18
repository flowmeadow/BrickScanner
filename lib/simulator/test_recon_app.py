#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Test the OpenGL simulator 3D reconstruction using a pointcloud of a sphere
@File      : test_recon_app.py
@Project   : BrickScanner
@Time      : 30.03.22 17:31
@Author    : flowmeadow
"""

from typing import Tuple

import numpy as np
import pyglet.image
from glpg_flowmeadow.camera.fly_motion import FlyMotion
from glpg_flowmeadow.rendering.methods import draw_coordinates, draw_text_2D
from glpg_flowmeadow.rendering.models.point_cloud import PointCloud
from glpg_flowmeadow.transformations.methods import construct_T, rot_mat, rotate_vec
from overrides import overrides
from pyglet.window import key as gl_key

from lib.simulator.stereo_app import StereoApp, PoseCam


def init_cam_transforms(
    dist=3.0,
    axis_1=(0.0, -1.0, 0.0),
    axis_2=(1.0, 0.0, 0.0),
    angle_1=60.0,
    angle_2=60.0,
    gen_rand=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define start positions of both stereo cameras
    :param dist: distance to the world center
    :param axis_1: rotation axis of cam 1
    :param axis_2: rotation axis of cam 2
    :param angle_1: rotation angle of cam 1
    :param angle_2: rotation angle of cam 2
    :param gen_rand: if true, generate random transformations
    :return: two transformation matrices (4, 4)
    """
    if gen_rand:
        axis_1, axis_2 = np.random.randn(3), np.random.randn(3)
        angle_1, angle_2 = 360 * np.random.rand(), 360 * np.random.rand()

    # cam 1
    R_W1 = rot_mat(axis_1, angle_1)
    t_W1 = R_W1 @ np.array([0.0, 0.0, -dist])
    T_W1 = construct_T(R_W1, t_W1)
    # cam 2
    R_W2 = rot_mat(axis_2, angle_2)
    t_W2 = R_W2 @ np.array([0.0, 0.0, -dist])
    T_W2 = construct_T(R_W2, t_W2)
    return T_W1, T_W2


class CirclingCam(PoseCam):
    """Camera class that is controlled with ASDF and is circling around the coordinate center"""

    def __init__(self, **kwargs):
        """
        :param T: transformation matrix to set the cameras pose (4, 4)
        :param name: name of the camera
        """
        super().__init__(**kwargs)

    def update(self, window: pyglet.window.Window) -> None:
        """
        Updates camera position based on keyboard inputs
        :param window: window object the cam is used in
        """
        speed = 0.5
        keys = window.keys
        for key in keys:
            if key in [gl_key.UP, gl_key.W]:  # move up
                axis = np.cross(self.camera_up, self.camera_view)
                self.T = construct_T(rot_mat(axis, speed), np.zeros(3)) @ self.T
            if key in [gl_key.DOWN, gl_key.S]:  # move down
                axis = np.cross(self.camera_up, self.camera_view)
                self.T = construct_T(rot_mat(axis, -speed), np.zeros(3)) @ self.T
            if key in [gl_key.RIGHT, gl_key.D]:  # move left
                axis = self.camera_up
                self.T = construct_T(rot_mat(axis, speed), np.zeros(3)) @ self.T
            if key in [gl_key.LEFT, gl_key.A]:  # move right
                axis = self.camera_up
                self.T = construct_T(rot_mat(axis, -speed), np.zeros(3)) @ self.T
        self.update_from_T(self.T)


class TestReconApp(StereoApp):
    """
    Test the OpenGL simulator 3D reconstruction using a pointcloud of a sphere

    - Move cameras with ASDF
    - Switch cameras with 1 (main), 2 (stereo cam 1) and 3 (stereo cam 2)
    - Save pictures and camera locations with SPACE
    - Quit app with Q
    """

    _frame_count = 0

    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        gen_rand=False,
        T_W1: np.ndarray = None,
        T_W2: np.ndarray = None,
        **kwargs,
    ):

        """
        :param points: point coordinates (m, 3)
        :param colors: point colors (m, 3)
        :param image_dir: directory to store the images
        :param automated: If true, the image generation is done automatically and the app is closed afterwards
        :param gen_rand: generate random camera positions
        :param T_W1: Initial transformation matrix for cam 1 (4, 4)
        :param T_W2: Initial transformation matrix for cam 2 (4, 4)
        :param kwargs: forwarded keyword arguments
        """
        super().__init__(**kwargs)

        # initialize matrices
        if T_W1 is not None and T_W2 is not None:
            self.T_W1, self.T_W2 = T_W1, T_W2
        else:
            self.T_W1, self.T_W2 = init_cam_transforms(gen_rand=gen_rand)

        # initialize cameras
        self.cam_fly = FlyMotion(self, camera_pos=(1.0, 1.0, 1.0), camera_view=(-1.0, -1.0, -1.0))
        self.cam_1 = CirclingCam(T=self.T_W1, name="cam_1")
        self.cam_2 = CirclingCam(T=self.T_W2, name="cam_2")

        self.cam = self.cam_fly  # start with fly cam

        # create point cloud object
        self.point_cloud = PointCloud(points, colors, point_size=10)

    @staticmethod
    def _initialize_transformations() -> Tuple[np.ndarray, np.ndarray]:
        """
        Define start positions of both stereo cameras
        :return: two transformation matrices (4, 4)
        """
        dist = 3.0
        angle = 60.0
        # cam 1
        R_W1 = rot_mat((0.0, -1.0, 0.0), angle)
        t_W1 = np.array([dist, 0.0, 0.0])
        t_W1 = rotate_vec(t_W1, (0.0, 1.0, 0.0), 90 - angle)
        T_W1 = construct_T(R_W1, t_W1)
        # cam 2
        R_W2 = rot_mat((1.0, 0.0, 0.0), angle)
        t_W2 = np.array([0.0, dist, 0.0])
        t_W2 = rotate_vec(t_W2, (-1.0, 0.0, 0.0), 90 - angle)
        T_W2 = construct_T(R_W2, t_W2)
        return T_W1, T_W2

    @overrides
    def handle_events(self) -> None:
        """
        Switch between cams or start image generation
        :return: None
        """
        for key in self.keys:
            if key == gl_key._1:
                self.cam = self.cam_fly
            elif key == gl_key._2:
                self.cam = self.cam_1
            elif key == gl_key._3:
                self.cam = self.cam_2
        super().handle_events()

    @overrides
    def draw_world(self) -> None:
        """
        draw objects in the world
        :return: None
        """
        # draw point cloud
        self.point_cloud.draw()
        super().save_image()  # save current scene

        # draw camera spaces in OpenCV convention
        draw_coordinates(scale=0.1, R=self.cam_1.T[:3, :3], t=self.cam_1.T[:3, 3])
        draw_coordinates(scale=0.1, R=self.cam_2.T[:3, :3], t=self.cam_2.T[:3, 3])
        # draw world coordinates
        draw_coordinates()
        # draw cameras
        self.cam_1.draw()
        self.cam_2.draw()
        self._frame_count += 1

    @overrides
    def draw_screen(self) -> None:
        """
        draw objects onto the screen
        :return: None
        """
        draw_text_2D(10, self.height - 10, f"{self.current_fps:.2f}")
        lines = [
            "- Move cameras with ASDF",
            "- Switch cameras with 1 (main), 2 (stereo cam 1) and 3 (stereo cam 2)",
            "- Save pictures and camera locations with SPACE",
            "- Quit app with Q",
        ]
        for idx, line in enumerate(lines):
            draw_text_2D(10, self.height - 100 - idx * 35, line)
