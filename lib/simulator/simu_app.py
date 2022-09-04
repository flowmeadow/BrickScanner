#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Builds a stereo cam real_setup focusing a brick on a belt
@File      : simu_app.py
@Project   : BrickScanner
@Time      : 19.04.22 16:49
@Author    : flowmeadow
"""


import numpy as np
import open3d as o3d
from definitions import *
from glpg_flowmeadow.camera.fly_motion import FlyMotion
from glpg_flowmeadow.rendering.lighting.lights import Lights
from glpg_flowmeadow.rendering.methods import draw_coordinates, draw_text_2D
from glpg_flowmeadow.rendering.models.model import Model
from glpg_flowmeadow.transformations.methods import construct_T, rot_mat, rotate_vec
from lib.simulator.stereo_app import PoseCam, StereoApp
from overrides import overrides
from pyglet.window import key as gl_key

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
LASER_SHADER = f"{CURRENT_PATH}/shader/laser_line"


def construct_cam_transformation(distance: float = 1.0, alpha: float = 5.0, beta: float = 20.0) -> np.ndarray:
    """
    Constructs a camera transformation matrix such that the cam focuses the focus point (laser position).
    :param distance: distance from cam to focus point
    :param alpha: rotation angle around z-axis in degree
    :param beta: rotation angle around y-axis in degree
    :return: transformation matrix (4, 4)
    """
    # compute translation vector
    t = np.array([distance, 0.0, 0.0])
    t = rotate_vec(t, (0.0, 0.0, -1.0), alpha)
    t = rotate_vec(t, (0.0, -1.0, 0.0), beta)

    # compute rotation matrix
    R = rot_mat((0.0, -1.0, 0.0), 90)
    R = rot_mat((-1.0, 0.0, 0.0), 90) @ R
    R = rot_mat((0.0, 0.0, -1.0), alpha) @ R
    R = rot_mat((0.0, -1.0, 0.0), beta) @ R

    return construct_T(R, t)


class SimuStereoApp(StereoApp):
    """
    Builds a stereo cam real_setup focusing a brick on a belt. This is used to take pictures while the
    brick crosses a laser line on the belt. Unit length 1 represents 100mm or 10cm
          - Move the camera with ASDF and Mouse
          - Switch between cameras with 1, 2 and 3
          - Run belt movement with Z
          - Start image generation with SPACE
    """

    _sim = False  # shows simulation preview (start with Z key)
    _frame_count = 0  # counts all frames that have been drawn
    _path_distance = 0.0  # shift of brick in y direction

    def __init__(
        self,
        T_W1: np.ndarray,
        T_W2: np.ndarray,
        mesh: o3d.geometry.TriangleMesh,
        step: float = 0.01,
        **kwargs,
    ):
        """
        Initialize app
        :param T_W1: Transformation matrix from world space to camera space 1
        :param T_W2: Transformation matrix from camera space 1 to camera space 2
        :param mesh: Open3D triangle mesh of a brick to scan
        :param step: shift in y direction between each frame in simulator dimensions (e.g. 0.01 -> 1mm)
        :param kwargs: forwarded keyword arguments
        """

        # compute number of required images
        self.step = step
        y_coords = np.array(mesh.vertices)[:, 1]
        self.y_dist = np.max(y_coords) - np.min(y_coords)
        self.y_dist += 2 * self.step
        max_images = int(self.y_dist / self.step)
        super().__init__(max_images=max_images, **kwargs)

        # initialize cameras
        self.cam_fly = FlyMotion(self, camera_pos=(1.0, 1.0, 1.0), camera_view=(-1.0, -1.0, -1.0))
        self.cam_1 = PoseCam(T=T_W1, name="cam_1")
        self.cam_2 = PoseCam(T=T_W2, name="cam_2")

        # define initial camera
        self.cam = self.cam_fly  # start with fly cam

        # add lighting
        self.lights = Lights()
        self.lights.add(
            position=(self.cam_1.camera_pos + self.cam_2.camera_pos) / 2,
            ambient=(1.0, 1.0, 0.5),
            diffuse=(1.0, 1.0, 0.5),
            specular=(1.0, 1.0, 0.5),
        )

        # add brick
        self.brick = Model(
            vertices=np.array(mesh.vertices),
            indices=np.array(mesh.triangles),
            shader_name=LASER_SHADER,
            color=(0.0, 0.5, 1.0),
            num_lights=self.lights.num_lights,
        )

        # add belt
        self.belt = self.generate_belt()

        # Counters

    def generate_belt(self, width: float = 1.0, length: float = 3.0) -> Model:
        """
        Generates a rectangle representing the belt
        :param width: width of the belt in x direction
        :param length: length of the belt in y direction
        :return: Model
        """
        vertices = np.array([[0.0, 0.0, 0.0], [width, 0.0, 0.0], [width, length, 0.0], [0.0, length, 0.0]])
        vertices -= np.array([width / 2, length / 2, 0.0])  # shift vertices
        indices = np.array([[0, 1, 2], [2, 3, 0]])
        model = Model(
            vertices=vertices,
            indices=indices,
            shader_name=LASER_SHADER,  # "blinn_phong",  # LASER_SHADER,
            color=(0.3, 0.3, 0.7),
            num_lights=self.lights.num_lights,
        )
        return model

    @overrides
    def handle_events(self) -> None:
        """
        handle keyboard inputs
        :return: None
        """
        # if self._sim:
        #     c_pos = self.cam_fly.camera_pos
        #     c_view = self.cam_fly.camera_view
        #     self.cam.camera_pos = rot_mat((0, 0, 1), 0.1) @ c_pos
        #     self.cam_fly.camera_view = rot_mat((0, 0, 1), 0.1) @ c_view

        for key in self.keys:
            # start simulator
            if key == gl_key.Z:
                self._sim = True
            # stop simulator
            elif key == gl_key.T:
                self._sim = False
                self._path_distance = 0
            # switch between cameras
            elif key == gl_key._1:
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
        if self._sim:
            d = 0.001
            if self._path_distance > self.y_dist:
                self._path_distance = 0
            else:
                self._path_distance += d

        else:
            d = self.step  # shift distance
            self._path_distance = d * self.image_count
        self.brick.translate(*(0.0, self._path_distance, 0.0))  # translate brick

        # draw objects
        self.brick.draw()
        self.belt.draw()
        super().save_image()

        draw_coordinates()
        self.lights.draw()
        self.cam_1.draw()
        self.cam_2.draw()

        self._frame_count += 1

    @overrides
    def draw_screen(self) -> None:
        """
        draw objects onto the screen
        :return: None
        """
        draw_text_2D(10, self.height - 10, f"{self.current_fps:.2f}")  # draw FPS
