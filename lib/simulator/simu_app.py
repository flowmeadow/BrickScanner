#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Builds a stereo cam setup focusing a brick on a belt
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
from lego.lego_bricks import load_from_id
from lib.data_management import new_stereo_img_dir
from lib.simulator.stereo_app import PoseCam, StereoApp
from overrides import overrides
from pyglet.window import key as gl_key

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FOCUS_POINT = np.array([0.0, 0.1, 0.0])
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
    t += FOCUS_POINT

    # compute rotation matrix
    R = rot_mat((0.0, -1.0, 0.0), 90)
    R = rot_mat((-1.0, 0.0, 0.0), 90) @ R
    R = rot_mat((0.0, 0.0, -1.0), alpha) @ R
    R = rot_mat((0.0, -1.0, 0.0), beta) @ R

    return construct_T(R, t)


def prepare_mesh(mesh):
    """

    :param mesh:
    :return:
    """
    # TODO
    mesh.scale(1 / 10, mesh.get_center())  # scale according to simulator
    mesh.translate(
        np.array(
            [
                -mesh.get_center()[0],  # center on belt
                -np.max(np.array(mesh.vertices)[:, 1]) + FOCUS_POINT[1] - 0.01,  # place close to laser
                -np.min(np.array(mesh.vertices)[:, 2]),  # place on top of belt
            ]
        )
    )
    return mesh


class SimuStereoApp(StereoApp):
    """
    Builds a stereo cam setup focusing a brick on a belt. This is used to take pictures while the
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
        max_images: int = 10,
        travel_dist=0.5,
        **kwargs,
    ):
        """
        :param T_W1: Transformation matrix from world space to camera space 1
        :param T_W2: Transformation matrix from camera space 1 to camera space 2
        :param mesh: Open3D triangle mesh of a brick to scan
        :param image_dir: file_path to store the images
        :param max_images: number of images to generate.
        :param kwargs: forwarded keyword arguments
        """
        super().__init__(max_images=max_images, **kwargs)
        # store attributes

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
        self.dist = travel_dist
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
        for key in self.keys:
            # start simulator
            if key == gl_key.Z:
                self._sim = True
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
            self._path_distance += d
            if self._path_distance > self.dist:
                self._path_distance = 0
                self._sim = False
        else:
            d = self.dist / self._max_images  # shift distance
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


if __name__ == "__main__":
    # --- demo case to my_test class

    # create world to cam transformations
    dist, alpha, beta = 1.2, 2.0, 20.0
    T_W1 = construct_cam_transformation(dist, alpha, beta)
    T_W2 = construct_cam_transformation(dist, -alpha, beta)

    # load and prepare mesh
    mesh: o3d.geometry.TriangleMesh = load_from_id("314")
    mesh.rotate(rot_mat((-1.0, 0.0, 0.0), 90))
    mesh.rotate(rot_mat((0.0, 0.0, 1.0), 30))
    mesh.translate(-mesh.get_center())
    mesh.scale(0.1, mesh.get_center())
    # create and run app
    img_path = new_stereo_img_dir(prefix="laser_test")
    app = SimuStereoApp(
        T_W1,
        T_W2,
        mesh=mesh,
        image_dir=img_path,
        fullscreen=True,
    )
    app.run()
