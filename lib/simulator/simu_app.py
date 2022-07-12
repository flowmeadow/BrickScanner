#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Builds a stereo cam setup focusing a brick on a belt
@File      : simu_app.py
@Project   : BrickScanner
@Time      : 19.04.22 16:49
@Author    : flowmeadow
"""

import shutil

import numpy as np
import open3d as o3d
from definitions import *
from glpg_flowmeadow.camera.camera import Camera
from glpg_flowmeadow.camera.fly_motion import FlyMotion
from glpg_flowmeadow.display.gl_screen import GLScreen
from glpg_flowmeadow.rendering.lighting.lights import Lights
from glpg_flowmeadow.rendering.methods import draw_coordinates, draw_text_2D
from glpg_flowmeadow.rendering.models.model import Model
from glpg_flowmeadow.transformations.methods import construct_T, get_K, get_P, rot_mat, rotate_vec
from lego.lego_bricks import load_from_id
from overrides import overrides
from pyglet.image import get_buffer_manager
from pyglet.window import key as gl_key
from lib.data_management import new_stereo_img_dir

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FOCUS_POINT = np.array([0.0, 0.1, 0.0])
LASER_SHADER = "./shader/laser_line"


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
    R = rot_mat((0.0, 1.0, 0.0), 90)
    R = rot_mat((0.0, 0.0, -1.0), alpha) @ R
    R = rot_mat((0.0, -1.0, 0.0), beta) @ R

    return construct_T(R, t)


class SimuStereoApp(GLScreen):
    """
    Builds a stereo cam setup focusing a brick on a belt. This is used to take pictures while the
    brick crosses a laser line on the belt. Unit length 1 represents 100mm or 10cm
          - Move the camera with ASDF and Mouse
          - Switch between cameras with 1, 2 and 3
          - Run belt movement with Z
          - Start image generation with SPACE
    """

    dist = 0.5  # distance in which images are taken

    def __init__(
        self,
        T_W1: np.ndarray,
        T_12: np.ndarray,
        mesh: o3d.geometry.TriangleMesh,
        file_path: str = "",
        max_images: int = 10,
        **kwargs,
    ):
        """
        :param T_W1: Transformation matrix from world space to camera space 1
        :param T_12: Transformation matrix from camera space 1 to camera space 2
        :param mesh: Open3D triangle mesh of a brick to scan
        :param file_path: file_path to store the images
        :param max_images: number of images to generate.
        :param kwargs: forwarded keyword arguments
        """
        super().__init__(**kwargs)

        # store attributes
        self.file_path = file_path

        # initialize cameras
        self.cam_fly = FlyMotion(self, camera_pos=(1.0, 1.0, 1.0), camera_view=(-1.0, -1.0, -1.0))
        self.cam_1 = Camera(camera_pos=T_W1[:3, 3], camera_view=T_W1[:3, :3] @ np.array([0.0, 0.0, -1.0]), name="cam_1")
        self.cam_2 = Camera(camera_pos=T_W2[:3, 3], camera_view=T_W2[:3, :3] @ np.array([0.0, 0.0, -1.0]), name="cam_2")

        # define initial camera
        self.cam = self.cam_fly  # start with fly cam

        # add lighting
        self.lights = Lights()
        self.lights.add(
            position=(1.0, 1.0, 0.0),
            ambient=(1.0, 1.0, 0.5),
            diffuse=(1.0, 1.0, 0.5),
            specular=(1.0, 1.0, 0.5),
        )

        # add brick
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
        self.frame_count = 0  # counts all frames that have been drawn
        self.save_count = 0  # counts the number of saved images per scene (max: 2 for stereo setup)
        self.img_count = 0  # counts the number of image (pairs) saved

        # Bools
        self.sim = False  # shows simulation preview (start with Z key)
        self.prepare_next_frame = False  # If True, shifts brick one step further
        self.save_frame = False  # If True, current color buffer will be saved

        # Others
        self.path_distance = 0.0  # shift of brick in y direction
        self.max_images = max_images  # number of images to take

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
            shader_name=LASER_SHADER,
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
            # start image generation
            if key == gl_key.SPACE:
                if not self.prepare_next_frame:
                    shutil.rmtree(self.file_path)  # delete images
                    os.makedirs(self.file_path)  # create directory
                self.prepare_next_frame = True

            # start simulator
            if key == gl_key.Z:
                self.sim = True

            # switch between cameras
            elif key == gl_key._1:
                self.cam = self.cam_fly
            elif key == gl_key._2:
                self.cam = self.cam_1
            elif key == gl_key._3:
                self.cam = self.cam_2

    @overrides
    def draw_world(self) -> None:
        """
        draw objects in the world
        :return: None
        """

        # draw reference objects only if no images are taken
        if not self.save_frame:
            draw_coordinates()
            self.lights[0].update("position", tuple(self.cam_1.camera_pos))
            self.lights.draw()
            self.cam_1.draw()
            self.cam_2.draw()

        if self.prepare_next_frame:
            # shift brick to generate next images
            d = self.dist / self.max_images  # shift distance
            self.path_distance += d
            self.save_frame = True
            self.prepare_next_frame = False
        elif self.sim:
            d = 0.001
            self.path_distance += d
            if self.path_distance > self.dist:
                self.path_distance = 0
                self.sim = False

        # draw objects
        self.brick.translate(*(0.0, self.path_distance, 0.0)) # translate brick
        self.brick.draw()
        self.belt.draw()

        # save images
        if self.save_frame:
            self.save_current_image()

        self.frame_count += 1

    @overrides
    def draw_screen(self) -> None:
        """
        draw objects onto the screen
        :return: None
        """
        draw_text_2D(10, self.height - 10, f"{self.current_fps:.2f}")  # draw FPS

    def save_current_image(self):
        """
        Saves the current color buffer if all conditions required are satisfied
        """

        def _save_frame(path):
            """saves current color buffer to path"""
            print(f"Save image {path}")
            get_buffer_manager().get_color_buffer().save(path)
            self.save_count += 1

        # define file_path and generate directory if neccessary
        file_name = f"image_{str(self.img_count).zfill(3)}.png"
        file_path = ""
        if self.save_count == 0:  # prepare path for left images
            file_path = f"{self.file_path}/left"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        elif self.save_count == 1:  # prepare path for right images
            file_path = f"{self.file_path}/right"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        file_path = f"{file_path}/{file_name}"

        # save color buffer or reset depending on save_count
        if self.cam.name == "cam_1" and self.save_count == 0:
            # save view of cam 1 and switch to cam 2
            _save_frame(file_path)
            self.cam = self.cam_2
        elif self.cam.name == "cam_2" and self.save_count == 1:
            # save view of cam 2 and switch to fly cam
            _save_frame(file_path)
            self.cam = self.cam_fly
        elif self.save_count >= 2:
            # reset save_count and save_state to translate brick for next images or stop the process
            self.save_count = 0
            self.save_frame = False
            self.img_count += 1

            if self.img_count < self.max_images:  # prepare next frame
                self.prepare_next_frame = True
            else:  # stop image saving process
                self.img_count = 0
                self.path_distance = 0
                # save camera matrix
                np.save(f"{CALIB_DIR}/sim/K.npy", get_K(get_P(), self.size))

        else:
            self.cam = self.cam_1  # switch to cam 1


if __name__ == "__main__":
    # --- demo case to test class

    # create world to cam transformations
    dist, alpha, beta = 1.2, 2.0, 20.0
    T_W1 = construct_cam_transformation(dist, alpha, beta)
    T_W2 = construct_cam_transformation(dist, -alpha, beta)

    # load and prepare mesh
    mesh: o3d.geometry.TriangleMesh = load_from_id("314")
    mesh.rotate(rot_mat((-1.0, 0.0, 0.0), 90))
    mesh.rotate(rot_mat((0.0, 0.0, 1.0), 30))

    # create and run app
    img_path = new_stereo_img_dir(prefix="laser_test")
    app = SimuStereoApp(
        T_W1,
        T_W2,
        mesh=mesh,
        file_path=img_path,
        fullscreen=True,
    )
    app.run()
