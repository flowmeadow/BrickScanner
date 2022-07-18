#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Displays a given pointcloud
@File      : cloud_app.py
@Project   : BrickScanner
@Time      : 21.03.22 23:29
@Author    : flowmeadow
"""
import os
from typing import Optional

import numpy as np
import open3d as o3d
from glpg_flowmeadow.camera.fly_motion import FlyMotion
from glpg_flowmeadow.display.gl_screen import GLScreen
from glpg_flowmeadow.rendering.lighting.lights import Lights
from glpg_flowmeadow.rendering.methods import draw_coordinates, draw_text_2D
from glpg_flowmeadow.rendering.models.model import Model
from glpg_flowmeadow.rendering.models.point_cloud import PointCloud
from pyglet.window import key as gl_key

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


class CloudApp(GLScreen):
    """
    Displays a given pointcloud
    """

    def __init__(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        mesh: Optional[o3d.geometry.TriangleMesh] = None,
        **kwargs,
    ):
        """
        :param points: point cloud points (n, 3)
        :param colors: point cloud colors (n, 3) (Optional)
        :param mesh: Open3D mesh object (Optional)
        :param kwargs: forwarded keyword arguments
        """
        super().__init__(**kwargs)
        # init camera
        self.cam = FlyMotion(self, camera_pos=(1.0, 1.0, 1.0), camera_view=(-1.0, -1.0, -1.0))

        # init light source
        self.lights = Lights()
        self.lights.add(
            position=(1.0, 1.0, 0.0),
            ambient=(1.0, 1.0, 0.5),
            diffuse=(1.0, 1.0, 0.5),
            specular=(1.0, 1.0, 0.5),
        )

        # generate point cloud
        if colors is None:
            self.colors = np.ones(points.shape)
        else:
            self.colors = colors
        self.point_cloud = PointCloud(points, self.colors)

        # generate mesh object
        self.obj = None
        if mesh is not None:
            self.obj = Model(
                vertices=np.array(mesh.vertices),
                indices=np.array(mesh.triangles),
                shader_name="blinn_phong",
                color=(0.0, 0.5, 1.0),
                num_lights=self.lights.num_lights,
            )

        self._button_press = False  # used to detect button press (used for toggling)
        self._display_obj = True  # Toggle between show mesh and don't show mesh

    def handle_events(self) -> None:
        """
        Detect SPACE press and toggles self._display_obj
        :return: None
        """
        if self.obj is not None:
            for key in self.keys:
                if key == gl_key.SPACE:
                    if not self._button_press:
                        self._display_obj = not self._display_obj
                    self._button_press = True
            if len(self.keys) == 0:
                self._button_press = False

    def draw_world(self) -> None:
        """
        draw objects in the world
        :return: None
        """
        draw_coordinates()  # world coordinates

        # move light source to camera center
        self.lights[0].update("position", tuple(self.cam.camera_pos))

        # draw objects
        if self.obj is not None and self._display_obj:
            self.obj.draw()
        self.point_cloud.draw()

    def draw_screen(self) -> None:
        """
        draw objects onto the screen
        :return: None
        """
        draw_text_2D(10, self.height - 10, f"{self.current_fps:.2f}")
