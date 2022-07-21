#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Script to reconstruct a virtual brick model from images generated with an OpenGL simulator
@File      : sim_recon_setup.py
@Project   : BrickScanner
@Time      : 19.04.22 16:47
@Author    : flowmeadow
"""
import os
import sys

sys.path.append(os.getcwd())  # required to run script from console

import numpy as np
import open3d as o3d
from definitions import *
from glpg_flowmeadow.transformations.methods import rot_mat
from lib.helper.cloud_operations import compute_dist_colors, m2c_dist_rough
from lib.helper.lego_bricks import load_stl
from lib.recon.reconstruction import reconstruct_point_cloud
from lib.simulator.cloud_app import CloudApp
from lib.simulator.simu_app import SimuStereoApp, construct_cam_transformation, prepare_mesh


def generate_images(
    folder_name: str,
    mesh: o3d.geometry.TriangleMesh,
    T_W1: np.ndarray = None,
    T_W2: np.ndarray = None,
    automated=False,
    num_images=20,
    travel_dist=0.5,
):
    """
    Generates images of a virtual brick model
    :param folder_name: folder name to store images in (in IMG_DIR)
    :param mesh: triangle mesh
    :param T_W1: pose of camera 1 (4, 4)
    :param T_W2: pose of camera 2 (4, 4)
    :param automated: if True, image generation is running automatically
    :param num_images: number of images to render for each cam
    :param travel_dist: the distance the model has to travel in y direction during image generation
    """
    # create directories
    data_dir = f"{DATA_DIR}/{folder_name}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    img_dir = f"{IMG_DIR}/{folder_name}"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # create camera poses, if not given
    if T_W1 is None or T_W2 is None:
        dist, alpha, beta = 1.2, 10.0, 45.0
        T_W1 = construct_cam_transformation(dist, alpha, beta)
        T_W2 = construct_cam_transformation(dist, -alpha, beta)

    # run app with given parameters
    app = SimuStereoApp(
        T_W1,
        T_W2,
        mesh,
        image_dir=img_dir,
        max_images=num_images,
        travel_dist=travel_dist,
        automated=automated,
        fullscreen=True,
    )
    app.run()

    # save camera poses and matrix
    if app.new_images:
        np.save(f"{data_dir}/K.npy", app.K)
        np.save(f"{data_dir}/T_W1.npy", app.T_W1)
        np.save(f"{data_dir}/T_W2.npy", app.T_W2)


if __name__ == "__main__":
    folder_name = "test_laser"
    brick_id = "314"
    mesh = load_stl(brick_id)
    # mesh = load_random_stl()

    mesh.rotate(rot_mat((-1.0, 0.0, 0.0), 90))
    mesh.rotate(rot_mat((0.0, 0.0, 1.0), np.random.rand() * 360))

    # prepare mesh to have correct size and be on top of the belt
    mesh = prepare_mesh(mesh)

    y_coords = np.array(mesh.vertices)[:, 1]
    travel_dist = np.max(y_coords) - np.min(y_coords)
    generate_images(folder_name, mesh, num_images=20, travel_dist=travel_dist)

    pc = reconstruct_point_cloud(folder_name, travel_dist=travel_dist)
    dist = m2c_dist_rough(mesh, pc)

    app = CloudApp(pc.points, compute_dist_colors(dist), mesh, fullscreen=True)
    app.run()
