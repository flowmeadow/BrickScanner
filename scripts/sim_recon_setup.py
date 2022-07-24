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
from lib.helper.cloud_operations import compute_dist_colors, m2c_dist_rough, data2cloud
from lib.helper.lego_bricks import load_stl
from lib.recon.reconstruction import reconstruct_point_cloud
from lib.simulator.cloud_app import CloudApp
from lib.simulator.simu_app import SimuStereoApp, construct_cam_transformation


def prepare_mesh(brick_id: str, z_angle=45.0, random=False, seed: int = None) -> o3d.geometry.TriangleMesh:
    """
    Prepare a brick model for reconstruction
    :param brick_id: brick id
    :return: triangle mesh
    :param z_angle: rotation angle around z axis in degrees
    :param random: if True, angle is generated randomly
    :param seed: set a seed for random number generation (Optional)
    """
    # load mesh
    mesh = load_stl(brick_id)

    # TODO: rotate mesh according to its longest side, so it lies flat on belt
    mesh.rotate(rot_mat((-1.0, 0.0, 0.0), 90))

    # rotate around z-axis
    if seed:
        np.random.seed(seed)
    z_angle = 360.0 * np.random.rand() if random else z_angle
    mesh.rotate(rot_mat((0.0, 0.0, 1.0), z_angle))

    # scale according to simulator dimension
    mesh.scale(1 / 10, mesh.get_center())

    # shift mesh to be centered on belt
    mesh.translate(
        np.array(
            [
                -mesh.get_center()[0],  # center on belt
                -np.max(np.array(mesh.vertices)[:, 1]) - 0.01,  # place close to laser
                -np.min(np.array(mesh.vertices)[:, 2]),  # place on top of belt
            ]
        )
    )
    return mesh


def generate_images(
    folder_name: str,
    mesh: o3d.geometry.TriangleMesh,
    T_W1: np.ndarray = None,
    T_W2: np.ndarray = None,
    automated=False,
    num_images=20,
    travel_dist=0.5,
) -> o3d.geometry.PointCloud:
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


def double_side_recon(
    folder_name,
    mesh,
    automated=False,
    generate_new=True,
    cam_dist=1.2,
    cam_alpha=10.0,
    cam_beta=45.0,
    num_images=20,
    height_offset=0.005,
    gap_window=10,
    y_extension=2.0,
):
    """
    Creates images from two stereo camera setups (4 cameras) and reconstructs the point cloud
    :param folder_name: folder name of the images, load from IMG_DIR
    :param mesh: triangle mesh of model to reconstruct
    :param automated: If true, the image generation is done automatically and the app is closed afterwards
    :param generate_new: to skip image generation, set this to False
    :param cam_dist: distance from cam to focus point (for cam 1, scene 1; other poses are computed respectively)
    :param cam_alpha: rotation angle around z-axis in degree (for cam 1, scene 1; other poses are computed respectively)
    :param cam_beta: rotation angle around y-axis in degree (for cam 1, scene 1; other poses are computed respectively)
    :param num_images: number of images to generate.
    :param height_offset: minimum height (z value) reconstructed points must have to filter out points on the belt
    :param gap_window: defines a window for how many rows to delete around 'gap' rows
    :param y_extension: extents the search area in y direction (in pixel dimension)
    :return:
    """
    folder_1 = f"{folder_name}/view_1"
    folder_2 = f"{folder_name}/view_2"

    y_coords = np.array(mesh.vertices)[:, 1]
    travel_dist = np.max(y_coords) - np.min(y_coords)

    if generate_new:
        # prepare camera poses ...
        # ... for first scene
        T_W1 = construct_cam_transformation(cam_dist, cam_alpha, cam_beta)
        T_W2 = construct_cam_transformation(cam_dist, -cam_alpha, cam_beta)
        poses_1 = (T_W1, T_W2)
        # ... for second scene
        T_W1 = construct_cam_transformation(cam_dist, 180 + cam_alpha, -cam_beta)
        T_W2 = construct_cam_transformation(cam_dist, 180 - cam_alpha, -cam_beta)
        poses_2 = (T_W1, T_W2)

        # generate images
        gen_kwargs = dict(
            num_images=num_images,
            travel_dist=travel_dist,
            automated=automated,
        )
        generate_images(folder_1, mesh, *poses_1, **gen_kwargs)
        generate_images(folder_2, mesh, *poses_2, **gen_kwargs)

    # reconstruct point cloud
    recon_kwargs = dict(
        travel_dist=travel_dist,
        gap_window=gap_window,
        y_extension=y_extension,
    )
    pc_1 = reconstruct_point_cloud(folder_1, **recon_kwargs)
    pc_2 = reconstruct_point_cloud(folder_2, **recon_kwargs)

    # concatenate both point clouds
    pc = data2cloud(np.append(np.array(pc_1.points), np.array(pc_2.points), axis=0))
    return pc


if __name__ == "__main__":
    # TODO: add argparser
    # SETTINGS
    folder_name = "sim_recon"
    brick_id = "314"
    settings = dict(
        automated=True,
        generate_new=True,
        cam_dist=1.2,
        cam_alpha=10.0,
        cam_beta=45.0,
        num_images=50,
        height_offset=0.005,
        gap_window=10,
        y_extension=1.0,
    )

    # generate mesh
    mesh = prepare_mesh(brick_id)

    # (create images and) reconstruct point cloud
    pc = double_side_recon(folder_name, mesh, **settings)
    # save point cloud
    o3d.io.write_point_cloud(f"{DATA_DIR}/{folder_name}/recon_pc.pcd", pc)

    # compute cloud to mesh distance
    dist = m2c_dist_rough(mesh, pc)

    # display reconstructed point cloud
    app = CloudApp(pc.points, compute_dist_colors(dist), mesh, fullscreen=True)
    app.run()
