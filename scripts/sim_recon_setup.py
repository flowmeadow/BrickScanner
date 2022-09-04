#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Script to reconstruct a virtual brick model from images generated with an OpenGL simulator
@File      : sim_recon_setup.py
@Project   : BrickScanner
@Time      : 19.04.22 16:47
@Author    : flowmeadow
"""
import copy
import os
import sys

from lib.recon.epipolar import compute_F

sys.path.append(os.getcwd())  # required to run script from console

import numpy as np
import open3d as o3d
from definitions import *
from glpg_flowmeadow.transformations.methods import rot_mat
from lib.helper.cloud_operations import data2cloud, m2c_dist_rough, display_dist
from lib.helper.lego_bricks import load_stl
from lib.recon.reconstruction import reconstruct_point_cloud
from lib.simulator.simu_app import SimuStereoApp, construct_cam_transformation


def prepare_mesh(
    mesh: o3d.geometry.TriangleMesh, z_angle=45.0, random=False, seed: int = None, scale_factor=1.0
) -> o3d.geometry.TriangleMesh:
    """
    Prepare a brick model for reconstruction
    :param mesh: triangle mesh
    :return: triangle mesh
    :param z_angle: rotation angle around z axis in degrees
    :param random: if True, angle is generated randomly
    :param seed: set a seed for random number generation (Optional)
    :param scale_factor: Scale model according to this
    """
    mesh = copy.deepcopy(mesh)

    # TODO: rotate mesh according to its longest side, so it lies flat on belt
    mesh.rotate(rot_mat((-1.0, 0.0, 0.0), 90))
    # rotate around z-axis
    if seed:
        np.random.seed(seed)
    z_angle = 360.0 * np.random.rand() if random else z_angle
    mesh.rotate(rot_mat((0.0, 0.0, 1.0), z_angle))

    # scale according to simulator dimension
    mesh.scale(scale_factor, mesh.get_center())

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
    data_dir: str,
    img_dir: str,
    mesh: o3d.geometry.TriangleMesh,
    T_W1: np.ndarray = None,
    T_W2: np.ndarray = None,
    automated=False,
    step: float = 0.01,
) -> o3d.geometry.PointCloud:
    """
    Generates images of a virtual brick model
    :param data_dir: directory to store reconstruction data
    :param img_dir: directory to store the stereo image pairs
    :param mesh: triangle mesh
    :param T_W1: pose of camera 1 (4, 4)
    :param T_W2: pose of camera 2 (4, 4)
    :param automated: if True, image generation is running automatically
    :param step: shift in y direction between each frame in simulator dimensions (e.g. 0.01 -> 1mm)
    """
    # create directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
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
        step=step,
        automated=automated,
        fullscreen=True,
    )
    app.run()

    # save camera poses and matrix
    if app.new_images:
        np.save(f"{data_dir}/K_1.npy", app.K)
        np.save(f"{data_dir}/K_2.npy", app.K)
        np.save(f"{data_dir}/T_W1.npy", app.T_W1)
        np.save(f"{data_dir}/T_W2.npy", app.T_W2)
        F = compute_F(app.T_W1, app.T_W2, app.K, app.K)
        np.save(f"{data_dir}/F.npy", F)


def double_side_recon(
    data_dir: str,
    img_dir: str,
    brick_id: str,
    cam_dist=1.2,
    cam_alpha=10.0,
    cam_beta=45.0,
    step=0.01,
    gap_window=10,
    y_extension=2.0,
    automated=False,
    generate_new=True,
    show_result=False,
):
    """
    Creates images from two stereo camera setups (4 cameras) and reconstructs the point cloud
    :param data_dir: directory to store reconstruction data
    :param img_dir: directory to store the stereo image pairs
    :param brick_id: LDraw ID of the LEGO brick to reconstruct
    :param cam_dist: distance from cam to focus point (for cam 1, scene 1; other poses are computed respectively)
    :param cam_alpha: rotation angle around z-axis in degree (for cam 1, scene 1; other poses are computed respectively)
    :param cam_beta: rotation angle around y-axis in degree (for cam 1, scene 1; other poses are computed respectively)
    :param step: shift in y direction between each frame in simulator dimensions (e.g. 0.01 -> 1mm)
    :param gap_window: defines a window for how many rows to delete around 'gap' rows
    :param y_extension: extents the search area in y direction (in pixel dimension)
    :param automated: If true, the image generation is done automatically and the app is closed afterwards
    :param generate_new: to skip image generation, set this to False
    :param show_result: Display the reconstructed point cloud in comparison to the original model
    :return:
    """
    data_dir_1 = f"{data_dir}/view_1"
    data_dir_2 = f"{data_dir}/view_2"
    img_dir_1 = f"{img_dir}/view_1"
    img_dir_2 = f"{img_dir}/view_2"

    # generate mesh
    scale_factor = 0.1
    mesh = load_stl(brick_id)
    mesh = prepare_mesh(mesh, scale_factor=scale_factor)

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
            step=step,
            automated=automated,
        )

        generate_images(data_dir_1, img_dir_1, mesh, *poses_1, **gen_kwargs)
        generate_images(data_dir_2, img_dir_2, mesh, *poses_2, **gen_kwargs)

    # reconstruct point cloud
    recon_kwargs = dict(
        step=step,
        gap_window=gap_window,
        y_extension=y_extension,
        sim=True,
    )
    pc_1 = reconstruct_point_cloud(data_dir_1, img_dir_1, **recon_kwargs)
    pc_2 = reconstruct_point_cloud(data_dir_2, img_dir_2, **recon_kwargs)

    # concatenate both point clouds
    pc = data2cloud(np.append(np.array(pc_1.points), np.array(pc_2.points), axis=0))

    # remove outlier
    pc, _ = pc.remove_radius_outlier(nb_points=2, radius=0.5 * step)

    if show_result:
        dist = m2c_dist_rough(mesh, pc)
        mesh.compute_triangle_normals()
        display_dist(dist, pc, mesh)

    # rescale point cloud back to cm and save point cloud
    pc.scale(1 / scale_factor, pc.get_center())
    o3d.io.write_point_cloud(f"{data_dir}/recon_{brick_id}.pcd", pc)
    return pc


if __name__ == "__main__":
    img_dir = f"{IMG_DIR}/sim_recon"
    data_dir = f"{DATA_DIR}/sim_recon"

    # brick settings
    brick_id = "3148"

    # recon settings
    settings = dict(
        automated=True,
        generate_new=True,
        show_result=True,
        # cam_dist=1.2,
        # cam_alpha=10.0,
        # cam_beta=45.0,
        # gap_window=0,
        # y_extension=1.0,
        # step=0.01,
    )

    # (create images and) reconstruct point cloud
    pc = double_side_recon(data_dir, img_dir, brick_id, **settings)
