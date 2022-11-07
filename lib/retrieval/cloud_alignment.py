#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains methods for point cloud alignment and model search
@File      : cloud_alignment.py
@Project   : BrickScanner
@Time      : 20.06.22 17:48
@Author    : flowmeadow
"""

import copy
import pickle
from typing import List, Optional, Tuple
import matplotlib.colors as mcolors
import numpy as np
import open3d as o3d
from definitions import *
from glpg_flowmeadow.transformations.methods import rot_mat
from lib.helper.cloud_operations import cloud2cloud_err, construct_T, draw_point_clouds, rotate_random, \
    compute_dist_colors
from lib.retrieval.bounding_box import (
    PCA_based_alignment,
    compute_obb_center,
    compute_obb_edges,
    find_closest_obb_edges,
)


def unique_rotations() -> List[np.ndarray]:
    """
    Returns a list of unique rotation matrices that perform rotations around the x-, y- and z-axis
    for 0°, 90°, 180° and 270°.
    :return: 24 entry list of rotation matrices (3, 3)
    """
    mats = []
    for a in [0, 90, 180, 270]:
        for b in [0, 90, 180, 270]:
            for c in [0, 270]:
                R_1 = rot_mat((1, 0, 0), a)
                R_2 = rot_mat((0, 1, 0), b)
                R_3 = rot_mat((0, 0, 1), c)
                mats.append((R_1 @ R_2 @ R_3).astype(int))
    mats = np.unique(np.array(mats), axis=0)
    return mats


def prepare_cloud(
    pc: o3d.geometry.PointCloud,
    random: bool = False,
    pca_method: str = "all",
) -> o3d.geometry.PointCloud:
    """
    prepares source point cloud for cloud alignment
    :param pc: source point cloud
    :param random: if True, the cloud is rotated randomly first
    :param pca_method: PCA method used for alignment ("all", "min", "max")
    :return: source point cloud
    """
    if random:
        pc = rotate_random(pc)
    # align point cloud with world axes
    pc = PCA_based_alignment(pc, method=pca_method)
    # move obb center to origin
    pc.translate(-compute_obb_center(pc))
    # estimate normals
    pc.estimate_normals()
    return pc


def align_point_clouds(
    pc_source: o3d.geometry.PointCloud,
    pc_target: o3d.geometry.PointCloud,
    debug: bool = False,
    pca_method: str = "all",
    initial_icp: bool = False,
) -> np.ndarray:
    """
    Aligns the target point_cloud with source point cloud
    IMPORTANT: pc_source needs to be PCA aligned and have its obb center in world center
    :param pc_source: source point cloud
    :param pc_target: target point cloud
    :param debug: show alignment pipeline steps
    :param pca_method: PCA method used for alignment ("all", "min", "max")
    :param initial_icp: perform initial icp optimization
    :return: transformation matrix for target to align it with source (4, 4)
    """
    pc_source = copy.deepcopy(pc_source)
    pc_target = copy.deepcopy(pc_target)
    T_target = np.eye(4)

    # ICP parameters
    threshold = 1.0
    trans_init = np.eye(4)

    # estimate normals for ICP
    pc_target.estimate_normals()
    pc_source.estimate_normals()

    if debug:
        print("\nDEBUG: Initial orientation")
        draw_point_clouds(pc_source, pc_target, coord_axes=False)

    # get orientation by PCA based alignment
    # TODO: This is different from:
    #       pc_target, R_init = PCA_based_alignment(pc_target, get_R=True)
    #       Why the heck???
    _, R_init = PCA_based_alignment(pc_target, get_R=True, method=pca_method)
    pc_target = pc_target.rotate(R_init, center=np.zeros(3))

    T_target = construct_T(R=R_init) @ T_target
    if debug:
        print("DEBUG: PCA based alignment")
        draw_point_clouds(pc_source, pc_target, coord_axes=False)

    # move it to world center
    t = -compute_obb_center(pc_target)
    T_target = construct_T(t=t) @ T_target
    pc_target = pc_target.translate(t)
    if debug:
        print("DEBUG: shift OBB center to origin")
        draw_point_clouds(pc_source, pc_target, coord_axes=False)

    # perform initial ICP
    if initial_icp:
        reg_p2l = o3d.pipelines.registration.registration_icp(
            pc_source,
            pc_target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        T_icp = reg_p2l.transformation
        T_icp = np.linalg.inv(T_icp)
        T_target = T_icp @ T_target
        pc_target.transform(T_icp)
        if debug:
            print("DEBUG: initial ICP")
            draw_point_clouds(pc_source, pc_target, coord_axes=False)

    # rotate around several axes to find best alignment
    min_err = cloud2cloud_err(pc_source, pc_target)  # current error
    R_min = np.eye(3)  # current rotation
    for R in unique_rotations()[1:, :, :]:
        # rotate point cloud
        pc_target.rotate(R, center=np.zeros(3))

        # compute new error
        err = cloud2cloud_err(pc_source, pc_target, method=np.mean)
        if err < min_err:  # did it get better?
            min_err = err
            R_min = R
        pc_target.rotate(R.T, center=np.zeros(3))  # rotate back

    # take optimal rotation and apply it
    pc_target.rotate(R_min, center=np.zeros(3))
    T_target = construct_T(R=R_min) @ T_target
    if debug:
        print("DEBUG: Optimized alignment")
        draw_point_clouds(pc_source, pc_target, coord_axes=False)

    # final optimization using ICP
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pc_source,
        pc_target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    T_icp = reg_p2l.transformation
    T_icp = np.linalg.inv(T_icp)
    T_target = T_icp @ T_target
    if debug:
        print("DEBUG: ICP alignment")
        pc_target.transform(T_icp)
        draw_point_clouds(pc_source, pc_target, coord_axes=False)
    return T_target


def find_model(
    pc_source: o3d.geometry.PointCloud,
    debug_file: str = None,
    threshold: float = 0.1,
    max_best: Optional[int] = None,
    num_samples: int = 5_000,
    num_tries: int = 3,
) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
    """
    Given a source point cloud, search for a model that matches from the Ldraw library
    :param pc_source: source point cloud
    :param debug_file: file_name of the correct model (Optional)
    :param threshold: maximum allowed error for PCA OBB comparison
    :param max_best: maximum amount of files used from preselection
    :param num_samples: number of samples used for alignment
    :param num_tries: number of tries to obtain the best alignment
    :return: [used files, alignment errors, list of target transformations (4, 4)]
    """

    # load PCA aligned obb edges
    with open(f"{DATA_DIR}/base_obb_data.pkl", "rb") as f:
        obb_data = pickle.load(f)

    # preselect models that have a PCA aligned OBB similar to source point cloud
    idcs = find_closest_obb_edges(compute_obb_edges(pc_source), obb_data["edges"], thresh=threshold, max_best=max_best)
    files = np.array(obb_data["files"])[idcs]
    print(f"Found {len(files)} files that have an obb edge error lower than {threshold}")
    if debug_file:
        print(f"Original file {debug_file} is {'' if debug_file in files else 'NOT '}in file selection")

    # align point clouds for each file
    errors = []
    transformations = []
    for file in files:
        # convert model to point cloud
        mesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{file}")
        pc_control: o3d.geometry.PointCloud = mesh.sample_points_uniformly(100_000)

        # find the optimal alignment within the given number of tries
        min_err = np.inf
        T_final = None
        for try_idx in range(num_tries):
            pc_target: o3d.geometry.PointCloud = mesh.sample_points_uniformly(num_samples)

            # compute best alignment transformation
            T_target = align_point_clouds(pc_source, pc_target, debug=(file == debug_file and try_idx == 0))

            # compute mismatch between clouds
            pc_control.transform(T_target)
            err = cloud2cloud_err(pc_source, pc_control, method=np.mean)
            pc_control.transform(np.linalg.inv(T_target))

            # update transformation is error is lower
            if err < min_err:
                T_final = T_target
                min_err = err

        print(f"\rAlignment for file {file} returned an error of {min_err:.4f}", end="")
        errors.append(min_err)
        transformations.append(T_final)
    errors = np.array(errors)
    files = np.array(files)
    transformations = np.array(transformations)

    if len(files) > 0:
        top_idx = np.argsort(errors).flatten()[0]
        print(f"\rBest guess is {files[top_idx]} with error {errors[top_idx]}")

    return files, errors, transformations


def rate_alignment(errors, epsilon_thresh=0.4, n_max=5):
    min_err = np.min(errors)
    rel_errors = 1. - min_err / errors
    num_below = len(rel_errors[rel_errors < epsilon_thresh])
    if num_below <= n_max:
        return True, np.argsort(errors)[:num_below]
    else:
        return False, None


def show_results(
    files: List[str],
    errors: Optional[np.ndarray] = None,
    transformations: Optional[List[np.ndarray]] = None,
    pc_source: Optional[o3d.geometry.PointCloud] = None,
    colored_dist: bool = True,
    mesh_only: bool = False,
):
    """
    Show alignment results for a file selection
    :param files: model files (n)
    :param errors: mismatch error array for each file (n, )
    :param transformations: list of transformations to get from target to source n x (4, 4)
    :param pc_source: source point cloud
    :param colored_dist: if True, color source vertices based on distance to the target model
    :param mesh_only: if True, draw only wire frame
    """
    color = mcolors.to_rgb(list(mcolors.TABLEAU_COLORS.values())[0])
    pc_source.paint_uniform_color(color)

    # if given, prepare idcs to loop from most likely to most unlikely
    err_idcs = range(len(files)) if errors is None else np.argsort(errors)
    for idx in err_idcs:
        file = files[idx]
        brick_id = file[:-4]

        # show error and prob
        prob = 100 * (1 / errors[idx]) / np.sum(1 / errors)
        print(f"---------------------\nModel {brick_id}:")
        if errors is not None:
            print(f"\tError:\t\t\t{errors[idx]:.6f}")
            print(f"\tProbability:\t{prob:.2f}%")

        # display current model
        mesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{brick_id}.stl")
        if transformations is not None:
            mesh.transform(transformations[idx])

        geometries = []
        if pc_source:
            if colored_dist:
                tmp_pc = mesh.sample_points_uniformly(100_000)
                dists = pc_source.compute_point_cloud_distance(tmp_pc)
                pc_source.colors = o3d.utility.Vector3dVector(compute_dist_colors(dists))
            geometries = [pc_source]

        if mesh_only:
            mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        else:
            mesh.compute_triangle_normals()
        geometries.append(mesh)

        o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True, left=1000, height=800)
