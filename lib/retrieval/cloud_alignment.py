#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : cloud_alignment.py
@Project   : BrickScanner
@Time      : 20.06.22 17:48
@Author    : flowmeadow
"""

import copy
import pickle
from typing import List

import numpy as np
import open3d as o3d
from definitions import *
from glpg_flowmeadow.transformations.methods import rot_mat
from lib.helper.cloud_operations import cloud2cloud_err, construct_T, draw_point_clouds, rotate_random
from lib.retrieval.bounding_box import (
    PCA_based_alignment,
    compute_obb_center,
    compute_obb_edges,
    find_closest_obb_edges,
)


def prepare_cloud(pc: o3d.geometry.PointCloud, random=False) -> o3d.geometry.PointCloud:
    """
    prepares source point cloud for cloud alignment
    :param pc: source point cloud
    :param random: if True, the cloud is rotated randomly first
    :return: source point cloud
    """
    if random:
        pc = rotate_random(pc)
    # align point cloud with world axes
    pc = PCA_based_alignment(pc)
    # move obb center to origin
    pc.translate(-compute_obb_center(pc))
    # estimate normals
    pc.estimate_normals()
    return pc


def align_point_clouds(
    pc_source: o3d.geometry.PointCloud, pc_target: o3d.geometry.PointCloud, debug=False
) -> np.ndarray:
    """
    Aligns the target point_cloud with source point cloud
    IMPORTANT: pc_source needs to be PCA aligned and have its obb center in world center
    :param pc_source: source point cloud
    :param pc_target: target point cloud
    :param debug: show alignment pipeline steps
    :return: transformation matrix for target to be aligned with source (4, 4)
    """
    pc_source = copy.deepcopy(pc_source)
    pc_target = copy.deepcopy(pc_target)
    T_target = np.eye(4)

    if debug:
        print("DEBUG: Initial orientation")
        draw_point_clouds(pc_source, pc_target)

    # get orientation by PCA based alignment
    # TODO: This is different from:
    #       pc_target, R_init = PCA_based_alignment(pc_target, get_R=True)
    #       Why the heck???
    _, R_init = PCA_based_alignment(pc_target, get_R=True)
    pc_target = pc_target.rotate(R_init, center=np.zeros(3))

    T_target = construct_T(R=R_init) @ T_target
    if debug:
        print("DEBUG: PCA based alignment")
        draw_point_clouds(pc_source, pc_target)

    # move it to world center
    t = -compute_obb_center(pc_target)
    T_target = construct_T(t=t) @ T_target
    pc_target = pc_target.translate(t)
    if debug:
        print("DEBUG: shift OBB center to origin")
        draw_point_clouds(pc_source, pc_target)

    # rotate around several axes to find best alignment
    rotation_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    min_err = cloud2cloud_err(pc_source, pc_target)  # current error
    R_min = np.eye(3)  # current rotation
    for ax in rotation_axes:
        # rotate around axis
        R = rot_mat(ax, 180)
        pc_target.rotate(R, center=np.zeros(3))

        # compute new error
        err = cloud2cloud_err(pc_source, pc_target)
        if err < min_err:  # did it get better?
            min_err = err
            R_min = R
        pc_target.rotate(R.T, center=np.zeros(3))  # rotate back

    # take optimal rotation and apply it
    pc_target.rotate(R_min, center=np.zeros(3))
    T_target = construct_T(R=R_min) @ T_target
    if debug:
        print("DEBUG: Optimized alignment")
        draw_point_clouds(pc_source, pc_target)

    # final optimization using ICP
    threshold = 1.0
    trans_init = np.eye(4)
    pc_target.estimate_normals()
    pc_source.estimate_normals()
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
        draw_point_clouds(pc_source, pc_target)
    return T_target


def find_model(pc_source, debug_file: str = None, threshold=0.01):
    """
    Given a source point cloud, search for a model that matches from the Ldraw library
    :param pc_source: source point cloud
    :param debug_file: file_name of the correct model (Optional)
    :param threshold: maximum allowed error for PCA OBB comparison
    :return: [used files, alignment errors, list of target transformations (4, 4), match probabilities for each file]
    """
    if debug_file:
        o3d.visualization.draw_geometries([pc_source], left=1000)

    # load PCA aligned obb edges
    # save_base_obb_data(f"{STL_DIR}/evaluation.csv")
    with open(f"{DATA_DIR}/base_obb_data.pkl", "rb") as f:
        obb_data = pickle.load(f)

    # preselect models that have a PCA aligned OBB similar to source point cloud
    idcs = find_closest_obb_edges(compute_obb_edges(pc_source), obb_data["edges"], thresh=threshold)
    files = np.array(obb_data["file_names"])[idcs]
    print(f"Found {len(files)} files that have an obb edge error lower than {threshold}")
    if debug_file:
        print(f"Original file {debug_file} is {'' if debug_file in files else 'NOT'} in file selection")

    # align point clouds for each file
    errors = []
    transformations = []
    for file in files:
        debug = file == debug_file

        # convert model to point cloud
        mesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{file}")
        pc_target: o3d.geometry.PointCloud = mesh.sample_points_uniformly(100_000)

        # compute best alignment transformation
        T_target = align_point_clouds(pc_source, pc_target, debug=debug)
        pc_target.transform(T_target)

        # compute mismatch between clouds
        err = cloud2cloud_err(pc_source, pc_target)

        print(f"Alignment for file {file} returned an error of {err:.4f}")
        errors.append(err)
        transformations.append(T_target)

    errors = np.array(errors)
    # compute a probability estimate for each file
    percentages = 100 * (1 / errors) / np.sum(1 / errors)
    return files, errors, transformations, percentages


def show_results(
    files: List[str],
    errors: np.ndarray = None,
    transformations: List[np.ndarray] = None,
    percentages: np.ndarray = None,
    pc_source: o3d.geometry.PointCloud = None,
):
    """
    Show alignment results for a file selection
    :param files: model files (n)
    :param errors: mismatch error array for each file (n, )
    :param transformations: list of transformations to get from target to source n x (4, 4)
    :param percentages: probability estimate array for each file (n, )
    :param pc_source: source point cloud
    """

    # if given, prepare idcs to loop from most likely to most unlikely
    err_idcs = range(len(files)) if errors is None else np.argsort(errors)
    for idx in err_idcs:
        file = files[idx]
        brick_id = file[:-4]

        # show error and prob
        print(f"Model {brick_id}:")
        if errors is not None:
            print(f"\tError: {errors[idx]}")
        if percentages is not None:
            print(f"\tProbability: {percentages[idx]}")

        # display current model
        mesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{brick_id}.stl")
        if transformations and mesh:
            mesh.transform(transformations[idx])
            geometries = [mesh, pc_source]
        else:
            geometries = [mesh]
        o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True, left=1000, height=800)
