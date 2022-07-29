#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Handles methods to find the minimum bounding box for a pointcloud
@File      : bounding_box.py
@Project   : BrickScanner
@Time      : 30.06.22 14:33
@Author    : flowmeadow
"""
import copy
from typing import Tuple, Union

import numpy as np
import open3d as o3d


def find_closest_obb_edges(source_edges, target_edges_array, thresh=0.01, max_best: int = None):
    """
    takes the summed quadratic distance between edge lengths as error. Returns an indices array
    containing the positions of errors below the threshold, sorted by their value
    :param source_edges: array (3,)
    :param target_edges_array: array (n, 3)
    :param thresh: maximum allowed error
    :return: indices array (m, 3); m <= n
    """
    # volume error
    # target_v_array = np.prod(target_edges_array, axis=1)
    # source_v = np.prod(source_edges)
    # err = np.abs(target_v_array / source_v - 1.0) ** 2

    err = (target_edges_array / source_edges - 1.0) ** 2
    err = np.sum(err, axis=1)

    thresh_idcs = np.argwhere(err <= thresh)
    idcs = thresh_idcs[np.argsort(err[thresh_idcs].flatten())].flatten()

    if max_best and len(idcs) > max_best:
        idcs = idcs[:max_best]
    return idcs


def compute_obb_edges(geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]) -> np.ndarray:
    """
    Computes the bounding box edges relative to world axes of given point cloud
    :param geometry: point cloud object or triangle mesh object
    :return: sorted array of edges from smallest to largest (3,)
    """
    if isinstance(geometry, o3d.geometry.PointCloud):
        pts = np.array(geometry.points)
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        pts = np.array(geometry.vertices)
    else:
        raise NotImplementedError("Only triangle mesh and point cloud objects are allowed")
    bb_min = np.min(pts, axis=0)
    bb_max = np.max(pts, axis=0)
    extents = bb_max - bb_min
    return np.sort(extents)


def compute_obb_center(pc: o3d.geometry.PointCloud) -> float:
    """
    Computes the bounding box center relative to world axes of given point cloud
    :param pc: point cloud object
    :return: obb volume
    """
    pts = np.array(pc.points)
    bb_min = np.min(pts, axis=0)
    bb_max = np.max(pts, axis=0)
    center = bb_min + (bb_max - bb_min) / 2
    return center


def compute_obb_volume(pc: o3d.geometry.PointCloud) -> float:
    """
    Computes the bounding box volume relative to world axes of given point cloud
    :param pc: point cloud object
    :return: obb volume
    """
    return np.prod(compute_obb_edges(pc))


def iterative_alignment(
    pc: o3d.geometry.PointCloud,
    start_step: float = np.pi * 0.25,
    min_diff: float = 10e-4,
    reduction_factor: float = 2.0,
    max_iteration: int = 1000,
) -> o3d.geometry.PointCloud:
    """
    Iterative approach to rotate the given pointcloud such that the bounding box relative
    to world axes gets a minimum volume. Likely to find a local minimum
    :param pc: point cloud object
    :param start_step: start rotation angle in rad
    :param min_diff: minimum volume difference between to steps
    :param reduction_factor: reduce rotation angle by this factor each step
    :param max_iteration: stop after this number of iterations
    :return: rotated point cloud object
    """
    pc = copy.deepcopy(pc)

    # repeat for all world axes
    for axis_idx in range(3):
        # update rotation axis
        axis = np.zeros(3)
        axis[axis_idx] = 1

        # compute volume
        volume = compute_obb_volume(pc)

        dv = np.inf  # differences between volumes
        cnt = 0  # iteration count
        step = start_step  # rotation angle
        while dv >= min_diff and cnt < max_iteration:
            # rotate PC around axis with current angle
            R = pc.get_rotation_matrix_from_axis_angle(step * axis)
            pc.rotate(R)

            # compute new volume
            new_volume = compute_obb_volume(pc)
            if new_volume > volume:  # no improvement
                # rotate in the opposite direction
                axis = -axis
                if np.sum(axis) > 0:  # if both directions have been tried, reduce rotation angle
                    step /= reduction_factor
                pc.rotate(R.T)
            else:  # improvement
                # update relative volume difference
                dv = np.abs(new_volume - volume) / volume
                volume = new_volume
            cnt += 1
        if cnt >= max_iteration:
            print(f"WARNING: maximum iteration number exceeded for axis {axis_idx}")
    return pc


def PCA_based_alignment(pc, get_R=False) -> Union[o3d.geometry.PointCloud, Tuple[o3d.geometry.PointCloud, np.ndarray]]:
    """
    PCA based approach to rotate the given pointcloud such that the bounding box relative
    to world axes matches PCA axes
    :param pc: point cloud object
    :param get_R: if True, return rotation matrix as well
    :return: point cloud object or (point cloud object, rotation matrix (3, 3))
    """
    pc = copy.deepcopy(pc)
    pts = np.array(pc.points)
    cov_mat = np.cov(pts, y=None, rowvar=0, bias=1)
    _, eigen_vecs = np.linalg.eigh(cov_mat)
    eigen_vecs /= np.linalg.norm(eigen_vecs, axis=0)
    R = eigen_vecs.T

    pc.rotate(R, center=np.zeros(3))
    if get_R:
        return pc, R
    else:
        return pc


class AxisAlignedCloud:
    """
    Class that handles a point cloud and performs axis alignment
    """

    def __init__(self, pc: o3d.geometry.PointCloud):
        """
        :param pc: point cloud object
        """
        self._pc = copy.deepcopy(pc)

    def optimize(self, method: str = None) -> o3d.geometry.PointCloud:
        """
        Performs point cloud axis alignment
        :param method: set an alignment method (Optional)
        :return: point cloud object
        """
        allowed_methods = ["iterative", "pca_based", "combined"]

        if method is None:  # set default method
            method = "combined"

        if method == "iterative":
            pc = iterative_alignment(self._pc)
        elif method == "pca_based":
            pc = PCA_based_alignment(self._pc)
        elif method == "combined":
            pc = iterative_alignment(PCA_based_alignment(self._pc))
        else:
            raise ValueError(f"method has to be in '{allowed_methods}' and not '{method}'")
        return pc
