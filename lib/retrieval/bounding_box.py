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
from glpg_flowmeadow.transformations.methods import rotate_vec
from scipy.spatial import ConvexHull


def find_closest_obb_edges(source_edges, target_edges_array, thresh=0.01, max_best: int = None, return_err=False):
    """
    takes the summed quadratic distance between edge lengths as error. Returns an indices array
    containing the positions of errors below the threshold, sorted by their value
    :param source_edges: array (3,)
    :param target_edges_array: array (n, 3)
    :param thresh: maximum allowed error
    :param max_best: limit the number of files to return
    :param return_err: return errors in addition
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
    if return_err:
        return idcs, err
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


def PCA_based_alignment(
    pc: o3d.geometry.PointCloud,
    method="all",
    get_R=False,
) -> Union[o3d.geometry.PointCloud, Tuple[o3d.geometry.PointCloud, np.ndarray]]:
    """
    PCA based approach to rotate the given pointcloud such that the bounding box relative
    to world axes matches PCA axes
    :param pc: point cloud object
    :param method: Select a method for PCA alignment ('all' by default)
    :param get_R: if True, return rotation matrix as well
    :return: point cloud object or (point cloud object, rotation matrix (3, 3))
    """
    pc = copy.deepcopy(pc)
    pts = np.array(pc.points)

    # find the eigenvectors of the covariance matrix
    cov_mat = np.cov(pts, y=None, rowvar=0, bias=1)
    _, eigen_vecs = np.linalg.eigh(cov_mat)
    eigen_vecs /= np.linalg.norm(eigen_vecs, axis=0)

    # select eigenvector for projection in 'min/max' case, return rotation matrix for 'all' case otherwise
    R, n = None, None
    method = method.lower()
    if method == "min":
        n = eigen_vecs[:, -1]
    elif method == "max":
        n = eigen_vecs[:, 0]
    elif method == "all":
        R = eigen_vecs.T
    else:
        raise NotImplementedError("Only 'all', 'max' and 'min' are implemented methods")

    # compute rotation matrix for min/max case
    if method in ["min", "max"]:
        # project points to the plane defined by n
        dists = np.dot(pts, n)
        pts_proj = pts - np.matmul(dists.reshape(-1, 1), n.reshape(1, -1))

        # rotate all points, such that n aligns with z axis; all points lie in the xy plane afterwards
        init_angle = 180 * np.arccos(np.dot(n, np.array([0.0, 0.0, 1.0]))) / np.pi
        z_axis = np.array([0.0, 0.0, 1.0])
        init_axis = np.cross(n, z_axis) if np.any(np.abs(n) != z_axis) else z_axis
        pts_proj = rotate_vec(pts_proj, init_axis, init_angle)

        # compute convex hull points; by default they are in counterclockwise order
        hull = ConvexHull(pts_proj[:, :2])  # third dimension has been removed
        pts_hull = hull.points[hull.vertices]

        # align each edge of the convex hull with the x-axis and compute the bounding box volume
        min_e, min_V = None, np.inf
        for idx in range(pts_hull.shape[0]):
            # get normalized edge
            edge = pts_hull[idx] - pts_hull[(idx + 1) % pts_hull.shape[0]]
            edge /= np.linalg.norm(edge)

            # find the angle between edge and x-axis
            angle = -np.arctan(edge[1] / edge[0]) if edge[0] != 0.0 else 0.0

            # rotate all points, such that the edge flush with the x-axis
            px, py = pts[:, 0], pts[:, 1]
            qx = np.cos(angle) * px - np.sin(angle) * py
            qy = np.sin(angle) * px + np.cos(angle) * py
            pts_rot = np.stack([qx, qy]).T

            # compute the axis aligned OBB volume
            bb_min = np.min(pts_rot[:, :2], axis=0)
            bb_max = np.max(pts_rot[:, :2], axis=0)
            volume = np.prod(bb_max - bb_min)

            # save the optimal edge
            if volume < min_V:
                min_V, min_e = volume, edge

        # rotate optimal edge back to 3d space
        e = np.append(min_e, 0.0)
        e = rotate_vec(e.T, init_axis, -init_angle)

        # compute cross_product between n and e to obtain the third axis
        c = np.cross(e, n)

        # build rotation matrix
        # TODO: order important here?
        R = np.stack([n, e, c])

    # rotate point cloud accordingly and return rotation matrix if desired
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
