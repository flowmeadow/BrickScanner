#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : PCA bounding box computation methods
@File      : pca_obb.py
@Project   : BrickScanner
@Time      : 01.08.22 23:01
@Author    : flowmeadow
"""
import copy

import numpy as np
import open3d as o3d
from glpg_flowmeadow.transformations.methods import rotate_vec

from lib.helper.cloud_operations import rotate_random
from lib.helper.lego_bricks import load_stl
from lib.retrieval.bounding_box import compute_obb_volume


def all_pca(pc: o3d.geometry.PointCloud):
    pc = copy.deepcopy(pc)
    pts = np.array(pc.points)
    cov_mat = np.cov(pts, y=None, rowvar=0, bias=1)
    _, eigen_vecs = np.linalg.eigh(cov_mat)
    eigen_vecs /= np.linalg.norm(eigen_vecs, axis=1)[:, np.newaxis]  # TODO: maybe not needed
    R = eigen_vecs.T
    pc.rotate(R, center=np.zeros(3))
    return pc


def rotate_vec_2D(pts, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    pts = pts.copy()
    px, py = pts[:, 0], pts[:, 1]
    qx = np.cos(angle) * px - np.sin(angle) * py
    qy = np.sin(angle) * px + np.cos(angle) * py
    return np.stack([qx, qy]).T


def minmax_pca(pc: o3d.geometry.PointCloud, method="min"):
    pc = copy.deepcopy(pc)
    pts = np.array(pc.points)
    cov_mat = np.cov(pts, y=None, rowvar=0, bias=1)
    _, eigen_vecs = np.linalg.eigh(cov_mat)

    eigen_vecs = eigen_vecs / np.linalg.norm(eigen_vecs, axis=0)

    R, n = None, None
    if method == "min":
        n = eigen_vecs[:, -1]
    elif method == "max":
        n = eigen_vecs[:, 0]
    elif method == "all":
        R = eigen_vecs.T
    else:
        raise NotImplementedError()

    if method in ["min", "max"]:
        dists = np.dot(pts, n)
        pts_proj = pts - np.matmul(dists.reshape(-1, 1), n.reshape(1, -1))

        # rotate such that first principal component aligns with z axis
        z = np.array([0.0, 0.0, 1.0])
        angle = 180 * np.arccos(np.dot(n, z)) / np.pi
        axis = np.cross(n, z)
        pts_proj = rotate_vec(pts_proj, axis, angle)

        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts_proj[:, :2])
        idcs = hull.vertices

        pts_hull = hull.points[idcs]

        min_edge, min_V = None, np.inf
        for idx in range(pts_hull.shape[0]):
            edge = pts_hull[idx] - pts_hull[(idx + 1) % pts_hull.shape[0]]
            edge /= np.linalg.norm(edge)

            x_angle = np.arctan(edge[1] / edge[0])
            pts_rot = rotate_vec_2D(pts_hull, -x_angle)

            bb_min = np.min(pts_rot[:, :2], axis=0)
            bb_max = np.max(pts_rot[:, :2], axis=0)
            volume = np.prod(bb_max - bb_min)

            if volume < min_V:
                min_V = volume
                min_edge = edge

        edge_3d = np.zeros(3)
        edge_3d[:2] = min_edge
        edge_3d /= np.linalg.norm(edge_3d)
        edge_3d = rotate_vec(edge_3d.T, axis, -angle)
        R = np.stack([n, edge_3d, np.cross(edge_3d, n)])

    pc.rotate(R, center=np.zeros(3))
    return pc


if __name__ == "__main__":
    brick_id = "3148"
    scale_factor = 0.1

    # generate mesh
    mesh = load_stl(brick_id)
    mesh.compute_vertex_normals()

    methods = ["all", "max", "min"]
    for i in range(10):
        pc: o3d.geometry.PointCloud = mesh.sample_points_uniformly(100_000)
        pc = rotate_random(pc)
        print("-----------")
        for method in methods:
            pc_all = minmax_pca(pc, method=method)
            v = compute_obb_volume(pc_all)
            print(method, v)
