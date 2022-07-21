#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : several point cloud and triangle mesh operations
@File      : cloud_operations.py
@Project   : BrickScanner
@Time      : 05.05.22 13:16
@Author    : flowmeadow
"""

from typing import Dict

import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap


def mdim_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    perform dot product for multidimensional arrays row wise
    :param a: first factor (m, 3)
    :param b: second factor (m, 3)
    :return: dot products (m, 1)
    """
    return np.sum(a * b, axis=1)


def data2mesh(mesh_data: Dict[np.ndarray, np.ndarray]) -> o3d.geometry.TriangleMesh:
    """
    Convert mesh data containing vertices and triangle indices to open3d TriangleMesh object
    :param mesh_data: dictionary containing vertices (m, 3) and triangle indices (n, 3)
    :return: TriangleMesh object
    """
    np_vertices = mesh_data["vertices"]
    np_triangles = mesh_data["indices"]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
    return mesh


def data2cloud(cloud_data: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert array of point coordinates to open3d PointCloud object
    :param cloud_data: array of point coordinates (m, 3)
    :return: PointCloud object"""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_data)
    return cloud


def pt_in_tri(p_vec: np.ndarray, triangles_vec: np.ndarray) -> np.ndarray:
    """
    Source: https://gdbooks.gitbooks.io/3dcollisions/content/Chapter4/point_in_triangle.html
    Takes an 2D array of point coordinates and an 3D array of triangle coordinates and computes for each point,
    if it lies inside its corresponding triangle. Returns an array of boolean values, one for each point
    :param p_vec: array of point coordinates (m, 3)
    :param triangles_vec: array of triangle coordinates, 3 point coordinates for each corner (m, 3, 3)
    :return: boolean array (m,)
    """
    a, b, c = triangles_vec[:, 0], triangles_vec[:, 1], triangles_vec[:, 2]
    p = p_vec

    # Move the triangle so that the point becomes the triangles origin
    a = a - p
    b = b - p
    c = c - p

    # Compute the normal vectors for triangles:
    # u = normal of PBC
    # v = normal of PCA
    # w = normal of PAB
    u = np.cross(b, c)
    v = np.cross(c, a)
    w = np.cross(a, b)

    # Test to see if the normals are facing the same direction, return false if not
    return np.logical_not(np.logical_or(mdim_dot(u, v) < 0, mdim_dot(u, w) < 0))


def pt2tri_dist(p_vec: np.ndarray, triangles_vec: np.ndarray) -> np.ndarray:
    """
    Source: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.479.8237&rep=rep1&type=pdf
    Takes an 2D array of point coordinates and an 3D array of triangle coordinates and computes for each point
    the distance to its corresponding triangle. If the projected point in (triangle) normal direction lies within
    the triangle, the projected distance is used. If the projected point lies outside, the minimum distance to one
    of the triangle corners is used. Returns an array of distance values, one for each point
    :param p_vec: array of point coordinates (m, 3)
    :param triangles_vec: array of triangle coordinates, 3 point coordinates for each corner (m, 3, 3)
    :return: array of distance values (m,)
    """
    p_1, p_2, p_3 = triangles_vec[:, 0], triangles_vec[:, 1], triangles_vec[:, 2]
    p = p_vec

    # compute projection distance
    edge = p_1 - p
    n = np.cross(p_2 - p_1, p_3 - p_1)
    n_length = np.linalg.norm(n, axis=1)
    n = n / n_length[:, np.newaxis]
    edge_dist = np.linalg.norm(edge, axis=1)
    c_alpha = mdim_dot(edge, n) / edge_dist
    dist_proj = edge_dist * c_alpha

    # compute the coordinates of the projected point
    p_proj = p - dist_proj[:, np.newaxis] * n

    # take absolute value for distance
    dist = np.abs(dist_proj)

    # check where projected point lies within triangle
    inside = pt_in_tri(p_proj, triangles_vec)

    # compute the minimum distance from point to one of the triangle corners
    # TODO: this is only an approximation. Actually the closest distance could be on an edge
    outer_dists = triangles_vec - p[:, np.newaxis]
    outer_dists = np.linalg.norm(outer_dists, axis=2)
    outer_dists = np.min(outer_dists, axis=1)

    # if points lie outside the triangle, replace distance with minimum corner distance
    replace_idcs = np.where(np.logical_not(inside))
    dist[replace_idcs] = outer_dists[replace_idcs]
    return dist


def m2c_dist_rough(
    mesh: o3d.geometry.TriangleMesh,
    cloud: o3d.geometry.PointCloud,
    num_samples: int = 1_000_000,
) -> np.ndarray:
    """
    Computes the distance between a given TriangleMesh object and a PointCloud object. The distance is estimated by
    sampling points from mesh and returning the distance from each cloud point to the closest sampled point.
    Accuracy increases with 'sample_num'
    :param mesh: Triangle mesh as open3d TriangleMesh object
    :param cloud: point cloud as open3d PointCloud object
    :param num_samples: number of points to sample from mesh
    :return: array of distance values for each point in cloud (m,)
    """
    # create point cloud from mesh
    sampled_cloud = mesh.sample_points_uniformly(num_samples)

    # compute distance
    dist = cloud.compute_point_cloud_distance(sampled_cloud)

    # o3d.visualization.draw_geometries([cloud, sampled_cloud])
    return np.asarray(dist)


def m2c_dist(mesh: o3d.geometry.TriangleMesh, cloud: o3d.geometry.PointCloud, k: int = 15) -> np.ndarray:
    """
    TODO: not working properly
    Computes the distance between a given TriangleMesh object and a PointCloud object. For each point the smallest
    distance to the k neighboring triangles is computed
    :param mesh: Triangle mesh as open3d TriangleMesh object
    :param cloud: point cloud as open3d PointCloud object
    :param k: number of neighboring triangle centers to find
    :return: array of distance values for each point in cloud (K,)
    """
    # get triangle center points of mesh
    triangles = np.array(mesh.triangles)
    vertices = np.array(mesh.vertices)
    t_centers = np.mean(vertices[triangles], axis=1)

    # prepare data
    points = np.array(cloud.points)
    mesh_cloud = data2cloud(t_centers)  # generate cloud from mesh triangle centers
    vertices_tree = o3d.geometry.KDTreeFlann(mesh_cloud)  # generate KDTree from mesh_cloud

    # find indices of the k-nearest neighboring triangles for each point in cloud
    t_idcs = np.zeros((len(cloud.points), k)).astype(np.uint)
    for idx, point in enumerate(cloud.points):
        _, idcs, _ = vertices_tree.search_knn_vector_3d(point, k)
        t_idcs[idx, :] = np.array(idcs)

    # compute distances from p to each of the k triangles
    dists = []
    for k_idx in range(k):
        v_idcs = triangles[t_idcs[:, k_idx]]
        tri_pts = vertices[v_idcs]
        d = pt2tri_dist(points, tri_pts)
        dists.append(d)
    # return minimum distance over all k triangles
    return np.min(np.array(dists), axis=0)


def compute_dist_colors(dist: np.ndarray) -> np.ndarray:
    """
    Computes a color for each distance value according to the given color map. The RGB color values are in range [0, 1]
    :param dist: distance values (m,)
    :return: RGB color array (m, 3)
    """
    c_weight = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))  # map distance values to [0, 1]
    color_keys = ["limegreen", "yellow", "orange", "red"]
    cmap = LinearSegmentedColormap.from_list("rg", color_keys, N=256)
    rgb = cmap(c_weight)[:, :3]
    return rgb


def display_dist(dist: np.ndarray, cloud: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh):
    """
    Displays point cloud to mesh distance
    :param dist: distance values (m,)
    :param cloud: Point cloud object; has to have m points
    :param mesh: reference mesh
    """
    # print distance results
    print(f"MAX dist: {np.max(dist)}")
    print(f"MEAN dist: {np.mean(dist)}")
    print(f"STD dist: {np.std(dist)}")
    print(f"SUM dist: {np.sum(dist)}")

    cloud.colors = o3d.utility.Vector3dVector(compute_dist_colors(dist))  # assign point colors
    o3d.visualization.draw_geometries([cloud, mesh], mesh_show_wireframe=True)
