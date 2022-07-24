#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : handles 3D reconstruction pipeline for a series of stereo images
@File      : reconstruction.py
@Project   : BrickScanner
@Time      : 21.07.22 19:25
@Author    : flowmeadow
"""
import cv2
import numpy as np
import open3d as o3d
from definitions import *
from lib.helper.cloud_operations import data2cloud
from lib.recon.epipolar import compute_F
from lib.recon.keypoints import find_hcenters, find_keypoint_pairs
from lib.recon.masking import red_light_mask
from lib.recon.triangulation import triangulate_points


def reconstruct_point_cloud(
    folder_name: str,
    travel_dist: float = 0.5,
    gap_window: int = 10,
    y_extension: float = 2.0,
) -> o3d.geometry.PointCloud:
    """
    Reconstruct point cloud from images
    :param folder_name: folder name of the images, load from IMG_DIR
    :param travel_dist: distance the brick has moved on the belt
    :param gap_window: defines a window for how many rows to delete around 'gap' rows
    :param y_extension: extents the search area in y direction (in pixel dimension)
    :return:
    """
    # directories
    data_dir = f"{DATA_DIR}/{folder_name}"
    img_dir = f"{IMG_DIR}/{folder_name}"
    img_names = sorted(os.listdir(f"{img_dir}/left"))

    # load parameter
    K = np.load(f"{data_dir}/K.npy")
    T_W1 = np.load(f"{data_dir}/T_W1.npy")
    T_W2 = np.load(f"{data_dir}/T_W2.npy")

    # compute fundamental matrix
    F = compute_F(T_W1, T_W2, K, K)

    # repeat for each image pair
    point_cloud = None
    for img_idx, name in enumerate(img_names):
        print(f"\rComputing image {name} ({img_idx + 1}|{len(img_names)})", end="")
        # load images
        img_1, img_2 = cv2.imread(f"{img_dir}/left/{name}"), cv2.imread(f"{img_dir}/right/{name}")

        # generate mask for laser line
        mask_1, mask_2 = red_light_mask(img_1, gap_window=gap_window), red_light_mask(img_2, gap_window=gap_window)

        # find keypoints from mask
        kpts_1, kpts_2 = find_hcenters(mask_1), find_hcenters(mask_2)
        if kpts_1.shape[0] == 0 or kpts_2.shape[0] == 0:
            continue

        # find corresponding points
        pts_1, pts_2 = find_keypoint_pairs(kpts_1, kpts_2, F, y_extension=y_extension)

        # triangulate points
        pts_recon = triangulate_points(pts_1.T, pts_2.T, K, K, T_W1, T_W2)

        # shift between each frame in y (belt) direction
        pts_recon[:, 1] -= img_idx * (travel_dist / len(img_names))

        # add reconstructed point "slice" to point cloud
        if point_cloud is None:
            point_cloud = pts_recon.copy()
        else:
            point_cloud = np.append(point_cloud, pts_recon, axis=0)
    print(f"\nFinished reconstruction of {folder_name}")
    return data2cloud(point_cloud)
