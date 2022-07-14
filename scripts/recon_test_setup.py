#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Script to test the 3d reconstruction from images generated with an OpenGL simulator
@File      : recon_test_setup.py
@Project   : BrickScanner
@Time      : 07.03.22 17:53
@Author    : flowmeadow
"""
import os
import sys

sys.path.append(os.getcwd())  # required to run script from console

import cv2
import numpy as np
from definitions import *
from glpg_flowmeadow.rendering.models.model_generation.geometry import sphere
from lib.simulator.cloud_app import CloudApp
from lib.simulator.test_recon_app import TestReconApp


def concat_and_show(frame_1: np.ndarray, frame_2: np.ndarray):
    """
    Concatenate and show two images side by side.
    :param frame_1: image array
    :param frame_2: image array
    """
    frame = cv2.hconcat((frame_1, frame_2))
    frame = cv2.resize(frame, (1760, 720))
    cv2.imshow("frame", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()


def unique_colors(pts: np.ndarray) -> np.ndarray:
    """
    Generate RGB color array based on point positions
    :param pts: points array (n, 3)
    :return: RGB array (n, 3)
    """
    colors = pts.copy()
    colors -= np.min(colors, axis=0)
    colors /= np.max(colors, axis=0)
    return colors


def recon_test(image_path: str, generate_new=True):
    """
    Script to test the 3d reconstruction from images generated with an OpenGL simulator
    :param image_path: path to save the images
    :param generate_new: to skip image generation, set this to False
    """

    # SECTION: generate reference points
    # create sphere point cloud
    pts_true, _ = sphere(radius=1.0, refinement_steps=3)
    pts_true = pts_true.astype(float)
    colors = unique_colors(pts_true)
    print("Num Points:", pts_true.shape[0])

    # SECTION: generate stereo images from reference points
    # generate images of point cloud in simulator
    if generate_new:
        app = TestReconApp(pts_true, colors, file_path=image_path, fullscreen=True, vsync=False)
        app.run()

        if app.new_images:
            np.save(f"{CALIB_DIR}/recon_test/K.npy", app.K)
            np.save(f"{CALIB_DIR}/recon_test/T_W1.npy", app.T_W1)
            np.save(f"{CALIB_DIR}/recon_test/T_W2.npy", app.T_W2)

    T_W1 = np.load(f"{CALIB_DIR}/recon_test/T_W1.npy")
    T_W2 = np.load(f"{CALIB_DIR}/recon_test/T_W2.npy")

    # SECTION: load images
    print("Load image pair")
    file_names = sorted(os.listdir(f"{image_path}/left"))
    img_left = cv2.imread(f"{image_path}/left/{file_names[0]}")
    img_right = cv2.imread(f"{image_path}/right/{file_names[0]}")
    concat_and_show(img_left, img_right)

    # SECTION: find keypoint correspondence by unique color
    kpts_left, kpts_right = np.zeros([2, colors.shape[0]]), np.zeros([2, colors.shape[0]])
    for idx, color in enumerate(colors):
        thresh = 1.0e-6 * 255  # in simulation only required for rounding errors. can be very small
        color = np.flip(color) * 255  # OpenCV uses BGR instead of RGB

        # for each color get a binary mask and its mass center as ...
        # ... keypoint for the left image
        mask = cv2.inRange(img_left, color - thresh, color + thresh)
        M = cv2.moments(mask)
        kp_left = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        # ... keypoint for the right image
        mask = cv2.inRange(img_right, color - thresh, color + thresh)
        M = cv2.moments(mask)
        kp_right = (M["m10"] / M["m00"], M["m01"] / M["m00"])

        kpts_left[:, idx] = kp_left
        kpts_right[:, idx] = kp_right

    # SECTION: reconstruct point cloud
    K = np.load(f"{CALIB_DIR}/recon_test/K.npy")  # OpenCV camera matrix
    T_1W = np.linalg.inv(T_W1)  # transformation matrix from 3d world space to 3d cam 1 space
    T_2W = np.linalg.inv(T_W2)  # transformation matrix from 3d world space to 3d cam 2 space
    P1 = K @ T_1W[:3]  # projection matrix from 3d world space to cam 1 image space
    P2 = K @ T_2W[:3]  # projection matrix from 3d world space to cam 2 image space

    # triangulate keypoints
    pts_recon = cv2.triangulatePoints(P1, P2, kpts_left, kpts_right).T
    # returns are homogeneous, so they need to be transformed back to 3D
    pts_recon = (pts_recon.T / pts_recon[:, 3]).T[:, :3]

    # SECTION: Presentation of reconstructed points and comparison with true points
    error = np.sum((pts_recon - pts_true) ** 2, axis=1)
    print(f"Max error distance: {np.max(error)}")
    print(f"Mean error distance: {np.mean(error)}")
    print(f"STD error distance: {np.std(error)}")

    app = CloudApp(points=pts_recon, colors=colors, fullscreen=True)
    app.run()


if __name__ == "__main__":
    recon_test(image_path=f"{IMG_DIR}/recon_test", generate_new=True)
