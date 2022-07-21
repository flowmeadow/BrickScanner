#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Script to test the 3d reconstruction from images generated with an OpenGL simulator
@File      : sim_recon_test.py
@Project   : BrickScanner
@Time      : 07.03.22 17:53
@Author    : flowmeadow
"""
import os
import sys

sys.path.append(os.getcwd())  # required to run script from console

from lib.recon.triangulation import triangulate_points
import cv2
import numpy as np
from definitions import *
from glpg_flowmeadow.rendering.models.model_generation.geometry import sphere
from lib.simulator.cloud_app import CloudApp
from lib.simulator.test_recon_app import TestReconApp


def concat_and_show(frame_1: np.ndarray, frame_2: np.ndarray):
    """
    Concatenate and show two images side by side
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


def recon_test(
    folder_name: str,
    generate_new=True,
    automated=False,
    gen_rand=False,
    T_W1: np.ndarray = None,
    T_W2: np.ndarray = None,
):
    """
    Script to test the 3d reconstruction from images generated with an OpenGL simulator
    :param folder_name: name of the experiment folder
    :param automated: If true, the image generation is done automatically and the app is closed afterwards
    :param gen_rand: generate random camera positions
    :param generate_new: to skip image generation, set this to False
    :param T_W1: Initial transformation matrix for cam 1 (4, 4)
    :param T_W2: Initial transformation matrix for cam 2 (4, 4)
    """
    image_path = f"{IMG_DIR}/{folder_name}"
    data_path = f"{DATA_DIR}/{folder_name}"
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # SECTION: generate reference points
    # create sphere point cloud
    pts_true, _ = sphere(radius=1.0, refinement_steps=3)
    pts_true = pts_true.astype(float)
    colors = unique_colors(pts_true)
    print("Num Points:", pts_true.shape[0])

    # SECTION: generate stereo images from reference points
    # generate images of point cloud in simulator
    if generate_new:
        app = TestReconApp(
            pts_true,
            colors,
            image_dir=image_path,
            automated=automated,
            fullscreen=True,
            gen_rand=gen_rand,
            T_W1=T_W1,
            T_W2=T_W2,
        )
        app.run()
        if app.new_images:
            np.save(f"{data_path}/K.npy", app.K)
            np.save(f"{data_path}/T_W1.npy", app.T_W1)
            np.save(f"{data_path}/T_W2.npy", app.T_W2)

    T_W1 = np.load(f"{data_path}/T_W1.npy")
    T_W2 = np.load(f"{data_path}/T_W2.npy")

    # SECTION: load images
    print("Load image pair")
    file_names = sorted(os.listdir(f"{image_path}/left"))
    img_left = cv2.imread(f"{image_path}/left/{file_names[0]}")
    img_right = cv2.imread(f"{image_path}/right/{file_names[0]}")
    if not automated:
        concat_and_show(img_left, img_right)

    # SECTION: find keypoint correspondence by unique color
    kpts_left, kpts_right = np.zeros([2, colors.shape[0]]), np.zeros([2, colors.shape[0]])
    remove_idcs = []  # remove certain points from pts_true, if no point correspondence was found
    for idx, color in enumerate(colors):
        thresh = 1.0e-6 * 255  # in simulation only required for rounding errors. can be very small
        color = np.flip(color) * 255  # OpenCV uses BGR instead of RGB

        # get a binary mask
        mask_l = cv2.inRange(img_left, color - thresh, color + thresh)
        mask_r = cv2.inRange(img_right, color - thresh, color + thresh)

        # find the moments
        M_l, M_r = cv2.moments(mask_l), cv2.moments(mask_r)
        if M_l["m00"] == 0.0 or M_r["m00"] == 0.0:
            remove_idcs.append(idx)
            continue

        # compute centers
        kp_left = (M_l["m10"] / M_l["m00"], M_l["m01"] / M_l["m00"])
        kp_right = (M_r["m10"] / M_r["m00"], M_r["m01"] / M_r["m00"])

        kpts_left[:, idx] = kp_left
        kpts_right[:, idx] = kp_right

    # remove points if necessary
    if remove_idcs:
        print(f"WARNING: no correspondence found for points {remove_idcs}")
        pts_true = np.delete(pts_true, remove_idcs, axis=0)
        kpts_left = np.delete(kpts_left, remove_idcs, axis=1)
        kpts_right = np.delete(kpts_right, remove_idcs, axis=1)

    # SECTION: reconstruct point cloud
    K = np.load(f"{data_path}/K.npy")  # OpenCV camera matrix
    pts_recon = triangulate_points(kpts_left, kpts_right, K, K, T_W1, T_W2)

    # SECTION: Presentation of reconstructed points and comparison with true points
    # the error is the quadratic distance between both point clouds
    error = np.linalg.norm(pts_recon - pts_true, axis=1) ** 2
    # error = np.sum((pts_recon - pts_true) ** 2, axis=1)
    np.save(f"{data_path}/errors.npy", error)

    print(f"Max error distance: {np.max(error)}")
    print(f"Mean error distance: {np.mean(error)}")
    print(f"STD error distance: {np.std(error)}")
    print(f"Summed error distance: {np.sum(error)}")

    if not automated:
        app = CloudApp(points=pts_recon, colors=colors, fullscreen=True)
        app.run()

    return error


if __name__ == "__main__":
    recon_test(folder_name=f"recon_test/demo", generate_new=True, automated=False, gen_rand=False)
