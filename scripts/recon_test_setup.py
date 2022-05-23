#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
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
from lib.recon.triangulation import dlt
from rendering.models.model_generation.geometry import sphere
from transformations.methods import construct_T, rot_mat

from scripts.point_cloud import CloudScreen
from scripts.stereo_test import TestScreen


def concat_and_show(frame_1, frame_2):
    frame = cv2.hconcat((frame_1, frame_2))
    frame = cv2.resize(frame, (1760, 720))
    cv2.imshow("frame", frame)
    cv2.waitKey()


def unique_colors(num):
    c_num = np.ceil(np.cbrt(num) + 1.0).astype(int)
    c_bar = np.linspace(
        0.0,
        1.0,
        c_num,
    )
    colors = np.array(np.meshgrid(c_bar, c_bar, c_bar)).T.reshape(-1, 3)
    return colors[1 : num + 1, :]


def main():
    image_path = f"{IMG_DIR}/gl_test_01"  # "epipolar"
    generate_new = True
    img_idx = 0
    scale = 0.01

    # SECTION: Generate image pairs

    # load rotation matrix and translation vector
    R = np.load(f"{SETUP_DIR}/R.npy")
    # R = rot_mat(np.array([1., 1., 1.]), 20.)
    t = np.load(f"{SETUP_DIR}/T.npy").flatten() * scale
    # t = np.array([0.5, -0.5, 0.5])

    # create transformation matrix from camera system 1 to camera system 2
    # WARNING: no idea why t needs to be negative
    T_cam = construct_T(R, -t)

    # create sphere point cloud
    vertices, _ = sphere(1.0, refinement_steps=2)
    vertices = vertices.astype(float)
    print("Num Points:", vertices.shape[0])

    # create unique colors for each vertex
    colors = unique_colors(vertices.shape[0])

    # generate images of point cloud in simulator
    if generate_new:
        pc = dict(points=vertices, colors=colors)
        screen = TestScreen(T_cam, point_cloud=pc, file_path=image_path, fullscreen=True)
        screen.run()

    # SECTION: load images
    print("Load image pair")
    file_names = sorted(os.listdir(f"{image_path}/left"))
    img_left = cv2.imread(f"{image_path}/left/{file_names[img_idx]}")
    img_right = cv2.imread(f"{image_path}/right/{file_names[img_idx]}")
    concat_and_show(img_left, img_right)

    # SECTION: find keypoint correspondence by unique color
    print("Find keypoints")

    def get_kp(frame, rgb, thresh=0.005):
        error = thresh * 255
        diff = np.sum(np.abs(frame - np.flip(rgb)), axis=2)
        idcs = np.array(np.where(diff < error)).T
        return np.flip(np.mean(idcs, axis=0))

    kp_left, kp_right = [], []
    for color in colors:
        color = color * 255
        kp_left.append(cv2.KeyPoint(*get_kp(img_left, color), 1))
        kp_right.append(cv2.KeyPoint(*get_kp(img_right, color), 1))

    # SECTION: reconstruct point cloud
    K = np.array(
        [
            [1.30367535e03, 0.00000000e00, 9.60000000e02],
            [0.00000000e00, -1.30367537e03, 5.40000000e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    K = np.load(f"{image_path}/cam_K.npy")

    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    RT2 = np.concatenate([R, np.array([t]).T], axis=-1)

    P1 = K @ RT1  # projection matrix for C1
    P2 = K @ RT2  # projection matrix for C2

    p3ds = []
    colors = []
    for kp_l, kp_r in zip(kp_left, kp_right):

        # compute with DLT
        # _p3d = dlt(P1, P2, kp_l.pt, kp_r.pt)

        # compute with OCV
        _p3d = cv2.triangulatePoints(P1, P2, kp_l.pt, kp_r.pt).flatten()
        _p3d = _p3d[:-1] / _p3d[-1]

        p3ds.append(_p3d)

        kp_l, kp_r = np.array(kp_l.pt).astype(int), np.array(kp_r.pt).astype(int)
        c_l = img_left[kp_l[1], kp_l[0], :].astype(float)
        c_r = img_right[kp_r[1], kp_r[0], :].astype(float)
        colors.append((c_l + c_r) / 2)
    p3ds = np.array(p3ds)
    colors = np.flip(np.array(colors), axis=1) / 255

    # SECTION: transform point cloud back to world coordinates

    # transform point cloud based on cam pose
    T_cam = np.load(f"{image_path}/cam_T.npy")
    # WARNING: No idea why rotation part needs to be inverted
    T_cam[:3, :3] *= -1

    p3ds = np.concatenate([p3ds.T, [np.ones(p3ds.shape[0])]])
    p3ds = T_cam @ p3ds
    p3ds = p3ds.T[:, :-1]

    print("Num Points:", p3ds.shape[0])
    # SECTION: interactive point cloud presentation

    error = np.sum((p3ds - vertices) ** 2, axis=1)
    print(f"Max error distance: {np.max(error)}")
    print(f"Mean error distance: {np.mean(error)}")
    print(f"STD error distance: {np.std(error)}")

    demo = CloudScreen(points=p3ds, colors=colors, fullscreen=True)
    demo.run()


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
