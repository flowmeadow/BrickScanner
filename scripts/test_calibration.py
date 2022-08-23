#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : test_calibration.py
@Project   : BrickScanner
@Time      : 06.08.22 14:54
@Author    : flowmeadow
"""
import math
import sys
import cv2
import numpy as np

from definitions import *
from lib.helper.cloud_operations import construct_T
from lib.helper.interactive_window import InteractiveWindow


def rodrigues_vec_to_rotation_mat(rodrigues_vec):
    theta = np.linalg.norm(rodrigues_vec)
    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vec / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array(
            [
                [r[0] * r[0], r[0] * r[1], r[0] * r[2]],
                [r[1] * r[0], r[1] * r[1], r[1] * r[2]],
                [r[2] * r[0], r[2] * r[1], r[2] * r[2]],
            ]
        )
        r_cross = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat


def main():
    img_path = f"{IMG_DIR}/220621-200516_calib"

    R_1 = np.load(f"{SETUP_DIR}/R_1.npy")[0]
    T_1 = np.load(f"{SETUP_DIR}/T_1.npy")[0]
    K_1 = np.load(f"{SETUP_DIR}/K_left.npy")
    K_2 = np.load(f"{SETUP_DIR}/K_right.npy")
    dist_1 = np.load(f"{SETUP_DIR}/dist_left.npy")
    dist_2 = np.load(f"{SETUP_DIR}/dist_right.npy")
    R_1 = rodrigues_vec_to_rotation_mat(R_1.flatten())
    folder = "left"

    win = InteractiveWindow("frame")

    for file_name in os.listdir(f"{img_path}/{folder}"):
        print(f"Reading image {folder}/{file_name}")
        orig = cv2.imread(f"{img_path}/{folder}/{file_name}")
        ret_1, K_1, dist_1, _, _ = cv2.calibrateCamera(obj_points, img_points_left, frame_1.shape[:2], None, None)
        print("Calibration of cam 2 ...")
        ret_2, K_2, dist_2, _, _ = cv2.calibrateCamera(obj_points, img_points_left, frame_2.shape[:2], None, None)
        # load fundamental matrix
        F = np.load(f"{SETUP_DIR}/F.npy")

        # program loop
        old_x, old_y = 0, 0
        while True:
            frame = orig.copy()

            # check for quit (q)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

            # check for mouse button press
            x, y = win.mouse_pos_x, win.mouse_pos_y
            if x is not None and y is not None:
                frame = cv2.circle(frame, (x, y), 4, (0, 0, 255), 1)  # draw circle at mouse pos

                if x != old_x or y != old_y:
                    print("-----------------------------")
                    print(f"2D point: ({x}, {y})")

                    # compute 3D point
                    T_W = construct_T(R_1, T_1.flatten())
                    # T_W = np.linalg.inv(T_W)
                    p2d = np.array([x, y, 1])
                    s = 1
                    z = 0
                    leftSideMat = np.linalg.inv(R_1) @ np.linalg.inv(K_1) @ p2d

                    rightSideMat = np.linalg.inv(R_1) @ T_1.flatten()
                    s = (z + rightSideMat[2]) / leftSideMat[2]
                    p3d = np.linalg.inv(R_1) @ (s * np.linalg.inv(K_1) @ p2d - T_1.flatten())
                    print(f"3D point: ({p3d[0]:.3f}, {p3d[1]:.3f}, {p3d[2]:.3f})")

                    # check with projection matrix
                    T_1W = construct_T(R_1, T_1.flatten())
                    P = K_1 @ T_1W[:3]  # projection matrix from 3d world space to cam 1 image space

                    p3d_homo = np.ones(4)
                    p3d_homo[:-1] = p3d
                    p2d_recon = P @ p3d_homo
                    p2d_recon /= p2d_recon[-1]
                    print(p2d_recon)
                    print("")

                old_x, old_y = x, y

            win.imshow(frame)
        return


if __name__ == "__main__":
    main()
