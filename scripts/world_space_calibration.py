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
from lib.capturing.calibration import find_chessboard
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


def get_world_space_poses(frame_1, frame_2, c_size, cell_width, K_1, K_2, dist_1, dist_2, **kwargs):
    ret, corners_1, corners_2, objp = find_chessboard(
        frame_1, frame_2, c_size, cell_width, center_point=(np.array(c_size) - 1) // 2, **kwargs
    )

    if not ret:
        raise ValueError("Could not find chessboard corners")

    ret_1, r_1, t_1 = cv2.solvePnP(objp, corners_1, K_1, dist_1)
    ret_2, r_2, t_2 = cv2.solvePnP(objp, corners_2, K_2, dist_2)
    if not (ret_1 and ret_2):
        raise ValueError("Could not find relative point poses")
    # compute transformation matrices
    R_1, t_1 = rodrigues_vec_to_rotation_mat(r_1.flatten()), t_1.flatten()
    R_2, t_2 = rodrigues_vec_to_rotation_mat(r_2.flatten()), t_2.flatten()

    T_1W = construct_T(R_1, t_1)
    T_2W = construct_T(R_2, t_2)
    return T_1W, T_2W


def check_error(T_1W, T_2W, T_12):
    T_12_check = T_1W @ np.linalg.inv(T_2W)
    err = np.sum(np.abs(T_12[:3, :3] - T_12_check[:3, :3]) / np.abs(T_12[:3, :3]))
    print(f"Summed error between T_12 from stereo calibration and T_1W @ T_W2: {err:.4f}")


def main(path, cell_width, c_size, img_path=None):
    z = 0

    if img_path is None:
        raise NotImplementedError()

    flip_view = False
    img_path = f"{IMG_DIR}/220621-200516_calib"
    T_12 = np.load(f"{path}/T_12.npy")
    K_1 = np.load(f"{path}/K_1.npy")
    K_2 = np.load(f"{path}/K_2.npy")
    dist_1 = np.load(f"{path}/dist_1.npy")
    dist_2 = np.load(f"{path}/dist_2.npy")

    file_name = os.listdir(f"{img_path}/left")[1]
    print(f"Reading image {file_name}")
    frame_1 = cv2.imread(f"{img_path}/left/{file_name}")
    frame_2 = cv2.imread(f"{img_path}/right/{file_name}")

    T_1W, T_2W = get_world_space_poses(
        frame_1, frame_2, c_size, cell_width, K_1, K_2, dist_1, dist_2, flip_view=flip_view
    )
    # check
    check_error(T_1W, T_2W, T_12)
    # program loop
    win = InteractiveWindow("Test")
    flip_images, update = False, False
    old_x, old_y = 0, 0
    while True:
        # keyboard inputs
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):  # quit
            break
        elif key & 0xFF == ord("f"):  # switch between camera 1 and 2
            flip_images, update = not flip_images, True
        elif key & 0xFF == ord("v"):  # switch between camera 1 and 2
            flip_view, update = not flip_view, True
            T_1W, T_2W = get_world_space_poses(
                frame_1, frame_2, c_size, cell_width, K_1, K_2, dist_1, dist_2, flip_view=flip_view, z_value=z
            )

        # update variables
        if not flip_images:
            frame = frame_1.copy()
            T_iW, K = T_1W, K_1
        else:
            frame = frame_2.copy()
            T_iW, K = T_2W, K_2

        # draw coordinate axes
        p_x = np.array([cell_width * 3, 0, 0])
        p_y = np.array([0, cell_width * 3, 0])
        p_z = np.array([0, 0, cell_width * 3])
        for idx, p in enumerate([p_x, p_y, p_z]):
            p_0 = K @ T_iW[:3] @ np.array([*np.zeros(3), 1.0])
            p_0 = (p_0 / p_0[-1])[:-1].astype(int)
            p_2d = K @ T_iW[:3] @ np.array([*p, 1.0])
            p_2d = (p_2d / p_2d[-1])[:-1].astype(int)

            color = np.zeros(3)
            color[2 - idx] = 255
            frame = cv2.line(frame, p_0, p_2d, color, thickness=4)

        # check for mouse button press
        x, y = win.mouse_pos_x, win.mouse_pos_y
        if x is not None and y is not None:
            frame = cv2.circle(frame, (x, y), 4, (0, 0, 255), 1)  # draw circle at mouse pos

            if x != old_x or y != old_y or update:
                update = False
                print("-----------------------------")
                print(f"2D point:\t\t\t ({x}, {y})")

                # compute 3D point
                p2d = np.array([x, y, 1])

                R, t = T_iW[:3, :3], T_iW[:3, -1]
                left_mat = np.linalg.inv(R) @ np.linalg.inv(K) @ p2d
                right_mat = np.linalg.inv(R) @ t
                s = (z + right_mat[2]) / left_mat[2]
                p3d = np.linalg.inv(R) @ (s * np.linalg.inv(K) @ p2d - t.flatten())
                print(f"3D point:\t\t\t ({p3d[0]:.3f}, {p3d[1]:.3f}, {p3d[2]:.3f})")

                # check with projection matrix
                P = K @ T_iW[:3]  # projection matrix from 3d world space to cam 1 image space

                p3d_check = np.array([*p3d, 1.0])
                p2d_check = P @ p3d_check
                p2d_check /= p2d_check[-1]
                print(f"2D point (check):\t ({x}, {y})")

            old_x, old_y = x, y
        win.imshow(frame)

    print("Save data? (y/n)")
    inp = input(">> ")
    if inp != "y":
        return

    np.save(f"{path}/T_W1.npy", np.linalg.inv(T_1W))
    np.save(f"{path}/T_W2.npy", np.linalg.inv(T_2W))
    print("Done!")


if __name__ == "__main__":
    cell_width = 1
    c_size = (6, 8)
    img_path = f"{IMG_DIR}/220621-200516_calib"

    folder_name = "real_setup/setup_A"
    path = f"{DATA_DIR}/{folder_name}"
    main(path, cell_width, c_size, img_path=img_path)
