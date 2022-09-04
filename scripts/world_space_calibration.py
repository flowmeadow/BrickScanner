#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Methods to obtain the world poses from calibration images, which show the checkerboard pattern
@File      : world_space_calibration.py
@Project   : BrickScanner
@Time      : 06.08.22 14:54
@Author    : flowmeadow
"""
from typing import Optional, Tuple

import cv2
import numpy as np
from definitions import *
from lib.capturing.calibration import get_world_space_poses
from lib.helper.interactive_window import InteractiveWindow
from lib.recon.keypoints import get_line_points


def check_error(T_1W: np.ndarray, T_2W: np.ndarray, T_12: np.ndarray) -> float:
    """
    Computes and displays the error between the relative pose T_12 obtained from calibration and world poses
    :param T_1W: transformation matrix of camera 1 in world space (4, 4)
    :param T_2W: transformation matrix of camera 2 in world space (4, 4)
    :param T_12: transformation matrix from camera 1 to camera 2 (4, 4)
    :return: computed summed error
    """
    T_12_check = T_1W @ np.linalg.inv(T_2W)
    err = np.sum(np.abs(T_12[:3, :3] - T_12_check[:3, :3]) / np.abs(T_12[:3, :3]))
    print(f"Summed error between T_12 from stereo calibration and T_1W @ T_W2: {err:.4f}")
    return err


def world_calib_app(
    path: str,
    cell_width: float,
    c_size: Tuple[int, int],
    center_point=(0, 0),
    z_value: float = 0.0,
    frame_idx: int = 0,
    ref_idx: int = 0,
    img_path: Optional[str] = None,
):
    """
    Application to obtain the world poses from calibration images, which show the checkerboard pattern
    :param path: directory of calibration data
    :param cell_width: length of a checkerboard cell edge in cm
    :param c_size: number of the checkerboards rows and columns
    :param center_point: checkerboard corner indices that define the world space origin
    :param z_value: z coordinate of the checkerboard plane in world space
    :param frame_idx: index of images used for checkerboard corner detection (only if img_path is not None)
    :param ref_idx: index of images used for visualization (only if img_path is not None)
    :param img_path: directory of images used for calibration
    """

    # SECTION: Initialization
    if img_path is None:
        # TODO: implement world space calibration on real time footage
        raise NotImplementedError()

    # load calibration parameter
    T_12 = np.load(f"{path}/T_12.npy")
    K_1 = np.load(f"{path}/K_1.npy")
    K_2 = np.load(f"{path}/K_2.npy")
    dist_1 = np.load(f"{path}/dist_1.npy")
    dist_2 = np.load(f"{path}/dist_2.npy")

    # define calibration images
    file_name = os.listdir(f"{img_path}/left")[frame_idx]  # image used for checkerboard detection
    ref_name = os.listdir(f"{img_path}/left")[ref_idx]  # image used for visualization

    print(f"Reading image {file_name} ...")
    # read images for checkerboard detection
    frame_1 = cv2.imread(f"{img_path}/left/{file_name}")
    frame_2 = cv2.imread(f"{img_path}/right/{file_name}")
    # read images for visualization
    ref_1 = cv2.imread(f"{img_path}/left/{ref_name}")
    ref_2 = cv2.imread(f"{img_path}/right/{ref_name}")

    # find world space poses based on detected checkerboard
    T_1W, T_2W = get_world_space_poses(
        frame_1,
        frame_2,
        c_size,
        cell_width,
        K_1,
        K_2,
        dist_1,
        dist_2,
        flip_view=False,
        z_value=z_value,
        center_point=center_point,
    )

    # compute and print error with relative pose T_12
    check_error(T_1W, T_2W, T_12)

    win = InteractiveWindow("Test")  # init window
    flip_view = False  # rotates the world coordinate system by 180°, if True
    flip_images = False  # switch from left to right image, if True
    update = False  # update 2D to 3D point computation, if True
    old_x, old_y = 0, 0  # stores last mouse click position

    while True:  # start program loop
        # SECTION: Keyboard Inputs
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):  # quit
            break
        elif key & 0xFF == ord("f"):  # switch between camera 1 and 2
            flip_images, update = not flip_images, True
        elif key & 0xFF == ord("v"):  # rotate coordinate system by 180°
            flip_view, update = not flip_view, True
            T_1W, T_2W = get_world_space_poses(
                frame_1,
                frame_2,
                c_size,
                cell_width,
                K_1,
                K_2,
                dist_1,
                dist_2,
                flip_view=flip_view,
                z_value=z_value,
                center_point=center_point,
            )

        # SECTION: Update Images and Frames
        if not flip_images:
            frame = ref_1.copy()
            T_iW, K = T_1W, K_1
        else:
            frame = ref_2.copy()
            T_iW, K = T_2W, K_2

        # SECTION: Draw coordinate system
        # define coordinate axes in world space
        p_x = np.array([cell_width * 3, 0, z_value])
        p_y = np.array([0, cell_width * 3, z_value])
        p_z = np.array([0, 0, cell_width * 3 + z_value])
        # compute origin in image space
        p_0 = K @ T_iW[:3] @ np.array([*np.array([0, 0, z_value]), 1.0])
        p_0 = (p_0 / p_0[-1])[:-1].astype(int)
        for idx, p in enumerate([p_x, p_y, p_z]):
            # compute axis end points in image space
            p_2d = K @ T_iW[:3] @ np.array([*p, 1.0])
            p_2d = (p_2d / p_2d[-1])[:-1].astype(int)
            # draw colored line
            color = np.zeros(3)
            color[2 - idx] = 255
            frame = cv2.line(frame, p_0, p_2d, color, thickness=4)

        # SECTION: Handle Mouse Click Event
        x, y = win.mouse_pos_x, win.mouse_pos_y
        if x is not None and y is not None:
            frame = cv2.circle(frame, (x, y), 4, (0, 0, 255), 1)  # draw circle at mouse pos

            if x != old_x or y != old_y or update:
                # print new mouse position
                update = False
                print("-----------------------------")
                print(f"2D point:\t\t\t ({x}, {y})")

                # compute corresponding 3D point for a given z value in world space
                p2d = np.array([x, y, 1])
                R, t = T_iW[:3, :3], T_iW[:3, -1]
                left_mat = np.linalg.inv(R) @ np.linalg.inv(K) @ p2d
                right_mat = np.linalg.inv(R) @ t
                s = (z_value + right_mat[2]) / left_mat[2]  # scale factor defining the 3d points distance to the camera
                p3d = np.linalg.inv(R) @ (s * np.linalg.inv(K) @ p2d - t.flatten())
                print(f"3D point:\t\t\t ({p3d[0]:.3f}, {p3d[1]:.3f}, {p3d[2]:.3f})")

                # check with projection matrix
                P = K @ T_iW[:3]  # projection matrix from 3d world space to cam 1 image space
                p3d_check = np.array([*p3d, 1.0])
                p2d_check = P @ p3d_check
                p2d_check /= p2d_check[-1]
                print(f"2D point (check):\t ({x}, {y})")

            old_x, old_y = x, y  # update mouse click position

        # display image
        win.imshow(frame)

    # SECTION: Compute laser line position
    P_1 = K_1 @ T_1W[:3]  # projection matrix from 3d world space to cam 1 image space
    P_2 = K_2 @ T_2W[:3]  # projection matrix from 3d world space to cam 2 image space

    # compute positions of the laser line on the belt at the upper and lower image edge
    line_1 = get_line_points(P_1, frame_1.shape[:2], z_value=z_value)
    line_2 = get_line_points(P_2, frame_2.shape[:2], z_value=z_value)
    ref_1 = cv2.line(ref_1, *line_1, 255, 3)
    ref_2 = cv2.line(ref_2, *line_2, 255, 3)

    # show projected laser line
    win.imshow(ref_1)
    cv2.waitKey()
    win.imshow(ref_2)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # SECTION: Save world poses
    print("Save data? (y/n)")
    inp = input(">> ")
    if inp != "y":
        return
    np.save(f"{path}/T_W1.npy", np.linalg.inv(T_1W))
    np.save(f"{path}/T_W2.npy", np.linalg.inv(T_2W))
    print("Done!")


if __name__ == "__main__":
    # calibration parameter
    cell_width = 0.578
    c_size = (8, 6)
    center_point = (5, 2)
    ref_idx = 1

    img_path = f"{IMG_DIR}/real_data/220826-152549_calib_world"  # directory of image data
    folder_name = "real_setup/setup_A"
    path = f"{DATA_DIR}/{folder_name}"  # directory of calibration data

    world_calib_app(path, cell_width, c_size, center_point=center_point, ref_idx=ref_idx, img_path=img_path)
