#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Calibrates a stereo camera real_setup based on chess boards
@File      : stereo_calibration.py
@Project   : BrickScanner
@Time      : 21.03.22 20:06
@Author    : flowmeadow
"""
import os
import sys

sys.path.append(os.getcwd())  # required to run script from console

from typing import Optional, Tuple

import cv2
import numpy as np
from definitions import *
from lib.camera.stereo_cam import StereoCam
from lib.capturing.calibration import find_chessboard
from lib.helper.cloud_operations import construct_T
from lib.helper.data_management import append_img_pair, new_stereo_img_dir


def calibrate_stereo_setup(
    path: str,
    cell_width: float,
    c_size: Tuple[int, int],
    img_path: Optional[str] = None,
):
    """
    Calibrate a stereo camera real_setup using a checkerboard
    :param path: file_path to store the calibration data
    :param cell_width: length of a checkerboard cell edge in cm
    :param c_size: number of the checkerboards rows and columns
    :param img_path: specify this to load images from a directory
    """
    # Pixel coordinates of checkerboards
    img_points_left = []  # 2d points in image plane.
    img_points_right = []

    # coordinates of the checkerboard in checkerboard world space.
    obj_points = []  # 3d point in real world space

    frame_1, frame_2 = None, None
    if img_path is None:  # record real-time images
        # create image directory
        image_path = new_stereo_img_dir(suffix="calib")

        cam = StereoCam(frame_rate=30, resolution=(320, 240))

        print("Start main loop")
        while True:
            # Capture frame-by-frame
            frame_1, frame_2 = cam.read()
            frame_2 = cv2.rotate(frame_2, cv2.ROTATE_180)  # rotate second frame about 180° (depends on setup)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

            ret, corners_1, corners_2, objp = find_chessboard(frame_1, frame_2, c_size, cell_width)

            if ret:
                tmp_1 = cv2.drawChessboardCorners(frame_1.copy(), c_size, corners_1, ret)
                tmp_2 = cv2.drawChessboardCorners(frame_2.copy(), c_size, corners_2, ret)

                # Display the resulting frame
                frame = cv2.hconcat([tmp_1, tmp_2])
                frame = cv2.resize(frame, (1760, 720))
                cv2.imshow("frame", frame)

                print("Use image pair? (y/n)")
                key = cv2.waitKey()
                if key & 0xFF == ord("y"):
                    append_img_pair(image_path, frame_1, frame_2)
                    print("Saved images")

                    obj_points.append(objp)
                    img_points_left.append(corners_1)
                    img_points_right.append(corners_2)

            # Display the resulting frame
            frame = cv2.hconcat([frame_1, frame_2])
            frame = cv2.resize(frame, (1760, 720))
            cv2.imshow("frame", frame)
        cv2.destroyAllWindows()
    else:
        img_paths = img_path
        for img_path in img_paths:
            print(f"Opening directory {img_path}")
            for file_name in os.listdir(f"{img_path}/left"):
                print(f"Reading image {file_name}")
                frame_1 = cv2.imread(f"{img_path}/left/{file_name}")
                frame_2 = cv2.imread(f"{img_path}/right/{file_name}")

                ret, corners_1, corners_2, objp = find_chessboard(frame_1, frame_2, c_size, cell_width)

                if ret:
                    print(f"Found checkerboard in {file_name}")
                    obj_points.append(objp)
                    img_points_left.append(corners_1)
                    img_points_right.append(corners_2)
                else:
                    print(f"Warning: No checkerboard found in {file_name}")

    print("Start calibration? (y/n)")
    inp = input(">> ")
    if inp != "y":
        return

    print("Calibration of cam 1 ...")
    ret_1, K_1, dist_1, _, _ = cv2.calibrateCamera(obj_points, img_points_left, frame_1.shape[:2], None, None)
    print("Calibration of cam 2 ...")
    ret_2, K_2, dist_2, _, _ = cv2.calibrateCamera(obj_points, img_points_left, frame_2.shape[:2], None, None)

    print("Stereo Calibration ...")
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_left,
        img_points_right,
        K_1,
        dist_1,
        K_2,
        dist_2,
        frame_1.shape[:2],
        # flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print("Error:", ret)

    print("Save data? (y/n)")
    inp = input(">> ")
    if inp != "y":
        return

    np.save(f"{path}/F.npy", F)
    np.save(f"{path}/T_12.npy", np.linalg.inv(construct_T(R, T.flatten())))
    np.save(f"{path}/K_1.npy", K_1)
    np.save(f"{path}/K_2.npy", K_2)
    np.save(f"{path}/dist_1.npy", dist_1)
    np.save(f"{path}/dist_2.npy", dist_2)
    print("Done!")


if __name__ == "__main__":
    # calibration parameter
    cell_width = 0.578  # in cm
    c_size = (6, 8)

    img_path = [f"{IMG_DIR}/real_data/220826-150941_calib"]  # directory of image data
    folder_name = "real_setup/setup_A"
    path = f"{DATA_DIR}/{folder_name}"  # directory of calibration data

    calibrate_stereo_setup(path, cell_width, c_size, img_path=img_path)
