#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Calibrates a stereo camera setup based on chess boards
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
from lib.data_management import append_img_pair, new_stereo_img_dir

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


def find_checkerboard(
    frame_1: np.array,
    frame_2: np.array,
    c_size: Tuple[int, int],
    cell_width: float,
) -> Tuple[bool, np.array, np.array, np.array]:
    """
    Looks for checkerboard corners in images and computes the 3D coordinates of these corners
    :param frame_1: left image frame
    :param frame_2: right image frame
    :param c_size: dimension of the chess corner grid
    :param cell_width: width of a chess cell square in mm
    :return: Tuple of 4 elements containing:
        a boolean return value (only true if checkerboard was detected in both frames)
        the 2D corner coordinates for frame_1
        the 2D corner coordinates for frame_2
        the 3D corner coordinates
    """
    ret = False

    # Convert images into grayscale
    gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    # Find 2D corners
    c_ret_1, corners_1 = cv2.findChessboardCorners(gray_1, c_size, None)
    c_ret_2, corners_2 = cv2.findChessboardCorners(gray_2, c_size, None)

    # If corners found in both images ...
    if c_ret_1 and c_ret_2:
        ret = True

        # refine corner positions
        corners_1 = cv2.cornerSubPix(gray_1, corners_1, (11, 11), (-1, -1), CRITERIA)
        corners_2 = cv2.cornerSubPix(gray_2, corners_2, (11, 11), (-1, -1), CRITERIA)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((c_size[0] * c_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : c_size[0], 0 : c_size[1]].T.reshape(-1, 2)
    objp = cell_width * objp

    return ret, corners_1, corners_2, objp


def calibrate_stereo_setup(
    cell_width: int,
    c_size: Tuple[int, int],
    img_path: Optional[str] = None,
):
    """
    Calibrate a stereo camera setup using a checkerboard
    :param c_size: dimension of the chess corner grid
    :param cell_width: width of a chess cell square in mm
    :param img_path: specify this to load images from a directory
    """
    # Pixel coordinates of checkerboards
    img_points_left = []  # 2d points in image plane.
    img_points_right = []

    # coordinates of the checkerboard in checkerboard world space.
    obj_points = []  # 3d point in real world space

    frame_1, frame_2 = None, None
    if img_path is None:
        # create image directory
        image_path = new_stereo_img_dir(suffix="calib")

        cam = StereoCam()

        print("Start main loop")
        while True:
            # Capture frame-by-frame
            frame_1, frame_2 = cam.read()

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

            ret, corners_1, corners_2, objp = find_checkerboard(frame_1, frame_2, c_size, cell_width)

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
        print(f"Opening directory {img_path}")
        for file_name in os.listdir(f"{img_path}/left"):
            print(f"Reading image {file_name}")
            frame_1 = cv2.imread(f"{img_path}/left/{file_name}")
            frame_2 = cv2.imread(f"{img_path}/right/{file_name}")

            ret, corners_1, corners_2, objp = find_checkerboard(frame_1, frame_2, c_size, cell_width)

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
        K_1,
        dist_1,
        frame_1.shape[:2],
        criteria=CRITERIA,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print("Error:", ret)

    print("Save calibration data? (y/n)")
    inp = input(">> ")
    if inp != "y":
        return

    np.save(f"{SETUP_DIR}/F.npy", F)
    np.save(f"{SETUP_DIR}/R.npy", R)
    np.save(f"{SETUP_DIR}/T.npy", T)
    np.save(f"{SETUP_DIR}/K_left.npy", K_1)
    np.save(f"{SETUP_DIR}/K_right.npy", K_2)
    np.save(f"{SETUP_DIR}/dist_left.npy", dist_1)
    np.save(f"{SETUP_DIR}/dist_right.npy", dist_2)


if __name__ == "__main__":
    cell_width = 11.97
    c_size = (6, 8)
    # img_path = f"{IMG_DIR}/220322-002703_calib"
    img_path = None

    calibrate_stereo_setup(cell_width, c_size, img_path=img_path)
    print("Done!")
