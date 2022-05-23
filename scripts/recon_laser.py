#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : recon_laser.py
@Project   : BrickScanner
@Time      : 19.04.22 16:47
@Author    : flowmeadow
"""
import os
import sys

from scipy.interpolate import interp1d

sys.path.append(os.getcwd())  # required to run script from console

import cv2
import numpy as np
from definitions import *
from lib.recon.mesh2cloud import data2cloud, data2mesh, m2c_dist
from lib.recon.triangulation import dlt
from rendering.models.model_generation.geometry import sphere
from transformations.methods import construct_T, rot_mat, rotate_vec
from lib.recon.mesh2cloud import m2c_dist, m2c_dist_rough, display_dist, print_dist
from scripts.point_cloud import CloudScreen
from scripts.stereo_brick import BrickScreen

# GENERAL VARIABLES

IMAGE_PATH = f"{IMG_DIR}/gl_laser_01"
MAX_IMAGES = 20  # 20
GENERATE_NEW = True
DEBUG = True
DEBUG_FRAME = 12
SUB_REGION = [(450, 550), (900, 1000)]
SUB_REGION = [(600, 700), (900, 1000)]

# SUBPIXEL REFINEMENT
SUB_PIXEL_REFINEMENT_1 = True  # weight by value in HSV
KERNEL = np.ones((2, 2), np.uint8)
SUB_PIXEL_REFINEMENT_2 = False  # polynomial fit
FIT_DEGREE = 1
FIT_RANGE = 8  # should be even


def crop(frame):
    return frame[SUB_REGION[0][0] : SUB_REGION[0][1], SUB_REGION[1][0] : SUB_REGION[1][1]]


def concat_and_show(frame_1, frame_2, wait=True):
    frame = cv2.hconcat((frame_1, frame_2))
    frame = cv2.resize(frame, (1760, 720))
    cv2.imshow("frame", frame)
    cv2.waitKey(None if wait else 50)


def concat_and_show_reg(frame_1, frame_2, wait=True, scale=8, points_lst=None):
    frame_1, frame_2 = crop(frame_1), crop(frame_2)
    w, h = (frame_1.shape[1] * scale, frame_1.shape[0] * scale)
    frame_1 = cv2.resize(frame_1, (w, h), interpolation=cv2.INTER_NEAREST)
    frame_2 = cv2.resize(frame_2, (w, h), interpolation=cv2.INTER_NEAREST)

    if points_lst is not None:
        frames = [frame_1, frame_2]
        canvas_lst = []
        for points, frame in zip(points_lst, frames):
            canvas = np.zeros((frame.shape[0], frame.shape[1], 3))
            for i in range(3):
                canvas[:, :, i] = frame
            for p in points:
                p = p - np.array([SUB_REGION[1][0], SUB_REGION[0][0]])
                p *= scale
                p += np.array([scale / 2, scale / 2])
                cv2.circle(canvas, p.astype(int), 3, (0, 0, 255), thickness=-1)
            canvas_lst.append(canvas)
        frame_1, frame_2 = canvas_lst
    frame = cv2.hconcat((frame_1, frame_2))
    cv2.imshow("frame", frame)
    cv2.waitKey(None if wait else 50)


def main():
    # SECTION: define camera 2 pose
    # # load rotation matrix and translation vector
    # scale = 0.01
    # R = np.load(f"{SETUP_DIR}/R.npy")
    # t = np.load(f"{SETUP_DIR}/T.npy").flatten() * scale

    laser_pos = np.array([0.0, 0.1])  # in 2D
    cam_pos = np.array([1.0, 0.0])  # in 2D (of camera 1)
    cam_dist = 2 * 0.1  # distance of cameras in y-direction (world)

    view_w = laser_pos - cam_pos  # view vector in world coordinates
    angle = np.tan(view_w[1] / view_w[0])  # rotation angle of cam 1 in world coordinates
    angle_d = angle * 180 / np.pi  # same angle in degrees

    t = cam_dist * np.array([np.cos(angle), 0.0, np.sin(angle)])  # distance to camera 2 in cam 1 coordinates
    R = rot_mat((0.0, 1.0, 0.0), -2 * angle_d)  # rotation of cam 2 relative to cam 1

    # t = np.array([0.199, 0., -0.0199])
    # R = rot_mat((0., 1., 0.), 2 * 5.71)

    # create transformation matrix from camera system 1 to camera system 2
    # WARNING: no idea why t needs to be negative
    T_cam = construct_T(R, -t)

    # SECTION: Generate image pairs
    # generate images of point cloud in simulator
    if GENERATE_NEW:
        screen = BrickScreen(T_cam, file_path=IMAGE_PATH, max_images=MAX_IMAGES, fullscreen=True)
        screen.run()

    file_names = sorted(os.listdir(f"{IMAGE_PATH}/left"))
    # for name in file_names:
    #     img_left = cv2.imread(f"{image_path}/left/{name}")
    #     img_right = cv2.imread(f"{image_path}/right/{name}")
    #     concat_and_show(img_left, img_right, wait=False)

    # SECTION: Compute F matrix
    S = np.mat([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = S @ R
    K = np.load(f"{IMAGE_PATH}/cam_K.npy")
    K_inv = np.linalg.inv(K)
    F = K_inv.T * (E * K_inv)

    # SECTION: Repeat for each image pair
    p3ds_total = None
    for img_idx, name in enumerate(file_names):
        if DEBUG and img_idx != DEBUG_FRAME:
            continue
        print(f"Processing image {name} ({img_idx + 1}|{len(file_names)})")
        img_left = cv2.imread(f"{IMAGE_PATH}/left/{name}")
        img_right = cv2.imread(f"{IMAGE_PATH}/right/{name}")

        concat_and_show(img_left, img_right, wait=True if DEBUG else False)
        imgs = [img_left, img_right]
        # resize images
        # idcs_left = np.nonzero(img_left)
        # idcs_right = np.nonzero(img_right)
        # idcs = (np.append(idcs_left[0], idcs_right[0]), np.append(idcs_left[1], idcs_right[1]))
        # x_1, x_2, y_1, y_2 = np.min(idcs[0]), np.max(idcs[0]), np.min(idcs[1]), np.max(idcs[1])
        # print(x_1, x_2, y_1, y_2)
        #
        # a = img_left[x_1: x_2, y_1: y_2]
        # b = img_right[x_1: x_2, y_1: y_2]
        #
        # concat_and_show(*imgs)
        #

        # SECTION: get mask of laser line
        masks = []
        for img in imgs:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_red1 = cv2.inRange(hsv, (0, 1, 0), (20, 255, 255))
            mask_red2 = cv2.inRange(hsv, (160, 1, 0), (180, 255, 255))
            mask = (mask_red1 + mask_red2) / 255

            if SUB_PIXEL_REFINEMENT_1:
                mask = cv2.dilate(mask, KERNEL)
                hsv = hsv / 255
                mask = mask * hsv[:, :, 2]

            masks.append(mask)

        if DEBUG:
            concat_and_show_reg(*[mask for mask in masks])
        # SECTION: find points
        key_pts_lst = []
        for img, mask in zip(imgs, masks):
            x_pos = np.arange(img.shape[1]) + 1

            pixel_pos = x_pos * mask
            pixel_counts = np.sum(mask, axis=1)
            y_idcs = np.nonzero(pixel_counts)[0]
            x_coords = np.sum(pixel_pos[y_idcs], axis=1) / pixel_counts[y_idcs] - 1

            if SUB_PIXEL_REFINEMENT_2 and y_idcs.shape[0] > 0:
                new_x_coords = [x_coords[i] for i in range(FIT_RANGE // 2)]
                for idx in range(x_coords.shape[0] - FIT_RANGE):
                    params = np.polyfit(y_idcs[idx : idx + FIT_RANGE], x_coords[idx : idx + FIT_RANGE], FIT_DEGREE)
                    fit_fun = np.poly1d(params)
                    new_x_coords.append(fit_fun(y_idcs[idx + FIT_RANGE // 2]))
                for i in range(FIT_RANGE // 2):
                    new_x_coords.append(x_coords[-(FIT_RANGE // 2 - i)])
                x_coords = np.array(new_x_coords)

            coords = np.array([x_coords, y_idcs]).T
            key_pts_lst.append(coords)

        if key_pts_lst[0].shape[0] == 0 or key_pts_lst[1].shape[0] == 0:
            print(f"WARNING: no keypoints found for image pair {name}")
            continue

        # show points
        if DEBUG:
            concat_and_show_reg(*[mask for mask in masks], points_lst=key_pts_lst)

        # SECTION: define epilines
        lines_lst = []
        for idx, (img, key_pts) in enumerate(zip(imgs, key_pts_lst)):
            lines = cv2.computeCorrespondEpilines(key_pts, idx + 1, F)
            lines_lst.append(lines)
        lines_lst.reverse()  # reverse lines as they were computed for the other image
        # draw lines
        # frames = [img_left.copy(), img_right.copy()]
        # for frame, lines in zip(frames, lines_lst):
        #     print("______________")
        #     c = frame.shape[1]
        #     for line in lines:
        #         line = line.flatten()
        #         if line[1] == 0:
        #             continue
        #         # draw line to frame
        #         x0, y0 = map(int, [0, -line[2] / line[1]])
        #         x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        #         frame = cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
        # concat_and_show(*frames)

        pt_pairs_lst = []
        base_pts_list = key_pts_lst.copy()
        base_pts_list.reverse()
        for idx, (lines, key_pts, base_pts) in enumerate(zip(lines_lst, key_pts_lst, base_pts_list)):
            lines = np.squeeze(lines)

            # SECTION: define bounding box (BB)
            # x range of key points
            x_min = np.min(key_pts[:, 0])
            x_max = np.max(key_pts[:, 0])

            # define bounding boxes for each line (ax + by + c = 0 => y = -(ax + c) / b)
            # the bounding box for line 'i' is defined as ([x_min, x_max], [y[i, 0], y[i, 1])

            y_1 = -(lines[:, 0] * np.array(x_min) + lines[:, 2]) / lines[:, 1]
            y_2 = -(lines[:, 0] * np.array(x_max) + lines[:, 2]) / lines[:, 1]
            y = np.array([y_1, y_2]).T
            y = np.sort(y, axis=1)  # sort values, so lower y value is at index 0

            pt_pairs = []
            for i in range(lines.shape[0]):
                # SECTION: find points in BB
                # add a threshold to enlarge the BB a bit
                thresh = 2  # threshold in pixels

                ll = np.array([x_min, y[i, 0] - thresh])  # lower-left
                ur = np.array([x_max, y[i, 1] + thresh])  # upper-right
                in_idx = np.all(np.logical_and(ll <= key_pts, key_pts <= ur), axis=1)
                in_pts = key_pts[in_idx]

                # SECTION: find corresponding points
                if in_pts.shape[0] <= 1:  # no point in BB
                    continue
                elif in_pts.shape[0] >= 2:  # more than one point
                    p_1 = np.array([x_min, y[i, 0]])
                    p_2 = np.array([x_max, y[i, 1]])

                    line_dists = []
                    for p in in_pts:
                        line_dist = np.cross(p_2 - p_1, p - p_1) / np.linalg.norm(p_2 - p_1)
                        line_dists.append(line_dist)
                    line_dists = np.array(line_dists)

                    if not (
                        np.any(line_dists < 0) and np.any(line_dists > 0)
                    ):  # a point on both sides of the line is required
                        continue
                    # find the closest point on both sides of the line
                    max_neg_idc = np.where(line_dists < 0, line_dists, -np.inf).argmax()
                    min_pos_idc = np.where(line_dists > 0, line_dists, np.inf).argmin()
                    key_p_1 = in_pts[max_neg_idc, :]
                    key_p_2 = in_pts[min_pos_idc, :]

                    def get_intersect(a1, a2, b1, b2):
                        """
                        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
                        a1: [x, y] a point on the first line
                        a2: [x, y] another point on the first line
                        b1: [x, y] a point on the second line
                        b2: [x, y] another point on the second line
                        """
                        s = np.vstack([a1, a2, b1, b2])  # s for stacked
                        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
                        l1 = np.cross(h[0], h[1])  # get first line
                        l2 = np.cross(h[2], h[3])  # get second line
                        x, y, z = np.cross(l1, l2)  # point of intersection
                        assert z != 0, "Lines are not allowed to be parallel"
                        return np.array([x / z, y / z])

                    intersect = get_intersect(p_1, p_2, key_p_1, key_p_2)
                    pt_pairs.append([base_pts[i, :], intersect])
            pt_pairs_lst.append(np.array(pt_pairs))

        # SECTION: reconstruct point cloud
        K = np.load(f"{IMAGE_PATH}/cam_K.npy")

        RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        RT2 = np.concatenate([R, np.array([t]).T], axis=-1)

        P1 = K @ RT1  # projection matrix for C1
        P2 = K @ RT2  # projection matrix for C2

        # kp_left = np.append(pt_pairs_lst[0][:, 1, :], pt_pairs_lst[1][:, 0, :], axis=0)
        # kp_right = np.append(pt_pairs_lst[0][:, 0, :], pt_pairs_lst[1][:, 1, :], axis=0)
        kp_left = pt_pairs_lst[0][:, 1, :]
        kp_right = pt_pairs_lst[0][:, 0, :]

        p3ds = []
        for kp_l, kp_r in zip(kp_left, kp_right):
            # compute with DLT
            # _p3d = dlt(P1, P2, kp_l.pt, kp_r.pt)

            # compute with OCV
            _p3d = cv2.triangulatePoints(P1, P2, kp_l, kp_r).flatten()
            _p3d = _p3d[:-1] / _p3d[-1]

            p3ds.append(_p3d)
        p3ds = np.array(p3ds)

        # SECTION: transform point cloud back to world coordinates
        # transform point cloud based on cam pose
        T_cam = np.load(f"{IMAGE_PATH}/cam_T.npy")
        # WARNING: No idea why rotation part needs to be inverted
        T_cam[:3, :3] *= -1
        p3ds = np.concatenate([p3ds.T, [np.ones(p3ds.shape[0])]])
        p3ds = T_cam @ p3ds
        p3ds = p3ds.T[:, :-1]

        d = 0.2 / len(file_names)  # shift point cloud according to belt speed (here it is 0.2 in n images)
        p3ds[:, 1] -= (img_idx + 2) * d  # WARNING: Bug here. Object was shifted twice before first image was rendered

        if p3ds_total is None:
            p3ds_total = p3ds.copy()
        else:
            p3ds_total = np.append(p3ds_total, p3ds, axis=0)
        print("Num Points:", p3ds_total.shape[0])

    # SECTION: interactive point cloud presentation
    model_name = "skull"
    np.save(f"{MODEL_DIR}/{model_name}_pc.npy", p3ds_total)
    demo = CloudScreen(points=p3ds_total, fullscreen=True, model="skull")
    demo.run()

    # SECTION: Evaluate quality
    # load model and reconstructed point cloud
    model_name = "skull"
    mesh_data = np.load(f"{MODEL_DIR}/{model_name}.npz")

    # change model size and orientation to match with cloud
    mesh_data = dict(mesh_data)
    scale = 0.01
    mesh_data["vertices"] *= scale
    mesh_data["vertices"] = rotate_vec(mesh_data["vertices"], (0.0, 0.0, 1.0), 90)

    # convert to open3d
    mesh = data2mesh(mesh_data)
    cloud = data2cloud(p3ds_total)

    # compute distance
    dist = m2c_dist(mesh, cloud)

    # print distance results
    print_dist(-1, dist)
    display_dist(dist, cloud, mesh)


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
