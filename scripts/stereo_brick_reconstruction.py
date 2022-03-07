#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : stereo_brick_reconstruction.py
@Project   : BrickScanner
@Time      : 07.03.22 17:53
@Author    : flowmeadow
"""
from definitions import *
import cv2
from lib.features import get_point_pairs
import numpy as np


def main():
    directory_name = "test_01"  # "epipolar"
    image_path = f"{IMG_DIR}/{directory_name}"

    cv2.namedWindow("frame")
    ref_left, ref_right = None, None
    for file_name in sorted(os.listdir(f"{image_path}/left")):
        # if ref_left is None and ref_right is None:
        #     ref_left = cv2.imread(f"{image_path}/left/{file_name}")
        #     ref_right = cv2.imread(f"{image_path}/right/{file_name}")
        #     continue
        # else:
        img_left = cv2.imread(f"{image_path}/left/{file_name}")
        img_right = cv2.imread(f"{image_path}/right/{file_name}")

        # for img, ref in zip([img_left, img_right], [ref_left, ref_right]):
        #     # prepare mask
        #     mask = cv2.subtract(img, ref)
        #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #     ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        #     kernel = np.ones((6, 6), np.uint8)
        #     for i in range(4):
        #         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #
        #     img = cv2.bitwise_and(img, img, mask=mask)
        #

        # TODO: find feature points
        pts_left, pts_right = get_point_pairs(img_left, img_right)

        img_l, img_r = img_left.copy(), img_right.copy()
        for pt_l, pt_r in zip(pts_left, pts_right):
            color = [int(np.random.rand() * 250) for _ in range(3)]
            img_l = cv2.circle(img_l, pt_l, 5, color, 2)
            img_r = cv2.circle(img_r, pt_r, 5, color, 2)
        frame = cv2.hconcat((img_l, img_r))
        frame = cv2.resize(frame, (1760, 720))
        cv2.imshow("frame", frame)
        cv2.waitKey()

        # TODO: thin out using F
        F = np.load(f"{SETUP_DIR}/F_mat.npy")
        thresh = 0.1
        new_l, new_r = [], []
        for p_l, p_r in zip(pts_left, pts_right):
            p_l_tmp = np.array([p_l[0], p_l[1], 1])
            p_r_tmp = np.array([p_r[0], p_r[1], 1])
            val = np.abs(p_l_tmp.T @ F @ p_r_tmp)
            if val < thresh:
                new_l.append(p_l)
                new_r.append(p_r)
        pts_left, pts_right = np.array(new_l), np.array(new_r)

        img_l, img_r = img_left.copy(), img_right.copy()
        for pt_l, pt_r in zip(pts_left, pts_right):
            color = [int(np.random.rand() * 250) for _ in range(3)]
            img_l = cv2.circle(img_l, pt_l, 5, color, 2)
            img_r = cv2.circle(img_r, pt_r, 5, color, 2)
        frame = cv2.hconcat((img_l, img_r))
        frame = cv2.resize(frame, (1760, 720))
        cv2.imshow("frame", frame)
        cv2.waitKey()

        # TODO: compute R, t
        # TODO: get point cloud
        # TODO: merge point cloud of every image pair


if __name__ == "__main__":
    main()
