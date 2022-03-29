#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : recon_point_cloud.py
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
from lib.recon.description import get_descriptors
from lib.recon.epipolar import epipolar_distance
from lib.recon.masking import mask_from_ref
from lib.recon.triangulation import dlt

from scripts.point_cloud import MyScreen


def concat_and_show(frame_1, frame_2):
    frame = cv2.hconcat((frame_1, frame_2))
    frame = cv2.resize(frame, (1760, 720))
    cv2.imshow("frame", frame)
    cv2.waitKey()


def main():
    directory_name = "test_11"  # "epipolar"
    ref_idx = 0
    img_idx = 1

    # load images
    image_path = f"{IMG_DIR}/{directory_name}"
    file_names = sorted(os.listdir(f"{image_path}/left"))
    ref_left = cv2.imread(f"{image_path}/left/{file_names[ref_idx]}")
    ref_right = cv2.imread(f"{image_path}/right/{file_names[ref_idx]}")
    img_left = cv2.imread(f"{image_path}/left/{file_names[img_idx]}")
    img_right = cv2.imread(f"{image_path}/right/{file_names[img_idx]}")

    # TODO: generate mask
    # generate mask
    mask_left = mask_from_ref(img_left, ref_left)
    mask_right = mask_from_ref(img_right, ref_right)
    img_l = cv2.addWeighted(img_left, 0.8, cv2.cvtColor(mask_left, cv2.COLOR_GRAY2BGR), 0.1, 0)
    img_r = cv2.addWeighted(img_right, 0.8, cv2.cvtColor(mask_right, cv2.COLOR_GRAY2BGR), 0.1, 0)
    concat_and_show(img_l, img_r)

    # TODO: find feature points
    # Initiate ORB detector
    kp_left, des_left = get_descriptors(img_left, mask_left)
    kp_right, des_right = get_descriptors(img_right, mask_right)
    img_l = cv2.drawKeypoints(img_left, kp_left, img_l)
    img_r = cv2.drawKeypoints(img_right, kp_right, img_r)
    concat_and_show(img_l, img_r)

    # TODO: find matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_left, des_right)
    matches = sorted(matches, key=lambda x: x.distance)
    print("Found matches:", len(matches))
    frame = cv2.drawMatches(
        img_left, kp_left, img_right, kp_right, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    frame = cv2.resize(frame, (1760, 720))
    cv2.imshow("frame", frame)
    cv2.waitKey()

    # TODO: thin out using F
    F = np.load(f"{SETUP_DIR}/F.npy")
    dists = []
    # errors = []
    for match in matches:
        p_l = kp_left[match.queryIdx].pt
        p_r = kp_right[match.trainIdx].pt
        dist = epipolar_distance(p_r, p_l, F)
        dists.append(dist)
        # errors.append(dist * match.distance)
    dists = np.array(dists)
    # errors = np.array(errors)

    thresh = 0.02
    idcs = np.argwhere(dists < thresh).flatten()
    matches = [matches[idx] for idx in idcs]
    print("Remaining matches:", len(matches))

    frame = cv2.drawMatches(
        img_left, kp_left, img_right, kp_right, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    frame = cv2.resize(frame, (1760, 720))
    cv2.imshow("frame", frame)
    cv2.waitKey()

    new_kp_left, new_kp_right = [], []
    for m in matches:
        new_kp_left.append(kp_left[m.queryIdx])
        new_kp_right.append(kp_right[m.trainIdx])

    kp_left, kp_right = new_kp_left.copy(), new_kp_right.copy()

    # TODO: compute R, t
    K_left = np.load(f"{SETUP_DIR}/K_left.npy")
    K_right = np.load(f"{SETUP_DIR}/K_right.npy")

    R = np.load(f"{SETUP_DIR}/R.npy")
    T = np.load(f"{SETUP_DIR}/T.npy")

    # TODO: get point cloud
    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = K_left @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = K_right @ RT2  # projection matrix for C2

    p3ds = []
    colors = []
    for kp_l, kp_r in zip(kp_left, kp_right):
        _p3d = dlt(P1, P2, kp_l.pt, kp_r.pt)
        p3ds.append(_p3d)
        kp_l, kp_r = np.array(kp_l.pt).astype(int), np.array(kp_r.pt).astype(int)
        c_l = img_left[kp_l[1], kp_l[0], :].astype(float)
        c_r = img_right[kp_r[1], kp_r[0], :].astype(float)
        colors.append((c_l + c_r) / 2)
    p3ds = np.array(p3ds)
    colors = np.flip(np.array(colors), axis=1) / 255

    # TODO: Remove outliers
    mean = np.mean(p3ds, axis=0)

    distances = np.linalg.norm(p3ds - mean, axis=1)
    max_dist = np.max(distances, axis=0)
    idcs = np.where(distances < max_dist / 20)[0]
    max_num = 10000
    if max_num > len(idcs):
        max_num = len(idcs)
    a = np.random.randint(0, len(idcs), max_num)
    idcs = idcs[a]

    p3ds = p3ds[idcs]
    colors = colors[idcs]

    cv2.destroyAllWindows()

    # TODO: interactive point cloud presentation
    min_vals = np.min(p3ds, axis=0)
    for idx, min_v in enumerate(min_vals):
        p3ds[:, idx] -= min_v

    p3ds /= np.max(p3ds)
    demo = MyScreen(points=p3ds, colors=colors, fullscreen=True)
    demo.run()


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
