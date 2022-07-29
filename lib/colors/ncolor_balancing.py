#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : ncolor_balancing.py
@Project   : BrickScanner
@Time      : 25.05.22 15:13
@Author    : flowmeadow
"""
from definitions import *
import cv2
import numpy as np
import skimage.color


def get_truth_colors():
    directory_name = "n-color_matching"  # "epipolar"

    # load images
    image_path = f"{IMG_DIR}/{directory_name}"
    ref_files = sorted(os.listdir(f"{image_path}/reference_colors"))
    test_files = sorted(os.listdir(f"{image_path}/test_colors"))

    ref_colors = []
    print("\nGround truth:")
    for f in ref_files:
        ref_img = cv2.imread(f"{image_path}/reference_colors/{f}")
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_color = np.mean(ref_img, axis=(0, 1))
        print(f"{str(f).ljust(15)}: {ref_color}")
        ref_colors.append(ref_color)

    test_colors = []
    print("\nValidation:")
    for f in test_files:
        test_img = cv2.imread(f"{image_path}/test_colors/{f}")
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_color = np.mean(test_img, axis=(0, 1))
        print(f"{str(f).ljust(15)}: {test_color}")
        test_colors.append(test_color)

    return ref_colors, test_colors, ref_files + test_files


def dist_XYZ(color_1, color_2):
    X_1, Y_1, Z_1 = color_1
    X_2, Y_2, Z_2 = color_2
    return np.sqrt((X_1 / Y_1 - X_2 / Y_2) ** 2 + (Z_1 / Y_1 - Z_2 / Y_2) ** 2)


def balance_color(color, ground_truths, targets):
    # convert to XYZ space
    color = skimage.color.rgb2xyz(color / 255)
    ground_truths = [skimage.color.rgb2xyz(c / 255) for c in ground_truths]
    targets = [skimage.color.rgb2xyz(c / 255) for c in targets]

    # type and value checks
    if len(ground_truths) != len(targets):
        raise ValueError("ground_truths and targets need to have the same size")

    M_A = np.eye(3)  # can be adjusted

    n = len(targets)  # balance with n colors

    # In case the color matches a target color, all weights become 0 except the one for the matching color
    if (color == np.array(targets)).all(1).any():
        idx = np.argwhere((color == np.array(targets)).all(1))[0][0]
        weights = np.zeros(n)
        weights[idx] = 1
    else:
        # compute the weights for each matrix
        dists = []
        for m, target in enumerate(targets):
            d_m = dist_XYZ(color, target)
            dists.append(d_m)
        dists = np.array(dists)

        d_inv = []
        for d_m in dists:
            d_m_inv = np.sum(dists) / d_m
            d_inv.append(d_m_inv)
        d_inv = np.array(d_inv)

        weights = []
        for d_m_inv in d_inv:
            k_m = d_m_inv / np.sum(d_inv)
            weights.append(k_m)
        weights = np.array(weights)

    # compute the matrices
    matrices = []
    for idx, (g, t) in enumerate(zip(ground_truths, targets)):
        S = M_A @ t
        D = M_A @ g
        K = np.array(
            [
                [D[0] / S[0], 0, 0],
                [0, D[1] / S[1], 0],
                [0, 0, D[2] / S[2]],
            ]
        )
        M_m = np.linalg.inv(M_A) @ K @ M_A
        matrices.append(M_m)
    matrices = np.array(matrices)

    # concatenate the matrices with the weights
    M_NCB = np.zeros((3, 3))
    for k_m, M_m in zip(weights, matrices):
        M_NCB += k_m * M_m

    # compute the balanced color and convert it back to RGB
    final_color = M_NCB @ color
    final_color = skimage.color.xyz2rgb(final_color) * 255
    return final_color


def prepare(arr):
    return np.expand_dims(arr, axis=1).astype(np.int32)


if __name__ == "__main__":
    # TODO: skimage.color.rgb2xyz(color / 255)
    ref_c, test_c, labels = get_truth_colors()
    val_c = ref_c + test_c

    noise = 1 + np.random.rand(3) * 0.3
    print("\nTest results:")
    white_idx = labels.index("white.png")
    for idx, c in enumerate(ref_c + test_c):
        nc_c = balance_color(c * noise, ref_c, ref_c * noise)
        white_c = balance_color(c * noise, [ref_c[white_idx]], [ref_c[white_idx] * noise])

        print("_____________")
        print(f"{labels[idx]}:")
        print("Original:", c)
        print("Noised:", c * noise)
        print("white-Balanced", white_c)
        print("nC-Balanced", nc_c)
        print("dist:", dist_RGB(c, c * noise))
        print("white-balanced dist:", dist_RGB(c, white_c))
        print("nC-balanced dist:", dist_RGB(c, nc_c))

