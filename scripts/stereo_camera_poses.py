#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : stereo_camera_poses.py
@Project   : BrickScanner
@Time      : 16.02.22 17:39
@Author    : flowmeadow
"""
import cv2
import matplotlib.colors as mcolors
import numpy as np
from lib.data_management import params_from_json
from matplotlib.colors import to_rgb
from definitions import *


def main():
    # Load the left and right images
    # in gray scale
    img_idx = 0
    directory_name = "epipolar_close_1"  # "epipolar"

    imgLeft = cv2.imread(f"{IMG_DIR}/{directory_name}/left/image_{img_idx}.png")
    imgRight = cv2.imread(f"{IMG_DIR}/{directory_name}/right/image_{img_idx}.png")

    # Detect the SIFT key points and
    # compute the descriptors for the
    # two images
    sift = cv2.SIFT_create()
    keyPointsLeft, descriptorsLeft = sift.detectAndCompute(imgLeft, None)

    keyPointsRight, descriptorsRight = sift.detectAndCompute(imgRight, None)

    # Create FLANN matcher object
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(indexParams, searchParams)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(descriptorsLeft, descriptorsRight, k=2)

    # Apply ratio test
    goodMatches = []
    ptsLeft = []
    ptsRight = []

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.8
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            goodMatches.append([m])
            ptsLeft.append(keyPointsLeft[m.queryIdx].pt)
            ptsRight.append(keyPointsRight[m.trainIdx].pt)
    # -- Show detected matches

    print("Done")

    ptsLeft = np.int32(ptsLeft)
    ptsRight = np.int32(ptsRight)
    F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_LMEDS)

    # We select only inlier points
    ptsLeft = ptsLeft[mask.ravel() == 1]
    ptsRight = ptsRight[mask.ravel() == 1]

    # -- Draw matches
    colors = list(mcolors.TABLEAU_COLORS)
    img_1, img_2 = imgLeft.copy(), imgRight.copy()
    for idx, (p_l, p_r) in enumerate(zip(ptsLeft, ptsRight)):
        color = np.array(to_rgb(colors[idx % 10])) * 255
        cv2.circle(img_1, p_l, radius=10, color=color, thickness=3)
        cv2.circle(img_2, p_r, radius=10, color=color, thickness=3)

    frame = cv2.hconcat((img_1, img_2))
    frame = cv2.resize(frame, (1760, 720))
    cv2.imshow("frame", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

    def drawlines(imgLeft, imgRight, lines, ptsLeft, ptsRight):

        r, c = imgLeft.shape[:2]
        img1 = imgLeft
        img2 = imgRight

        for r, ptLeft, ptRight in zip(lines, ptsLeft, ptsRight):
            color = tuple(np.random.randint(0, 255, 3).tolist())

            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(ptLeft), 5, color, -1)
            img2 = cv2.circle(img2, tuple(ptRight), 5, color, -1)
        return img1, img2

    # Find epilines corresponding to points
    # in right image (second image) and
    # drawing its lines on left image

    linesLeft = cv2.computeCorrespondEpilines(ptsRight.reshape(-1, 1, 2), 2, F)

    linesLeft = linesLeft.reshape(-1, 3)
    img_1, img_2 = imgLeft.copy(), imgRight.copy()
    img5, img6 = drawlines(img_1, img_2, linesLeft, ptsLeft, ptsRight)
    frame = cv2.hconcat((img5, img6))
    frame = cv2.resize(frame, (1760, 720))
    cv2.imshow("frame", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
    #
    # Find epilines corresponding to
    # points in left image (first image) and
    # drawing its lines on right image
    linesRight = cv2.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2), 1, F)
    linesRight = linesRight.reshape(-1, 3)
    img_1, img_2 = imgLeft.copy(), imgRight.copy()
    img4, img3 = drawlines(img_2, img_1, linesRight, ptsRight, ptsLeft)

    frame = cv2.hconcat((img3, img4))
    frame = cv2.resize(frame, (1760, 720))
    cv2.imshow("frame", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Compute R and t from F
    K, dist = params_from_json(f"{CAM_DIR}/params.json")
    E = K.T @ F @ K
    print(E)
    R_1, R_2, t = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3))

    cv2.decomposeEssentialMat(E, R_1, R_2, t)
    print(R_1)
    print(R_2)
    print(t)

    for p_1, p_2 in zip(ptsLeft, ptsRight):
        img_1, img_2 = imgLeft.copy(), imgRight.copy()
        P_1, P_2 = np.ones(3), np.ones(3)
        P_1[:2] = p_1
        P_2[:2] = p_2


if __name__ == "__main__":
    main()
