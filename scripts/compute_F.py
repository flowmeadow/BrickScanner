#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : compute_F.py
@Project   : BrickScanner
@Time      : 07.03.22 18:03
@Author    : flowmeadow
"""
import cv2
import matplotlib.colors as mcolors
import numpy as np
from definitions import *
from lib.windows.interactive_window import InteractiveWindow
from matplotlib.colors import to_rgb

DISPLAY_RES = (1760, 720)


def main():
    directory_name = "calibration_5"

    image_path = f"{IMG_DIR}/{directory_name}"
    new_points_right, new_points_left = [], []

    win = InteractiveWindow("frame")
    old_F = np.load(f"{SETUP_DIR}/F_mat.npy")

    for file_name in sorted(os.listdir(f"{image_path}/left")):
        imgLeft = cv2.imread(f"{image_path}/left/{file_name}")
        imgRight = cv2.imread(f"{image_path}/right/{file_name}")

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
        ptsLeft = np.int32(ptsLeft)
        ptsRight = np.int32(ptsRight)
        F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_LMEDS)
        # We select only inlier points
        ptsLeft = ptsLeft[mask.ravel() == 1]
        ptsRight = ptsRight[mask.ravel() == 1]

        # -- Draw matches
        colors = list(mcolors.TABLEAU_COLORS)

        # show all
        # thresh = 0.5
        # new_l, new_r = [], []
        # for p_l, p_r in zip(ptsLeft, ptsRight):
        #     p_l_tmp = np.array([p_l[0], p_l[1], 1])
        #     p_r_tmp = np.array([p_r[0], p_r[1], 1])
        #     if np.abs(p_l_tmp @ old_F @ p_r_tmp.T) < thresh:
        #         new_l.append(p_l)
        #         new_r.append(p_r)
        # ptsLeft, ptsRight = np.array(new_l), np.array(new_r)

        skip = False
        while True:
            img_1, img_2 = imgLeft.copy(), imgRight.copy()
            for idx, (p_l, p_r) in enumerate(zip(ptsLeft, ptsRight)):
                color = np.array(to_rgb(colors[idx % 10])) * 255
                cv2.circle(img_1, p_l, radius=10, color=color, thickness=3)
                cv2.circle(img_2, p_r, radius=10, color=color, thickness=3)

            frame = cv2.hconcat((img_1, img_2))
            frame = cv2.resize(frame, (1760, 720))
            win.imshow(frame)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                return
            if key & 0xFF == ord("s"):
                skip = True
                break
            if key & 0xFF == ord("y"):
                break

            if win.mouse_clicked():
                x, y = win.mouse_pos_x, win.mouse_pos_y
                if x < DISPLAY_RES[0] / 2:
                    x = int(x * 2 * IMAGE_RES[0] / DISPLAY_RES[0])
                    y = int(y * IMAGE_RES[1] / DISPLAY_RES[1])
                    img_side = "left"
                else:
                    x = int((x - DISPLAY_RES[0] / 2) * 2 * IMAGE_RES[0] / DISPLAY_RES[0])
                    y = int(y * IMAGE_RES[1] / DISPLAY_RES[1])
                    img_side = "right"

                # find closest point to mouse_click
                pts = ptsLeft.copy() if img_side == "left" else ptsRight.copy()
                pt_clicked = np.full(pts.shape, np.array([x, y]))
                diff = (pts - pt_clicked) ** 2
                my_sum = np.sum(diff, axis=1)
                idx = np.argmin(my_sum)
                ptsLeft = np.delete(ptsLeft, idx, 0)
                ptsRight = np.delete(ptsRight, idx, 0)
                win.reset_mouse()

        if not skip:
            for p_l, p_r in zip(ptsLeft, ptsRight):
                new_points_left.append(p_l)
                new_points_right.append(p_r)

    ptsLeft, ptsRight = np.array(new_points_left), np.array(new_points_right)
    F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_LMEDS)

    def drawlines(imgLeft, imgRight, lines, ptsLeft, ptsRight):

        r, c = imgLeft.shape[:2]
        img1 = imgLeft
        img2 = imgRight

        for r, ptLeft, ptRight in zip(lines, ptsLeft, ptsRight):
            color = tuple(np.random.randint(0, 255, 3).tolist())

            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            # img1 = cv2.circle(img1, tuple(ptLeft), 5, color, -1)
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

    F_path = f"{SETUP_DIR}/F_mat.npy"

    print(f"Computed F: \n{F}")

    inp = input("Save F? (y/n)\n>>")
    if inp == "y":
        np.save(F_path, F)
        print("F saved")
    else:
        print("F not saved")


if __name__ == "__main__":
    main()
