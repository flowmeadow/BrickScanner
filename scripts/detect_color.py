#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Real-time application to identify the color of a brick within a camera image
@File      : detect_color.py
@Project   : BrickScanner
@Time      : 23.08.22 11:25
@Author    : flowmeadow
"""
import os
import sys

sys.path.append(os.getcwd())  # required to run script from console

import cv2
import numpy as np
from lib.camera.stereo_cam import Cam
from lib.capturing.color import compare_hsv_colors, hsv_from_region
from lib.helper.image_operations import fill_sub_region, get_sub_region
from lib.helper.interactive_window import InteractiveWindow


class ColorDetector:
    """
    Application class that handles the color detection in real-time
    """

    win = "ColorDetector"  # window name

    # image frames
    frame = None
    display_frame = None
    checker_frame = np.zeros((720, 720, 3)).astype(np.uint8)

    # ArUco marker parameter
    aruco_marker = {}
    allowed_ids = [8, 9, 10, 11]
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # checkerboard parameter
    checker_size = (6, 6)
    ignore_idcs = [6, 18, 30]
    checker_colors = np.zeros((np.prod(checker_size) - len(ignore_idcs), 3))

    # brick detection parameter
    detection_region = [(363, 136), (552, 255)]  # [None, None]
    reference_region = [(347, 152), (271, 279)]  # [None, None]
    brick_color = np.zeros(3)

    def __init__(self, cam_id: int = 0):
        """
        :param cam_id: Camera ID
        """
        # start camera
        self.cam = Cam(cam_id)
        self.init_cam()

    def init_cam(self):
        """
        Initialize and start camera thread
        """
        self.cam.start()
        print("Wait for camera ...", end="")
        while self.frame is None:
            # Capture frame-by-frame
            self.frame = self.cam.read()
            if self.frame is None:
                continue
        print("\rCamera initialized!")

    def run(self):
        """
        Main loop of the application
        """
        # cv2.namedWindow(self.win, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.display_frame = self.frame.copy()
        msg_text = ""
        win = InteractiveWindow(self.win)
        while True:
            # get current frame
            self.frame = self.cam.read()
            if self.frame is None:
                continue

            # keyboard inputs
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):  # quit app
                break
            elif key & 0xFF == ord("r"):  # reset the detection and reference region
                self.detection_region = [None, None]  # [None, None]
                self.reference_region = [None, None]  # [None, None]

            # draw message
            cv2.putText(self.display_frame, msg_text, (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # display frame
            self.display_frame = cv2.hconcat([self.display_frame, self.checker_frame])
            win.imshow(self.display_frame)
            self.display_frame = self.frame.copy()
            msg_text = ""

            # Update marker
            self.update_marker()
            if len(self.aruco_marker) != 4:
                msg_text = "Marker Detection running ..."
                self.checker_frame = np.zeros(self.checker_frame.shape).astype(np.uint8)
                continue

            # update checkerboard colors
            self.update_checker_colors()

            # get region by double-clicking into the application window
            if win.mouse_clicked():
                m_x, m_y = win.mouse_pos_x, win.mouse_pos_y
                if self.detection_region[0] is None:
                    self.detection_region[0] = (m_x, m_y)
                elif self.detection_region[1] is None:
                    self.detection_region[1] = (m_x, m_y)
                    print(f"Detection region: {self.detection_region}")
                elif self.reference_region[0] is None:
                    self.reference_region[0] = (m_x, m_y)
                elif self.reference_region[1] is None:
                    self.reference_region[1] = (m_x, m_y)
                    print(f"Reference region: {self.reference_region}")
                else:
                    pass
                win.reset_mouse()

            # draw regions
            if self.detection_region[0] is None or self.detection_region[1] is None:
                msg_text = "Select detection area ..."
                continue
            else:
                self.display_frame = cv2.rectangle(self.display_frame, *self.detection_region, (0, 255, 0), 3)
            if self.reference_region[0] is None or self.reference_region[1] is None:
                msg_text = "Select reference area ..."
                continue
            else:
                pass
                # self.display_frame = cv2.rectangle(self.display_frame, *self.reference_region, (255, 0, 0), 3)

            # get brick color
            self.update_brick_color()

            # compare colors
            guess, d_err, r_err = compare_hsv_colors(self.brick_color, self.checker_colors, weights=np.array([1, 1, 1]))
            msg_text = f"Brick color: {np.array(self.brick_color).astype(int)}   Best guess: {guess}"

    def update_marker(self):
        """
        Tries to detect the ArUco markers in the current image and updates them.
        """
        # Update marker
        (corners, ids, _) = cv2.aruco.detectMarkers(self.frame, self.aruco_dict, parameters=self.aruco_params)
        if len(corners) > 0:
            ids = ids.flatten()
            for corner, id in zip(corners, ids):
                if id not in self.allowed_ids:
                    raise NotImplementedError("Wrong ID detected")

                if id not in self.aruco_marker.keys():
                    self.aruco_marker[id] = corner
                else:
                    diff = np.sum(corner - self.aruco_marker[id])
                    if diff > 5:
                        self.aruco_marker = {}
                    else:
                        self.aruco_marker[id] = corner

        # visualize markers
        for id, corner in self.aruco_marker.items():
            # extract the marker corners (which are always returned in top-left, top-right, bottom-right,
            # and bottom-left order)
            corners = corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            # draw box around each marker
            cv2.line(self.display_frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(self.display_frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(self.display_frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(self.display_frame, bottom_left, top_left, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((top_left[0] + bottom_right[0]) / 2.0)
            cY = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(self.display_frame, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the frame
            cv2.putText(
                self.display_frame,
                str(id),
                (top_left[0], top_left[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    def update_checker_colors(self):
        """
        Update the reference colors, given by the checkerboard
        """
        # find inner points of markers
        corners = [self.aruco_marker[id] for id in sorted(self.aruco_marker)]
        points = np.array([corner[0][idx] for (corner, idx) in zip(corners, [2, 3, 0, 1])])
        for p in points:
            cv2.circle(self.display_frame, p.astype(int), 4, (255, 0, 0), -1)

        # rectify region
        frame_size = np.flip(self.checker_frame.shape[:2])
        dst_pts = np.array([[0, 0], [frame_size[0], 0], [*frame_size], [0, frame_size[1]]]).astype(np.float32)
        res_matrix = cv2.getPerspectiveTransform(points, dst_pts)
        self.checker_frame = cv2.warpPerspective(self.frame, res_matrix, frame_size)

        # identify color cells
        offset = 15  # hard coded
        size_fac = 0.6

        rect_thickness = 4

        shift = np.array([0.5 * f / c + o / 2 for f, c, o in zip(frame_size, self.checker_size, (offset, offset))])
        grid = np.meshgrid(
            np.linspace(0, frame_size[0] - offset, self.checker_size[0], endpoint=False),
            np.linspace(0, frame_size[1] - offset, self.checker_size[1], endpoint=False),
        )
        centers = np.moveaxis(grid, 0, -1).reshape(self.checker_size[0] * self.checker_size[1], 2)
        centers += shift

        c_idx = 0
        for idx, center in enumerate(centers):
            if idx in self.ignore_idcs:
                continue
            rect_size = np.array([size_fac * f / c for f, c in zip(frame_size, self.checker_size)])

            # compute top-left and bottom-right corner of the selection area
            c_1, c_2 = center - rect_size / 2, center + rect_size / 2
            c_1, c_2 = c_1.astype(np.uint32), c_2.astype(np.uint32)

            # compute checkerboard color
            self.checker_colors[c_idx, :] = hsv_from_region(self.checker_frame, c_1, c_2)

            # draw
            sub = get_sub_region(self.checker_frame, c_1, c_2)
            sub = 0.5 * sub + 0.5 * np.zeros(sub.shape)
            self.checker_frame = fill_sub_region(self.checker_frame, sub, c_1, c_2)
            self.checker_frame = cv2.rectangle(self.checker_frame, c_1, c_2, (255, 255, 255), thickness=rect_thickness)

            txt_pos = (c_1 + np.array([8, 20])).astype(int)
            cv2.putText(self.checker_frame, str(c_idx), txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            for i, color in enumerate(self.checker_colors[c_idx, :]):
                txt_pos = (c_1 + np.array([30, 20 + 20 * i])).astype(int)
                cv2.putText(
                    self.checker_frame, str(int(color)), txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 2
                )
            c_idx += 1

    def update_brick_color(self):
        """
        Computes the brick color
        TODO: several methods have been tried to identify the pixels belonging to the brick.
              Best so far is K-Clustering
        """
        ref_color = hsv_from_region(self.frame, *self.reference_region)
        brick_frame = get_sub_region(self.frame, *self.detection_region)
        hsv_frame = cv2.cvtColor(brick_frame, cv2.COLOR_BGR2HSV)
        rgb_frame = cv2.cvtColor(brick_frame, cv2.COLOR_BGR2RGB)

        # SECTION: separation based on reference color
        # offset = np.array([255, 120, 15])
        # lower_bound = ref_color - offset
        # upper_bound = ref_color + offset
        # mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        # mask = cv2.bitwise_not(mask)
        # for _ in range(3):
        #     kernel = np.ones((3, 3), np.uint8)
        #     mask = cv2.erode(mask, kernel, iterations=1)

        # SECTION: separation based on contour detection
        # gray = cv2.cvtColor(brick_frame, cv2.COLOR_RGB2GRAY)
        # _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        # edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
        # cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        # mask = np.zeros(brick_frame.shape, np.uint8)
        # mask = cv2.drawContours(mask, [cnt], -1, 255, -1)[:, :, 0]
        #
        # kernel = np.ones((3, 3), np.uint8)
        kernel = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ]
        ).astype(np.uint8)
        # for _ in range(3):
        #     mask = cv2.erode(mask, kernel, iterations=1)

        # SECTION: separation based on clustering
        img_arr = rgb_frame.reshape((-1, 3))
        img_arr = np.float32(img_arr)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        attempts = 10
        ret, label, center = cv2.kmeans(img_arr, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        # center = np.uint8(center)

        mask = 255 * label.reshape(brick_frame.shape[:2]).astype(np.uint8)
        if np.count_nonzero(mask) > mask.shape[0] * mask.shape[1] / 2:
            mask = cv2.bitwise_not(mask)
        mask = cv2.erode(mask, kernel, iterations=1)

        c_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        c_mask[:, :, (0, 2)] = 0

        # compute brick color
        # TODO: does OpenCV consider that classical averaging for the hue leads to errors? (See 'hsv_from_region')
        self.brick_color = cv2.mean(hsv_frame, mask=mask)[:-1]
        # tmp = brick_frame.copy()
        # tmp[:, :, :] = rgb_color
        # self.brick_color = cv2.mean(cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV), mask=mask)[:-1]

        c_mask = 0.5 * c_mask + 0.5 * brick_frame
        self.display_frame = fill_sub_region(self.display_frame, c_mask, *self.detection_region)

    def __del__(self):
        """
        Cleanup
        """
        self.cam.stop()


if __name__ == "__main__":
    app = ColorDetector()
    app.run()
