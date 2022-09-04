#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : class that handles window interaction e.g. mouse events
@File      : interactive_window.py
@Project   : BrickScanner
@Time      : 07.03.22 18:13
@Author    : flowmeadow
"""

import cv2
import numpy as np


class InteractiveWindow(object):
    """
    class that handles window interaction e.g. mouse events
    """

    def __init__(self, name: str):
        """
        Initialize window
        :param name: window name
        """
        self.name = name
        self.mouse_pos_x, self.mouse_pos_y = None, None
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.get_mouse_pos)

    def get_mouse_pos(self, event: int, x: int, y: int, *_):
        """
        Updates mouse coordinates
        :param event: window event ID
        :param x: current x-coordinate of mouse
        :param y: current x-coordinate of mouse
        """
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.mouse_pos_x, self.mouse_pos_y = x, y

    def imshow(self, frame: np.ndarray):
        """
        Display the given frame
        :param frame: image frame (h, w)
        """
        cv2.imshow(self.name, frame)

    def reset_mouse(self):
        """
        Resets mouse coordinates
        """
        self.mouse_pos_x, self.mouse_pos_y = None, None

    def mouse_clicked(self) -> bool:
        """
        Returns True, if the mouse was clicked since the last reset
        """
        return self.mouse_pos_x is not None and self.mouse_pos_y is not None

    def __del__(self):
        """
        Cleanup
        """
        cv2.destroyAllWindows()
