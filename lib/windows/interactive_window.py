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


class InteractiveWindow(object):
    """
    class that handles window interaction e.g. mouse events
    """
    def __init__(self, name):
        self.name = name
        self.mouse_pos_x, self.mouse_pos_y = None, None
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.get_mouse_pos)

    def __del__(self):
        cv2.destroyAllWindows()

    def get_mouse_pos(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.mouse_pos_x, self.mouse_pos_y = x, y

    def imshow(self, frame):
        cv2.imshow(self.name, frame)

    def reset_mouse(self):
        self.mouse_pos_x, self.mouse_pos_y = None, None

    def mouse_clicked(self):
        return self.mouse_pos_x is not None and self.mouse_pos_y is not None

