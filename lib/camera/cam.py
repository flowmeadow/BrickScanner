#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Threaded camera image capturing object
@File      : cam.py
@Project   : BrickScanner
@Time      : 28.03.22 11:36
@Author    : flowmeadow
"""

from copy import deepcopy
from queue import Queue
from threading import Thread

import cv2 as cv
import numpy as np


class Cam:
    """
    Description: Threaded camera image capturing object
    """

    _size = (1280, 720)  # default frame size

    def __init__(self, cam_id: int, queue_size: int = 3):
        """
        Based on: https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
        :param cam_id: system id of the desired camera
        :param queue_size: size of the image queue
        """
        self._cam_id = cam_id
        # if True, thread will be stopped
        self._stopped = False
        # initialize the queue used to store frames read from camera
        self._queue = Queue(maxsize=queue_size)
        # initialize thread that reads camera images
        self._thread = Thread(target=self._read_images, args=())

    def start(self):
        """
        Starts the thread
        """
        # start a thread to read frames from the file video stream
        self._thread.daemon = True
        self._thread.start()

    def _read_images(self):
        """
        Thread loop to read camera images and append them to queue
        """
        # initialize camera object
        self._cam = self._init_video_capture(self._cam_id)

        while not self._stopped:
            # check if queue is full
            if self._queue.full():
                self._queue.get()
            # read the next frame from camera
            grabbed, cam_frame = self._cam.read()

            # check if camera is running
            if not grabbed:
                raise ConnectionError("Lost connection to camera")

            # add the frame to the queue
            self._queue.put(cam_frame)

    def read(self) -> np.array:
        """
        Read the latest camera image from queue
        :return: camera frame or 'None', if 'wait' = 'False'
        """
        # check if thread is still running
        if not self._thread.is_alive():
            raise RuntimeError("Video stream thread stopped working or was not started")
        frame = None
        if self.more():
            frame = deepcopy(self._queue.get())
        return frame

    def more(self) -> bool:
        """
        :return: 'True' if elements are in queue, 'False' otherwise
        """
        # return True if there are still frames in the queue
        return self._queue.qsize() > 0

    def stop(self):
        """
        Stop the thread and release the camera object
        """
        self._stopped = True
        self._thread.join()  # wait for termination
        self._cam.release()

    def _init_video_capture(self, cam_id: int) -> cv.VideoCapture:
        """
        initializes the camera for OpenCV
        :param cam_id: system ID of the camera
        :return: camera object
        """
        camera = cv.VideoCapture(cam_id)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, self._size[0])
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, self._size[1])
        # camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
        camera.set(cv.CAP_PROP_FPS, 5)
        return camera
