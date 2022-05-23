#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : video_compression.py
@Project   : BrickScanner
@Time      : 04.05.22 01:48
@Author    : flowmeadow
"""

from moviepy.editor import *


def main():
    # Import everything needed to edit video clips

    # loading video gfg
    clip = VideoFileClip("/home/florian/snap/obs-studio/1284/2022-05-17 18-14-42.mp4")

    # getting subclip
    clip1 = clip.subclip(0)
    clip1 = clip1.without_audio()

    # getting width and height of clip 1
    w1 = clip1.w
    h1 = clip1.h

    print("Width x Height of clip 1 : ", end=" ")
    print(str(w1) + " x ", str(h1))

    print("---------------------------------------")

    # resizing video downsize 50 %
    clip2 = clip1.resize(0.5)

    # getting width and height of clip 1
    w2 = clip2.w
    h2 = clip2.h

    print("Width x Height of clip 2 : ", end=" ")
    print(str(w2) + " x ", str(h2))

    print("---------------------------------------")
    clip2.write_videofile("movie_resized.mp4")
    # showing final clip


if __name__ == "__main__":
    main()
