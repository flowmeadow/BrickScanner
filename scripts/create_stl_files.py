#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Converts all Ldraw files from BRICK_DIR to STL file format and stores them in STL_DIR
@File      : create_stl_files.py
@Project   : BrickScanner
@Time      : 21.07.22 20:30
@Author    : flowmeadow
"""
import os
import sys

sys.path.append(os.getcwd())  # required to run script from console

from definitions import *
from lib.helper.lego_bricks import convert_to_stl

if __name__ == "__main__":
    """
    Converts all Ldraw files from BRICK_DIR to STL file format and stores them in STL_DIR.
    LDView needs to be installed. Only tested in Linux
    """
    # use only files with a numeric ID of length 4 or 5
    files = [f for f in os.listdir(BRICK_DIR) if f.endswith(".dat") and f[:-4].isnumeric() and len(f[:-4]) in [4, 5]]
    for idx, file in enumerate(files):
        print(f"Processing file {file} ({idx + 1}|{len(files)})")
        file_base = file[:-4]
        out_file = f"{file_base}.stl"
        if out_file in os.listdir(STL_DIR):
            print("File already converted!")
            continue
        convert_to_stl(file_base, f"{STL_DIR}/{out_file}")
