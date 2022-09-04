#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Converts all Ldraw files from BRICK_DIR to STL file format and stores them in STL_DIR
@File      : create_stl_files.py
@Project   : BrickScanner
@Time      : 21.07.22 20:30
@Author    : flowmeadow
"""
from definitions import *
from lib.helper.lego_bricks import convert_to_stl

if __name__ == "__main__":
    """
    Converts all Ldraw files from BRICK_DIR to STL file format and stores them in STL_DIR. 
    LDView needs to be installed. Only tested in Linux
    """
    for idx, file in enumerate(os.listdir(BRICK_DIR)):
        if file.endswith(".dat"):
            print(f"Processing file {file} ({idx + 1}|{len(os.listdir(BRICK_DIR))})")
            file_base = file[:-4]
            out_file = f"{file_base}.stl"
            if out_file in os.listdir(STL_DIR):
                print("File already converted!")
                continue
            convert_to_stl(file_base, f"{STL_DIR}/{out_file}")
