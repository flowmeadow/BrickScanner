#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Ldraw brick database operations
@File      : lego_bricks.py
@Project   : BrickScanner
@Time      : 14.06.22 15:35
@Author    : flowmeadow
"""

import numpy as np
import open3d as o3d
from definitions import *


def convert_to_stl(id: str, out_path: str):
    """
    Converts a Ldraw file to STL
    :param id: brick id of the file to convert
    :param out_path: path to store the STL file
    """
    # check if id exists
    file_name = f"{id}.dat"
    if file_name not in os.listdir(BRICK_DIR):
        print(f"{id} is not a valid id. There is no file called {file_name}")
        return None
    os.system(f"LDView {BRICK_DIR}/{file_name} -ExportFile={out_path}")


def load_stl(id: str) -> o3d.geometry.TriangleMesh:
    """
    loads the given brick id as triangle mesh
    :param id: brick id
    :return: triangle mesh
    """
    stl_path = f"{STL_DIR}/{id}.stl"
    mesh = o3d.io.read_triangle_mesh(stl_path)
    return mesh


def load_random_stl(seed: int = None) -> o3d.geometry.TriangleMesh:
    """
    Loads a random brick model
    :param seed: seed for random generator (Optional)
    :return: Triangle mesh
    """
    files = os.listdir(STL_DIR)

    if seed:
        np.random.seed(seed)
    idx = np.random.randint(len(files))
    stl_path = f"{STL_DIR}/{files[idx]}"
    mesh = o3d.io.read_triangle_mesh(stl_path)
    return mesh


def get_base_bricks(path=STL_DIR):
    files = [
        file for file in os.listdir(path) if file.endswith(".stl") and file[:-4].isnumeric() and len(file[:-4]) == 4
    ]
    return files
