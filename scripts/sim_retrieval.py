#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Script to find the best match for a virtual brick point cloud generated with an OpenGL simulator
@File      : sim_retrieval.py
@Project   : BrickScanner
@Time      : 22.07.22 19:18
@Author    : flowmeadow
"""
from lib.retrieval.cloud_alignment import find_model, prepare_cloud, show_results
import open3d as o3d
from definitions import *


if __name__ == "__main__":
    folder_name = "sim_recon"
    debug = False

    # get brick id
    prefix = "recon_"
    brick_id = None
    for file in os.listdir(f"{DATA_DIR}/{folder_name}"):
        if file.startswith(prefix):
            brick_id = file[len(prefix) : -4]
    if brick_id is None:
        raise FileNotFoundError()

    # load source point cloud
    pc_source = o3d.io.read_point_cloud(f"{DATA_DIR}/{folder_name}/recon_{brick_id}.pcd")

    # prepare it for alignment
    pc_source = prepare_cloud(pc_source)

    # find the best alignment matches
    files, errors, transformations, percentages = ret = find_model(
        pc_source, debug_file=f"{brick_id}.stl" if debug else None
    )

    # show results
    show_results(*ret, pc_source=pc_source)
