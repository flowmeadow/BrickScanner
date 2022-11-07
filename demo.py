#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Demo script for the simulation-based brick identification pipeline
@File      : test.py
@Project   : BrickScannerTest
@Time      : 04.09.22 20:47
@Author    : flowmeadow
"""

import open3d as o3d

from definitions import *
from lib.retrieval.cloud_alignment import (find_model, prepare_cloud,
                                           rate_alignment, show_results)
from scripts.sim_recon_setup import double_side_recon

if __name__ == "__main__":
    # define directory for epipolar geometry data and reconstructed point cloud
    data_dir = f"{DATA_DIR}/demo"
    # define directory for the stereo image pairs
    img_dir = f"{IMG_DIR}/demo"
    # select a brick by its LDraw ID
    brick_id = "3039"  # 3001

    # settings for image generation and 3D reconstruction
    settings = dict(
        automated=True,  # if True, image generation starts and stops automatically
        generate_new=True,  # if True, generate new images
        show_result=True,  # show reconstructed point cloud
        step=0.02,  # reconstruction resolution
    )
    double_side_recon(data_dir, img_dir, brick_id, **settings)

    # load pointcloud and prepare it for alignment
    pc = o3d.io.read_point_cloud(f"{data_dir}/recon_{brick_id}.pcd")
    pc = prepare_cloud(pc)

    # find best models including the errors and transformations to match the reconstructed point cloud
    files, errors, transformations = find_model(
        pc,
        debug_file=f"{brick_id}.stl",  # select a model file for alignment visualization
        max_best=50,  # maximum amount of files used from preselection
    )

    # rate the alignment and show results
    ret, idcs = rate_alignment(errors)
    if ret:
        print(f"Found {len(idcs)} possible models:")
        show_results(files[idcs], errors[idcs], transformations[idcs], pc, mesh_only=True)
    else:
        print("Brick identification failed!")
