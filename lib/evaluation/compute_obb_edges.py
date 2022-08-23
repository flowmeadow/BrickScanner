#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : compute_obb_edges.py
@Project   : BrickScanner
@Time      : 11.08.22 15:55
@Author    : flowmeadow
"""
import pickle

import numpy as np
from glpg_flowmeadow.transformations.methods import rot_mat

from definitions import *
from lib.helper.lego_bricks import get_base_bricks
import open3d as o3d

from lib.retrieval.bounding_box import compute_obb_edges, compute_obb_volume
from lib.retrieval.cloud_alignment import prepare_cloud


def main():
    files = get_base_bricks()

    path = f"{DATA_DIR}/base_obb_data_2.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            obb_data = pickle.load(f)
    else:
        obb_data = dict(files=[], edges=[], volume=[])

    for idx, file in enumerate(files):
        if file in obb_data["files"]:
            print("skipped")
            continue
        print(file, idx, len(files))
        mesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{file}")
        edges, volumes = [], []
        for i in range(10):
            pc = mesh.sample_points_uniformly(100_000)
            pc = prepare_cloud(pc, pca_method="all")
            edges.append(compute_obb_edges(pc))
            volumes.append(compute_obb_volume(pc))
        edges = np.mean(np.array(edges), axis=0)
        volume = np.mean(np.array(volumes))

        obb_data["files"].append(file)
        obb_data["edges"].append(edges)
        obb_data["volume"].append(volume)

        with open(f"{DATA_DIR}/base_obb_data_2.pkl", "wb") as f:
            pickle.dump(obb_data, f)


if __name__ == "__main__":
    main()
