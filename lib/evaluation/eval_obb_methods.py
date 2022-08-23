#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : evaluates minimum bounding box search
@File      : eval_obb_methods.py
@Project   : BrickScanner
@Time      : 22.07.22 18:17
@Author    : flowmeadow
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
from definitions import *
from lib.helper.timing import time_fun
from lib.retrieval.bounding_box import AxisAlignedCloud, compute_obb_edges, compute_obb_volume


def evaluate_optimal_obb(path: str, files: List[str], out_path=None, tries=10):
    """
    Evaluates minimum bounding box search for different approaches and saves them in a CSV file
    :param path: directory of STL files
    :param files: file names of STL files
    :param out_path: directory to store the CSV file (Optional)
    :param tries: number of tries for each method
    """
    if out_path is None:
        out_path = f"{path}/evaluation.csv"
    length = len(files)

    # generate pandas dataframe
    methods = ["iterative", "pca_based", "combined"]
    if not os.path.exists(out_path):
        df = pd.DataFrame()
        df["model"] = files
        df["edges"] = ""
        df["volume"] = np.zeros(length)
        for method in methods:
            for func in ["mean", "std"]:
                df[f"{method}_edges_{func}"] = ""
                df[f"{method}_volume_{func}"] = np.zeros(length)
                df[f"{method}_error_{func}"] = np.zeros(length)
                df[f"{method}_time_{func}"] = np.zeros(length)
        df.to_csv(out_path)

    # repeat for each file
    for file_idx, file in enumerate(files):
        print(f"Processing file {file} ({file_idx + 1}|{len(files)}) ")
        df = pd.read_csv(out_path, index_col=0)
        location = df["model"] == file
        if float(df.loc[location, "volume"].values[0]) != 0:
            print(f"Skipped file {file}. Already computed")
            continue
        try:
            mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(f"{path}/{file}")
            pc: o3d.geometry.PointCloud = mesh.sample_points_uniformly(100_000)
        except ImportError as e:
            print(f"Failed to load and generate point cloud for file {file}:\n{e}")
            continue

        # place pc in world center
        pc.translate(-pc.get_center())
        edges_true = compute_obb_edges(pc)
        volume_true = compute_obb_volume(pc)
        df.loc[location, "edges"] = str(edges_true)
        df.loc[location, "volume"] = volume_true
        # generate uniform randomly rotated point cloud

        edges = dict()
        times = dict()

        for method in methods:
            edges[method] = np.zeros((tries, 3))
            times[method] = np.zeros(tries)

        # try several times with random start rotation
        for idx in range(tries):
            # rotate point cloud randomly
            angle = np.random.rand() * np.pi * 2
            axis = np.random.random(3)
            axis /= np.linalg.norm(axis)
            pc.rotate(pc.get_rotation_matrix_from_axis_angle(angle * axis))

            # compute optimal bounding box ...
            optimizer = AxisAlignedCloud(pc)

            # ... for every method
            for method in methods:
                t, pc = time_fun(optimizer.optimize, kwargs=dict(method=method))
                edges[method][idx, :] = compute_obb_edges(pc)
                times[method][idx] = t

        # compute mismatch between true and optimized values
        for method in methods:
            e_all = edges[method]
            t_all = times[method]
            for key, func in zip(["mean", "std"], [np.mean, np.std]):
                e = func(e_all, axis=0)
                t = func(t_all, axis=0)
                error = func(np.sum((e_all - edges_true) ** 2, axis=0))
                df.loc[location, f"{method}_edges_{key}"] = str(e)
                df.loc[location, f"{method}_volume_{key}"] = np.prod(e)
                df.loc[location, f"{method}_error_{key}"] = error
                df.loc[location, f"{method}_time_{key}"] = t
        df.to_csv(out_path)


def evaluate_results(path: str, min_error=0.01):
    """
    load CSV file and plot results
    :param path: file path of CSV file
    :param min_error: consider only errors below this for plotting and handle all above as fail
    """
    values = ["error_mean", "time_mean"]

    for value in values:
        df = pd.read_csv(path, index_col=0)
        # df = df.dropna()
        method = "combined"
        err = df[f"{method}_volume_std"] / df[f"{method}_volume_mean"]
        print(np.max(err))

        methods = ["iterative", "pca_based", "combined"]
        data = {}
        for method in methods:
            # OP's data
            # only consider parts, where the optimal OBB has smaller or equal volume
            df = df.loc[df["volume"] <= df[f"{method}_volume_mean"]]
            means = df[f"{method}_{value}"]
            means_reduced = np.array(means.loc[means < min_error])
            success_rate = len(means_reduced) / len(means)
            print(f"Sucess: {success_rate * 100:.2f}%")
            data[method] = np.array(means_reduced if not value == "time_mean" else means)

        # Create DataFrame where NaNs fill shorter arrays
        plot_df = pd.DataFrame([data[m] for m in methods]).transpose()

        # Label the columns of the DataFrame
        plot_df = plot_df.set_axis(methods, axis=1)

        # Violin plot
        plt.figure(value)
        sns.boxplot(data=plot_df, showfliers=False)
        plt.show()


if __name__ == "__main__":
    # compute bounding boxes for all files and evaluate them
    # files = []
    # for file in os.listdir(STL_DIR):
    #     if file.endswith(".stl"):
    #         files.append(file)
    # evaluate_optimal_obb(STL_DIR, files)

    # plot results
    path = f"{STL_DIR}/evaluation.csv"
    evaluate_results(path)
