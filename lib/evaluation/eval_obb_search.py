#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : eval_obb_search.py
@Project   : BrickScanner
@Time      : 09.08.22 17:17
@Author    : flowmeadow
"""
import pickle

import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import pyplot as plt

from definitions import *
from lib.helper.lego_bricks import get_base_bricks
from lib.retrieval.bounding_box import find_closest_obb_edges, compute_obb_edges, compute_obb_volume
from lib.retrieval.cloud_alignment import prepare_cloud
import seaborn as sns


def main(base_name, folder_name, out_file="eval_data"):
    with open(f"{DATA_DIR}/base_obb_data.pkl", "rb") as f:
        obb_data = pickle.load(f)

    exp_paths = os.listdir(f"{DATA_DIR}/{base_name}")
    sucess = 0

    # TODO: do with pandas
    brick_ids = [folder.split("_id-")[-1] for folder in exp_paths]

    # Create empty dataframe with column names
    eval_data = pd.DataFrame(index=brick_ids, columns=["num_files", "volume", "z_offset", "gt_err", "best_errors"])

    base_files = get_base_bricks()
    base_idcs = np.array([list(obb_data["files"]).index(f) for f in base_files])
    target_edges_lst = np.array(obb_data["edges"])[base_idcs]

    for f_idx, folder in enumerate(exp_paths):
        exp_dir = f"{DATA_DIR}/{base_name}/{folder}"
        if os.path.isdir(exp_dir):
            brick_id = folder.split("_id-")[-1]
            file_name = f"{brick_id}.stl"
            # load target edges
            idx = base_files.index(f"{brick_id}.stl")
            target_edges = target_edges_lst[idx]

            # load reconstructed point cloud
            pc_source = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
            pc_source = prepare_cloud(pc_source, pca_method="all")
            source_edges = compute_obb_edges(pc_source)

            # load target point cloud
            # mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{brick_id}.stl")
            # pc_target = mesh.sample_points_uniformly(100_000)
            # pc_target = prepare_cloud(pc_target, pca_method="all")
            # target_edges = compute_obb_edges(pc_target)

            # compute ground truth error
            err = (target_edges / source_edges - 1.0) ** 2
            err = np.sum(err)

            idcs, errors = find_closest_obb_edges(
                compute_obb_edges(pc_source), target_edges_lst, thresh=0.1, return_err=True
            )
            errors = errors[idcs]

            best_files = list(np.array(base_files)[idcs])
            if file_name in best_files:
                pos = best_files.index(file_name)
            else:
                pos = -1

            # rescale was based on the pc center, so we correct this
            scale = 0.1
            pc_tmp = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
            pc_tmp.scale(scale, pc_tmp.get_center())
            z_offset = np.min(np.array(pc_tmp.points)[:, -1]) / scale  # distance from the lowest point to belt in cm

            # fill evaluation data
            # eval_data.at[brick_id, "success"] = file_name in files
            eval_data.at[brick_id, "num_files"] = len(idcs)
            eval_data.at[brick_id, "volume"] = compute_obb_volume(pc_source)
            eval_data.at[brick_id, "z_offset"] = z_offset
            eval_data.at[brick_id, "gt_err"] = err
            eval_data.at[brick_id, "best_pos"] = pos

            eval_data["best_errors"] = eval_data["best_errors"].astype(object)
            eval_data.at[brick_id, "best_errors"] = errors

    directory = f"{DATA_DIR}/{folder_name}"
    with open(f"{directory}/{out_file}.pkl", "wb") as f:
        pickle.dump(eval_data, f)


def plot_error(folder_name, file_name="eval_data"):
    sns.set()
    sns.set(font_scale=1.5)
    params = {'text.usetex': True,
              'font.family': 'serif',
              }
    plt.rcParams.update(params)

    directory = f"{DATA_DIR}/{folder_name}"

    with open(f"{directory}/{file_name}.pkl", "rb") as f:
        df = pickle.load(f)
    file_num = len(df)
    df = df.where(df["z_offset"] < 0.1)
    df = df.dropna()
    print(f"Removed {len(df)} of {file_num} files ({100 * len(df) / file_num:.2f}% remaining)")

    # from matplotlib import rc
    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # ## for Palatino and other serif fonts use:
    # # rc('font',**{'family':'serif','serif':['Palatino']})
    # rc('text', usetex=True)

    fontsize = 16
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    fig.set_size_inches(15, 4)

    # create violin plot
    # sns.boxplot(data=eval_data, ax=ax, showfliers=False)
    sns.scatterplot(ax=axes[0], data=df, x="volume", y="gt_err")
    sns.boxplot(ax=axes[1], data=df, y="gt_err", showfliers=False)

    # compute percentile
    val = 0.95
    errors = np.sort(df["gt_err"])
    idx = int(val * len(df))
    percentile = errors[idx]
    print(percentile)

    # axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[0].axhline(percentile, ls='--', c="tab:red")
    axes[1].axhline(percentile, ls='--', c="tab:red")
    axes[0].set(xscale="log")
    axes[0].set_ylabel(r"\textbf{Ground truth error [-]}")
    axes[1].set_ylabel("")
    axes[0].set_xlabel(r"\textbf{Volume [$\mathbf{cm^{3}}$]}")
    # post process and show
    plt.tight_layout()
    plt.show()


def plot_results(folder_name, file_name="eval_data"):
    sns.set()
    sns.set(font_scale=1.5)
    params = {'text.usetex': True,
              'font.family': 'serif',
              }
    plt.rcParams.update(params)

    directory = f"{DATA_DIR}/{folder_name}"

    with open(f"{directory}/{file_name}.pkl", "rb") as f:
        df = pickle.load(f)
    file_num = len(df)
    df = df.where(df["z_offset"] < 0.1)
    df = df.where(df["best_pos"] != -1)
    df = df.dropna()

    print(f"Removed {len(df)} of {file_num} files ({100 * len(df) / file_num:.2f}% remaining)")

    fontsize = 16
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(15, 4)

    # create violin plot
    # sns.boxplot(data=eval_data, ax=ax, showfliers=False)
    sns.histplot(ax=axes[0], data=df, x="num_files", bins=20)
    sns.histplot(ax=axes[1], data=df, x="best_pos", bins=50)
    # axes[1].set(yscale="log")

    val = 0.8
    errors = np.sort(df["best_pos"])
    idx = int(val * len(df))
    percentile = errors[idx]
    print(percentile)
    axes[1].axvline(percentile, ls='--', c="tab:red")

    axes[1].set_xlim([0, 100])
    axes[0].set_xlabel(r"\textbf{Number of selected models [-]}")
    axes[0].set_ylabel(r"\textbf{Count}")
    axes[1].set_xlabel(r"\textbf{Position of ground truth in selection [-]}")
    axes[1].set_ylabel("")
    axes[0].set_title(r"\textbf{(a)}")
    axes[1].set_title(r"\textbf{(b)}")

    # sns.boxplot(ax=axes[1], data=df, y="gt_err", showfliers=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_name = "eval_obb_search"
    file_name = "eval_data"
    seed = 2
    base_name = f"eval_retrieval/seed_{str(seed).zfill(4)}"
    # main(base_name, folder_name, out_file=file_name)
    plot_error(folder_name, file_name=file_name)
    plot_results(folder_name, file_name=file_name)
