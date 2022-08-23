#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : eval_obb_search.py
@Project   : BrickScanner
@Time      : 09.08.22 17:17
@Author    : flowmeadow
"""
import copy
import pickle
import time

import numpy as np
import open3d as o3d
import pandas as pd
from glpg_flowmeadow.transformations.methods import rot_mat
from matplotlib import pyplot as plt

from definitions import *
from lib.helper.cloud_operations import draw_point_clouds, cloud2cloud_err, display_recon_rate
from lib.helper.lego_bricks import get_base_bricks, load_stl
from lib.retrieval.bounding_box import find_closest_obb_edges, compute_obb_edges, compute_obb_volume
from lib.retrieval.cloud_alignment import prepare_cloud, align_point_clouds, find_model
import seaborn as sns
import matplotlib.patheffects as path_effects

from scripts.sim_recon_setup import prepare_mesh


def add_median_labels(ax, fmt=".1f"):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4 : len(lines) : lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f"{value:{fmt}}", ha="center", va="center", fontweight="bold", color="white")
        # create median-colored border around white text for contrast
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ]
        )


def eval_computation_time(base_name, folder_name):
    out_file = "eval_time"

    with open(f"{DATA_DIR}/base_obb_data.pkl", "rb") as f:
        obb_data = pickle.load(f)

    exp_paths = os.listdir(f"{DATA_DIR}/{base_name}")

    # TODO: do with pandas
    brick_ids = [folder.split("_id-")[-1] for folder in exp_paths]

    # Create empty dataframe with column names
    sample_nums = [
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
    ]

    num_bricks = 50

    for num_samples in sample_nums:
        eval_data = pd.DataFrame(
            index=brick_ids[:num_bricks], columns=["volume", "z_offset", "err", "completeness", "times"]
        )

        base_files = get_base_bricks()
        base_idcs = np.array([list(obb_data["files"]).index(f) for f in base_files])
        target_edges_lst = np.array(obb_data["edges"])[base_idcs]

        for f_idx, folder in enumerate(exp_paths):
            exp_dir = f"{DATA_DIR}/{base_name}/{folder}"
            if os.path.isdir(exp_dir):
                if f_idx >= num_bricks:
                    continue

                # # rescale was based on the pc center, so we correct this
                scale = 0.1
                pc_tmp = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
                pc_tmp.scale(scale, pc_tmp.get_center())
                z_offset = (
                    np.min(np.array(pc_tmp.points)[:, -1]) / scale
                )  # distance from the lowest point to belt in cm
                if z_offset > 0.1:
                    print("WARNING")

                brick_id = folder.split("_id-")[-1]
                file_name = f"{brick_id}.stl"
                # load target edges
                idx = base_files.index(f"{brick_id}.stl")
                target_edges = target_edges_lst[idx]

                # load reconstructed point cloud
                pc_source = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
                pc_source = prepare_cloud(pc_source, pca_method="all")
                volume = compute_obb_volume(pc_source)

                # load target point cloud
                mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{brick_id}.stl")
                pc_orig = mesh.sample_points_uniformly(100_000)

                errors, times, completeness_arr = [], [], []
                for i in range(20):
                    pc_target = mesh.sample_points_uniformly(num_samples)
                    # pc_target = prepare_cloud(pc_target, pca_method="all")

                    start_time = time.time()
                    # compute ground truth error
                    T_target = align_point_clouds(pc_source, pc_target, debug=False)
                    end_time = time.time() - start_time

                    pc_control = copy.deepcopy(pc_orig)
                    pc_control.transform(T_target)
                    pc_target.transform(T_target)
                    err = cloud2cloud_err(pc_source, pc_control)
                    errors.append(err)
                    times.append(end_time)

                    min_dist = 0.1
                    t2s_dists = np.array(pc_control.compute_point_cloud_distance(pc_source))
                    idcs_recon = np.where(t2s_dists < min_dist)[0]
                    completeness = len(idcs_recon) / len(t2s_dists)
                    completeness_arr.append(completeness)

                print(f"{file_name} with {num_samples} samples:")
                print(f"\ttime: {np.mean(times):.4f}\terr: {np.mean(errors):.6f}\trate: {np.mean(completeness_arr)}")

                # fill evaluation data
                # eval_data.at[brick_id, "success"] = file_name in files
                eval_data.at[brick_id, "volume"] = volume
                eval_data.at[brick_id, "z_offset"] = z_offset
                eval_data.at[brick_id, "err"] = errors
                eval_data.at[brick_id, "completeness"] = completeness_arr
                eval_data.at[brick_id, "times"] = times

            directory = f"{DATA_DIR}/{folder_name}"
            with open(f"{directory}/{out_file}_{num_samples}.pkl", "wb") as f:
                pickle.dump(eval_data, f)


def eval_gt_data(base_name, folder_name):
    num_samples = 5000
    out_file = f"eval_data_gt_{num_samples}"

    with open(f"{DATA_DIR}/base_obb_data.pkl", "rb") as f:
        obb_data = pickle.load(f)

    exp_paths = os.listdir(f"{DATA_DIR}/{base_name}")

    # TODO: do with pandas
    brick_ids = [folder.split("_id-")[-1] for folder in exp_paths]

    # Create empty dataframe with column names

    eval_data = pd.DataFrame(
        index=brick_ids, columns=["volume", "z_offset", "err", "completeness", "times", "transforms"]
    )

    for f_idx, folder in enumerate(exp_paths):
        exp_dir = f"{DATA_DIR}/{base_name}/{folder}"
        if os.path.isdir(exp_dir):

            # # rescale was based on the pc center, so we correct this
            scale = 0.1
            pc_tmp = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
            pc_tmp.scale(scale, pc_tmp.get_center())
            z_offset = np.min(np.array(pc_tmp.points)[:, -1]) / scale  # distance from the lowest point to belt in cm
            if z_offset > 0.1:
                print("WARNING")

            brick_id = folder.split("_id-")[-1]
            file_name = f"{brick_id}.stl"
            # load target edges

            # load reconstructed point cloud
            pc_source = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
            pc_source = prepare_cloud(pc_source, pca_method="all")
            volume = compute_obb_volume(pc_source)

            # load target point cloud
            mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{brick_id}.stl")
            pc_orig = mesh.sample_points_uniformly(100_000)

            errors, times, completeness_arr = [], [], []
            transforms = []
            for i in range(20):
                start_time = time.time()
                pc_target = mesh.sample_points_uniformly(num_samples)
                # pc_target = prepare_cloud(pc_target, pca_method="all")

                # compute ground truth error
                T_target = align_point_clouds(pc_source, pc_target, debug=False)
                end_time = time.time() - start_time

                pc_control = copy.deepcopy(pc_orig)
                pc_control.transform(T_target)

                # draw_point_clouds(pc_control, pc_source)

                err = cloud2cloud_err(pc_source, pc_control)
                errors.append(err)
                times.append(end_time)
                min_dist = 0.1
                t2s_dists = np.array(pc_control.compute_point_cloud_distance(pc_source))
                idcs_recon = np.where(t2s_dists < min_dist)[0]
                completeness = len(idcs_recon) / len(t2s_dists)
                completeness_arr.append(completeness)
                transforms.append(T_target)

            # idx = np.argsort(errors)[0]
            # pc_control = copy.deepcopy(pc_orig)
            # pc_control.transform(transforms[idx])
            # draw_point_clouds(pc_control, pc_source)

            print(f"{file_name} ({f_idx + 1}| {len(exp_paths)}):")
            print(f"\ttime: {np.mean(times):.4f}\terr: {np.mean(errors):.6f}\trate: {np.mean(completeness_arr)}")

            # fill evaluation data
            # eval_data.at[brick_id, "success"] = file_name in files
            eval_data.at[brick_id, "volume"] = volume
            eval_data.at[brick_id, "z_offset"] = z_offset
            eval_data.at[brick_id, "err"] = errors
            eval_data.at[brick_id, "completeness"] = completeness_arr
            eval_data.at[brick_id, "times"] = times
            eval_data.at[brick_id, "transforms"] = transforms

    directory = f"{DATA_DIR}/{folder_name}"
    with open(f"{directory}/{out_file}.pkl", "wb") as f:
        pickle.dump(eval_data, f)


def eval_retrieval(base_name, folder_name):
    num_samples = 5000
    num_tries = 3
    out_file = f"eval_data_retrieval"
    directory = f"{DATA_DIR}/{folder_name}"

    with open(f"{DATA_DIR}/base_obb_data.pkl", "rb") as f:
        obb_data = pickle.load(f)

    base_files = get_base_bricks()
    base_idcs = np.array([list(obb_data["files"]).index(f) for f in base_files])
    target_edges_lst = np.array(obb_data["edges"])[base_idcs]

    exp_paths = os.listdir(f"{DATA_DIR}/{base_name}")

    # TODO: do with pandas
    brick_ids = [folder.split("_id-")[-1] for folder in exp_paths]

    # Create empty dataframe with column names
    successes, total_files = 0, 0
    for f_idx, folder in enumerate(exp_paths):
        exp_dir = f"{DATA_DIR}/{base_name}/{folder}"
        if os.path.isdir(exp_dir):
            brick_id = folder.split("_id-")[-1]
            file_name = f"{brick_id}.stl"
            print(f"{file_name} ({f_idx + 1}| {len(exp_paths)}):")

            if os.path.exists(f"{directory}/{out_file}.pkl"):
                with open(f"{directory}/{out_file}.pkl", "rb") as f:
                    eval_data = pickle.load(f)
            else:
                eval_data = pd.DataFrame(
                    index=brick_ids,
                    columns=["volume", "z_offset", "best_files", "err", "completeness", "times", "transforms", "DONE"],
                )
                eval_data["DONE"] = False

            # # Reset data
            # eval_data["DONE"] = False
            # with open(f"{directory}/{out_file}.pkl", "wb") as f:
            #     pickle.dump(eval_data, f)
            # return

            if eval_data.at[brick_id, "DONE"]:
                print("\t Already computed")
                continue

            # # rescale was based on the pc center, so we correct this
            scale = 0.1
            pc_tmp = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
            pc_tmp.scale(scale, pc_tmp.get_center())
            z_offset = np.min(np.array(pc_tmp.points)[:, -1]) / scale  # distance from the lowest point to belt in cm
            if z_offset > 0.1:
                print("\t Skipped because of high z_offset")
                continue
                # print(f"Wrong offset in {fails}/{f_idx + 1} cases")

            # load reconstructed point cloud
            pc_source = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
            pc_source = prepare_cloud(pc_source, pca_method="all")
            volume = compute_obb_volume(pc_source)

            # load target edges
            idcs = find_closest_obb_edges(compute_obb_edges(pc_source), target_edges_lst, thresh=np.inf, max_best=50)
            best_files = np.array(base_files)[idcs]

            if file_name not in best_files:
                best_files[-1] = file_name
                # print(f"Model {file_name} was not in obb search")

            errors, times, completeness_arr = [], [], []
            transforms = []
            for t_file in best_files:
                # load target point cloud
                mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{t_file}")

                err_min, completeness, T, t = np.inf, None, None, 0
                for i in range(num_tries):
                    start_time = time.time()
                    pc_target = mesh.sample_points_uniformly(num_samples)
                    # pc_target = prepare_cloud(pc_target, pca_method="all")

                    # compute ground truth error
                    T_target = align_point_clouds(pc_source, pc_target, debug=False)
                    t += time.time() - start_time

                    pc_target.transform(T_target)
                    err = cloud2cloud_err(pc_source, pc_target, method=np.mean)
                    if err < err_min:
                        err_min = err
                        T = T_target

                        min_dist = 0.1
                        t2s_dists = np.array(pc_target.compute_point_cloud_distance(pc_source))
                        idcs_recon = np.where(t2s_dists < min_dist)[0]
                        completeness = len(idcs_recon) / len(t2s_dists)
                errors.append(err_min)
                times.append(t)
                completeness_arr.append(completeness)
                transforms.append(T)

            best_idx = np.argsort(errors)[0]
            total_files += 1
            if best_files[best_idx] == file_name:
                successes += 1
            print(
                f"\tBest Match: {best_files[best_idx]}\tError: {errors[best_idx]}\tSuccesses: {successes}/{total_files}"
            )

            # idx = np.argsort(errors)[0]
            # pc_control = copy.deepcopy(pc_orig)
            # pc_control.transform(transforms[idx])
            # draw_point_clouds(pc_control, pc_source)

            # fill evaluation data
            # eval_data.at[brick_id, "success"] = file_name in files
            eval_data.at[brick_id, "volume"] = volume
            eval_data.at[brick_id, "z_offset"] = z_offset
            eval_data.at[brick_id, "best_files"] = best_files
            eval_data.at[brick_id, "err"] = errors
            eval_data.at[brick_id, "completeness"] = completeness_arr
            eval_data.at[brick_id, "times"] = times
            eval_data.at[brick_id, "transforms"] = transforms
            eval_data.at[brick_id, "DONE"] = True

            with open(f"{directory}/{out_file}.pkl", "wb") as f:
                pickle.dump(eval_data, f)


def plot_performance(folder_name, file_name="eval_data"):
    sns.set()
    sns.set(font_scale=1.5)
    params = {'text.usetex': True,
              'font.family': 'serif',
              }
    plt.rcParams.update(params)

    directory = f"{DATA_DIR}/{folder_name}"

    df_list = []
    for file in os.listdir(directory):
        if file.startswith("eval_time"):
            num_samples = int(file.split("_")[-1].split(".")[0])
            with open(f"{directory}/{file}", "rb") as f:
                eval_data = pickle.load(f)
            eval_data["num_samples"] = num_samples
            eval_data["err_min"] = np.min(np.array(eval_data["err"].to_list()), axis=1)
            eval_data["err_std"] = np.std(np.array(eval_data["err"].to_list()), axis=1)
            eval_data["err_max"] = np.max(np.array(eval_data["err"].to_list()), axis=1)
            eval_data["err_gap"] = eval_data["err_max"] / eval_data["err_min"] - 1.0

            df_list.append(eval_data)
    df = pd.concat(df_list)
    df = df.explode(["err", "times", "completeness"])
    df["times"] *= 1000

    file_num = len(df)
    df = df.where(df["z_offset"] < 0.1)
    df = df.dropna()
    print(f"Removed {len(df)} of {file_num} files ({100 * len(df) / file_num:.2f}% remaining)")

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(15, 4)
    df["num_samples"] = df["num_samples"].astype(int)
    sns.boxplot(ax=axes[0], data=df, x="num_samples", y="times", showfliers=False)
    sns.boxplot(ax=axes[1], data=df, x="num_samples", y="err", showfliers=False)

    # axes[0].set(yscale="log")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    axes[0].set_xlabel(r"\textbf{Target PC samples [-]}")
    axes[1].set_xlabel(r"\textbf{Target PC samples [-]}")

    axes[0].set_ylabel(r"\textbf{Computation time [ms]}")
    axes[1].set_ylabel(r"\textbf{Ground truth error [-]}")
    axes[0].set_title(r"\textbf{(a)}")
    axes[1].set_title(r"\textbf{(b)}")

    # add_median_labels(axes[0])

    plt.tight_layout()
    plt.show()


def plot_gt_data(folder_name, file_name="eval_data_gt.pkl"):
    sns.set()
    sns.set(font_scale=1.5)
    params = {'text.usetex': True,
              'font.family': 'serif',
              }
    plt.rcParams.update(params)

    directory = f"{DATA_DIR}/{folder_name}"

    df_list = []
    for file in os.listdir(directory):
        if file.startswith("eval_data_gt"):
            num_samples = int(file.split("_")[-1].split(".")[0])
            with open(f"{directory}/{file}", "rb") as f:
                eval_data = pickle.load(f)
            eval_data["num_samples"] = num_samples
            df_list.append(eval_data)
    df = pd.concat(df_list)
    df = df.dropna()
    df = df.explode(["err", "times", "completeness"]).reset_index().rename(columns={"index": "files"})
    df["try"] = df.groupby(["files", "num_samples"]).cumcount() + 1

    errors = np.array(df["err"])
    times = np.array(df["times"])
    min_errors = np.full((errors.shape[0],), np.inf)
    total_times = np.full((times.shape[0],), np.inf)
    num_tries = 20
    for idx in range(min_errors.shape[0]):
        start_idx = num_tries * (idx // num_tries)
        min_errors[idx] = np.min(errors[start_idx: idx + 1])
        total_times[idx] = np.sum(times[start_idx: idx + 1])

    df["err_min"] = min_errors
    df["times"] = total_times
    df["times"] *= 1000

    file_num = len(df)
    df = df.where(df["z_offset"] < 0.1)
    df = df.dropna()
    print(f"Removed {len(df)} of {file_num} files ({100 * len(df) / file_num:.2f}% remaining)")

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)

    # create violin plot
    # sns.boxplot(data=eval_data, ax=ax, showfliers=False)
    # sns.scatterplot(ax=axes[0], data=df, x="volume", y="gt_err")

    df = df.where((df["try"] <= 5) | (df["try"] % 10 == 0))
    df = df.dropna()
    df["try"] = df["try"].astype(int)
    df["num_samples"] = df["num_samples"].astype(int)

    sns.boxplot(ax=axes[0], data=df, x="try", y="err_min", hue="num_samples", showfliers=False)
    sns.boxplot(ax=axes[1], data=df, x="try", y="times", hue="num_samples", showfliers=False)

    # df_time = df.where((df["try"] == 3) & (df["num_samples"] == 5000))
    # print(np.mean(df[]))

    axes[1].set(yscale="log")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    axes[0].legend(title=r"\textbf{Target PC samples}")
    axes[1].get_legend().remove()

    axes[0].set_xlabel(r"\textbf{Number of tries [-]}")
    axes[1].set_xlabel(r"\textbf{Number of tries [-]}")
    axes[0].set_ylabel(r"\textbf{Minimum error of all tries [-]}")
    axes[1].set_ylabel(r"\textbf{Total computation time [ms]}")
    axes[0].set_title(r"\textbf{(a)}")
    axes[1].set_title(r"\textbf{(b)}")
    # add_median_labels(axes[1])

    # add_median_labels(axes[0], fmt='.1e')
    # add_median_labels(axes[1])
    # sns.scatterplot(ax=axes[1], data=df, x="volume", y="err")

    # df = df[df.index == "4223"]

    # sns.boxplot(ax=axes[1], data=df, x="num_samples", y="err_min", showfliers=False)

    # sns.scatterplot(ax=axes[1], data=df, x="volume", y="err")

    #
    # # compute percentile
    # val = 0.95
    # errors = np.sort(df["gt_err"])
    # idx = int(val * len(df))
    # percentile = errors[idx]
    # print(percentile)
    # axes[0].axhline(percentile, ls="--", c="tab:red")
    # axes[1].axhline(percentile, ls="--", c="tab:red")
    # axes[0].set(xscale="log")
    # axes[0].set_ylabel("Ground truth error [-]", fontsize=fontsize)
    # axes[1].set_ylabel("Ground truth error [-]", fontsize=fontsize)
    # axes[0].set_xlabel("Volume [$\mathrm{cm}^{3}$]", fontsize=fontsize)
    # post process and show
    plt.tight_layout()
    plt.show()


def plot_compared_data(folder_name, file_name="eval_data_retrieval.pkl"):
    sns.set()
    sns.set(font_scale=1.5)
    params = {'text.usetex': True,
              'font.family': 'serif',
              }
    plt.rcParams.update(params)

    directory = f"{DATA_DIR}/{folder_name}"
    with open(f"{directory}/{file_name}", "rb") as f:
        df = pickle.load(f)

    # sort based on error values
    df = df.dropna()

    df["similarity"] = None
    df["similarity"] = df["similarity"].astype(object)
    for brick_id in df.index:
        err = df.at[brick_id, "err"]
        idcs = np.argsort(err)
        for val in ["best_files", "err", "completeness", "times", "transforms"]:
            array = np.array(df.at[brick_id, val])
            df.at[brick_id, val] = array[idcs]

        best_files = df.at[brick_id, "best_files"]
        df.at[brick_id, "best_pos"] = list(best_files).index(f"{brick_id}.stl") + 1

        new_err = np.array(df.at[brick_id, "err"])
        similarity = 1 - new_err[0] / new_err
        df.at[brick_id, "similarity"] = similarity

    df = df.explode(["best_files", "err", "times", "completeness", "similarity"]).reset_index().rename(columns={"index": "files"})
    df["guess"] = df.groupby(["files"]).cumcount() + 1

    df["times"] *= 1000
    df_orig = df.copy()

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)

    # create violin plot
    # sns.boxplot(data=eval_data, ax=ax, showfliers=False)
    # sns.scatterplot(ax=axes[0], data=df, x="best_pos", y="similarity")

    # axes.set_xlim(-1, 20)
    df = df.where(df["guess"] == 1)
    df = df.dropna()
    df["best_pos"] = df["best_pos"].astype(int)
    sns.histplot(ax=axes, data=df, x="best_pos", discrete=True)
    val = 0.95
    errors = np.sort(df["best_pos"])
    idx = int(val * len(df))
    percentile = errors[idx]
    print(percentile)
    axes.axvline(percentile, ls='--', c="tab:red")
    axes.set(yscale="log")

    axes.set_xlabel(r"\textbf{Position of Ground Truth [-]}")
    axes.set_ylabel(r"\textbf{Count}")
    # # SECTION: Percentages
    # total = len(df)
    # for i in range(50):
    #     gt_in = len(df.where(df["best_pos"] <= i + 1).dropna())
    #     print(f"i = {i + 1}\t total = {total}\t gt_in = {gt_in}\t perc = {100 * gt_in / total}")
    #
    # # SECTION: TRUE FALSE
    pd.set_option('display.max_columns', 500)
    for mismatch_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for max_set_size in [1, 2, 3, 4, 5]:
            df = df_orig.copy()

            selection = df.where(df["similarity"] < mismatch_thresh).dropna().sort_values("files")
            selection["gt_in"] = False
            selection["failed"] = False
            for file in selection["files"]:

                tmp = selection.where(df["files"] == file).dropna()
                gt_condition = f"{file}.stl" in list(tmp["best_files"])[:max_set_size]
                fail_condition = len(list(tmp["best_files"])) > max_set_size
                selection['gt_in'] = np.where(selection['files'] == file, gt_condition, selection['gt_in'])
                selection['failed'] = np.where(selection['files'] == file, fail_condition, selection['failed'])
                # print(df)

            df = selection.where(df["guess"] == 1).dropna()
            total = len(df)

            tmp = df.copy()
            # df = tmp.copy()
            df = selection.where(df["failed"] == True).dropna()

            total_f = len(df)

            num_gt_in_f = len(df.where(df["gt_in"] == True).dropna())
            num_gt_out_f = total_f - num_gt_in_f
            perc_in_f = f"{100 * num_gt_in_f / total_f:.2f}"
            perc_out_f = f"{100 * num_gt_out_f / total_f:.2f}"
            perct_in_f = f"{100 * num_gt_in_f / total:.2f}"
            perct_out_f = f"{100 * num_gt_out_f / total:.2f}"

            df = tmp.copy()
            df = selection.where(df["failed"] == False).dropna()
            total_s = len(df)
            num_gt_in_s = len(df.where(df["gt_in"] == True).dropna())
            num_gt_out_s = total_s - num_gt_in_s
            perc_in_s = f"{100 * num_gt_in_s / total_s:.2f}"
            perc_out_s = f"{100 * num_gt_out_s / total_s:.2f}"
            perct_in_s = f"{100 * num_gt_in_s / total:.2f}"
            perct_out_s = f"{100 * num_gt_out_s / total:.2f}"

            # print(f"\tgt_in:{100 * num_gt_in / total:.2f}\tgt_out:{100 * (total - num_gt_in) / total:.2f}")
            # print(total, num_gt_in, total - num_gt_in)

            table_txt = rf"""
                        {mismatch_thresh} & {max_set_size} & Count & {num_gt_in_s} & {num_gt_out_s} & {num_gt_in_f} & {num_gt_out_f} \\
                        \ & \ & Total Count & {total_s} & {total_s} & {total_f} & {total_f} \\
                        \ & \ & Percentage & {perc_in_s}\% & {perc_out_s}\% & {perc_in_f}\% & {perc_out_f}\%  \\
                        \ & \ & Total Percentage & {perct_in_s}\% & {perct_out_s}\% & {perct_in_f}\% & {perct_out_f}\%  \\\
                        \hline
                        """
            print(table_txt)
    return



    # df = df_orig.copy()
    # # print(df)
    # # print(np.array(df["files"])[0])
    # df = df.where(df["best_pos"] == 6)
    # df = df.dropna()
    #
    # df = df.where(df["files"] == np.array(df["files"])[0])
    # df = df.dropna()
    #
    # df["best_pos"] = df["best_pos"].astype(int)
    # sns.scatterplot(ax=axes[0], data=df, x="guess", y="similarity")
    # sns.scatterplot(ax=axes[1], data=df, x="guess", y="err")
    #
    # df = df.where(df["guess"] == df["best_pos"])
    # df = df.dropna()
    # sns.scatterplot(ax=axes[0], data=df, x="guess", y="similarity")
    # sns.scatterplot(ax=axes[1], data=df, x="guess", y="err")

    # sns.boxplot(ax=axes[1], data=df, x="try", y="times", hue="num_samples", showfliers=False)



    # df_time = df.where((df["try"] == 3) & (df["num_samples"] == 5000))
    # print(np.mean(df[]))

    # axes[0].set(xscale="log")
    # axes[1].set_xlim([-1, 4])



    # axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    #
    # # axes[0].legend(title="Target PC samples")
    # # axes[1].get_legend().remove()
    #
    # axes[0].set_xlabel("", fontsize=fontsize)
    # axes[1].set_xlabel("Number of tries [-]", fontsize=fontsize)
    # axes[0].set_ylabel("Minimum error of all tries [-]", fontsize=fontsize)
    # axes[1].set_ylabel("Total computation time [ms]", fontsize=fontsize)

    # add_median_labels(axes[0], fmt='.1e')
    # add_median_labels(axes[1])
    # sns.scatterplot(ax=axes[1], data=df, x="volume", y="err")

    # df = df[df.index == "4223"]

    # sns.boxplot(ax=axes[1], data=df, x="num_samples", y="err_min", showfliers=False)

    # sns.scatterplot(ax=axes[1], data=df, x="volume", y="err")

    #
    # # compute percentile
    # val = 0.95
    # errors = np.sort(df["gt_err"])
    # idx = int(val * len(df))
    # percentile = errors[idx]
    # print(percentile)
    # axes[0].axhline(percentile, ls="--", c="tab:red")
    # axes[1].axhline(percentile, ls="--", c="tab:red")
    # axes[0].set(xscale="log")
    # axes[0].set_ylabel("Ground truth error [-]", fontsize=fontsize)
    # axes[1].set_ylabel("Ground truth error [-]", fontsize=fontsize)
    # axes[0].set_xlabel("Volume [$\mathrm{cm}^{3}$]", fontsize=fontsize)
    # post process and show
    plt.tight_layout()
    plt.show()


def visualize_alignment(base_name, folder_name):
    num_samples = 5000
    num_tries = 3
    out_file = f"eval_data_retrieval"
    directory = f"{DATA_DIR}/{folder_name}"

    exp_paths = os.listdir(f"{DATA_DIR}/{base_name}")

    # TODO: do with pandas
    brick_ids = [folder.split("_id-")[-1] for folder in exp_paths]

    # Create empty dataframe with column names
    successes, total_files = 0, 0

    for f_idx, folder in enumerate(exp_paths):

        exp_dir = f"{DATA_DIR}/{base_name}/{folder}"
        if os.path.isdir(exp_dir):
            # if f_idx != 215:
            #     continue
            brick_id = folder.split("_id-")[-1]
            if brick_id != "4748":
                continue
            file_name = f"{brick_id}.stl"
            print(f"{file_name} ({f_idx + 1}| {len(exp_paths)}):")

            # # rescale was based on the pc center, so we correct this
            scale = 0.1
            pc_tmp = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
            pc_tmp.scale(scale, pc_tmp.get_center())
            z_offset = np.min(np.array(pc_tmp.points)[:, -1]) / scale  # distance from the lowest point to belt in cm
            if z_offset > 0.1:
                print("\t Skipped because of high z_offset")
                continue
                # print(f"Wrong offset in {fails}/{f_idx + 1} cases")

            # load reconstructed point cloud
            pc_source = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
            pc_source = prepare_cloud(pc_source, pca_method="all")

            mesh = load_stl(brick_id)
            mesh = prepare_mesh(mesh, random=bool(seed), seed=seed)
            pc_target = mesh.sample_points_uniformly(50_000)
            t_center = pc_target.get_center()
            pc_target.translate(-t_center)
            mesh.translate(-t_center)

            # o3d.visualization.draw_geometries([mesh])
            T = align_point_clouds(pc_source, pc_target, debug=False)
            mesh.transform(T)
            pc_target = mesh.sample_points_uniformly(1000_000)
            # debug_rotation = rot_mat((1., 1., 0.), 90.)
            debug_rotation = rot_mat((1.1, 2., 0.), 180-130.)
            # debug_rotation = rot_mat((0.0, 1.5, -1.5), -120.0)
            pc_target.rotate(debug_rotation, np.zeros(3))
            pc_source.rotate(debug_rotation, np.zeros(3))

            display_recon_rate(pc_source, pc_target, lightness=0.9, coord_axes=False)

            # mesh.compute_triangle_normals()
            #
            # bb = mesh.get_axis_aligned_bounding_box()
            # bb.color = np.zeros(3)
            #
            # pc_source.translate(np.array([0.15, 2., 0.0]))
            #
            # bb_2 = pc_source.get_axis_aligned_bounding_box()
            # bb_2.color = np.zeros(3)
            #
            # import matplotlib.colors as mcolors
            # from matplotlib.colors import to_rgb
            #
            # colors = np.array([to_rgb(c) for c in list(mcolors.TABLEAU_COLORS.values())])[0]
            #
            # pc_source.paint_uniform_color(colors)
            # o3d.visualization.draw_geometries([mesh, bb, pc_source, bb_2])



if __name__ == "__main__":
    folder_name = "eval_alignment"
    file_name = "eval_data_gt"
    seed = 2
    base_name = f"eval_retrieval/seed_{str(seed).zfill(4)}"
    # eval_computation_time(base_name, folder_name)
    # eval_gt_data(base_name, folder_name)
    # eval_retrieval(base_name, folder_name)

    # plot_performance(folder_name)
    # plot_gt_data(folder_name)
    # plot_compared_data(folder_name)
    visualize_alignment(base_name, folder_name)