#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : eval_brick_recon.py
@Project   : BrickScanner
@Time      : 10.08.22 16:03
@Author    : flowmeadow
"""
import pickle

import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
from glpg_flowmeadow.transformations.methods import rot_mat

from definitions import *
from lib.helper.lego_bricks import load_stl
from lib.retrieval.bounding_box import compute_obb_edges, compute_obb_volume, find_closest_obb_edges
from lib.retrieval.cloud_alignment import prepare_cloud
from matplotlib import pyplot as plt

from scripts.sim_recon_setup import prepare_mesh
from lib.helper.cloud_operations import cloud2cloud_err, display_dist, m2c_dist_rough


def main(base_name, folder_name, out_file="eval_data", seed=0):
    exp_paths = os.listdir(f"{DATA_DIR}/{base_name}")

    brick_ids = [folder.split("_id-")[-1] for folder in exp_paths]

    # Create empty dataframe with column names
    eval_data = pd.DataFrame(index=brick_ids, columns=["volume", "z_offset", "dist", "completeness"])

    for f_idx, folder in enumerate(exp_paths):
        print(f_idx)
        exp_dir = f"{DATA_DIR}/{base_name}/{folder}"
        if os.path.isdir(exp_dir):
            brick_id = folder.split("_id-")[-1]

            # load reconstructed point cloud
            pc_source = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")

            # load model
            scale_factor = 0.1
            mesh = load_stl(brick_id)
            mesh = prepare_mesh(mesh, scale_factor=scale_factor, random=bool(seed), seed=seed)

            # correct wrong scaling
            pc_source.scale(scale_factor, pc_source.get_center())
            pc_source.scale(1 / scale_factor, np.zeros(3))
            mesh.scale(1 / scale_factor, np.zeros(3))

            # convert to point cloud
            num_samples = 100_000
            pc_target = mesh.sample_points_uniformly(num_samples)
            pc_target.estimate_normals()

            # compute distance error
            dist = np.max(np.array(pc_source.compute_point_cloud_distance(pc_target)))

            # compute completeness
            min_dist = 0.1
            t2s_dists = np.array(pc_target.compute_point_cloud_distance(pc_source))
            idcs_recon = np.where(t2s_dists < min_dist)[0]
            completeness = len(idcs_recon) / len(t2s_dists)

            # visualize completeness
            # colors = np.full((num_samples, 3), np.array([1., 0, 0]))
            # colors[idcs_recon, :] = np.array([0., 1., 0.])
            # pc_target.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pc_target], mesh_show_wireframe=True)

            # rescale was based on the pc center, so we correct this
            pc_tmp = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
            pc_tmp.scale(scale_factor, pc_tmp.get_center())
            z_offset = (
                np.min(np.array(pc_tmp.points)[:, -1]) / scale_factor
            )  # distance from the lowest point to belt in cm

            # fill evaluation data
            pc_source = prepare_cloud(pc_source)
            eval_data.at[brick_id, "volume"] = compute_obb_volume(pc_source)
            eval_data.at[brick_id, "z_offset"] = z_offset
            eval_data.at[brick_id, "dist"] = dist
            eval_data.at[brick_id, "completeness"] = completeness

    directory = f"{DATA_DIR}/{folder_name}"
    with open(f"{directory}/{out_file}.pkl", "wb") as f:
        pickle.dump(eval_data, f)


def plot_error(folder_name, file_name="eval_data"):
    directory = f"{DATA_DIR}/{folder_name}"

    with open(f"{directory}/{file_name}.pkl", "rb") as f:
        df = pickle.load(f)
    file_num = len(df)
    df = df.where(df["z_offset"] < 0.1)
    df = df.dropna()
    print(f"Took {len(df)} of {file_num} files ({100 * len(df) / file_num:.2f}% remaining)")

    # from matplotlib import rc
    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # ## for Palatino and other serif fonts use:
    # # rc('font',**{'family':'serif','serif':['Palatino']})
    # rc('text', usetex=True)

    sns.set()
    fontsize = 16
    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]}, sharey=True)
    fig.set_size_inches(15, 5)

    # create violin plot
    # sns.boxplot(data=eval_data, ax=ax, showfliers=False)
    # sns.scatterplot(ax=axes[0], data=df, x="volume", y="dist")
    sns.regplot(
        ax=axes[0],
        x=df["volume"].astype(float),
        y=df["dist"].astype(float),
        order=1,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "black"},
    )
    sns.boxplot(ax=axes[1], data=df, y="dist", showfliers=False)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # compute percentile
    axes[0].set(xscale="log")
    # axes[0].set(yscale="log")

    axes[0].set_ylabel("Mean distance [cm]", fontsize=fontsize)
    axes[1].set_ylabel("", fontsize=fontsize)
    axes[0].set_xlabel("Volume [$\mathrm{cm}^{3}$]", fontsize=fontsize)
    # post process and show
    plt.tight_layout()
    plt.show()


def plot_results(folder_name, file_name="eval_data"):
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

    sns.set()
    fontsize = 16
    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]}, sharey=True)
    fig.set_size_inches(15, 5)

    # create violin plot
    sns.regplot(
        ax=axes[0],
        x=df["volume"].astype(float),
        y=df["completeness"].astype(float),
        order=1,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "black"},
    )
    sns.boxplot(ax=axes[1], data=df, y="completeness", showfliers=False)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    axes[0].set(xscale="log")
    axes[0].set_ylabel("Reconstruction rate [-]", fontsize=fontsize)
    axes[1].set_ylabel("", fontsize=fontsize)
    axes[0].set_xlabel("Volume [$\mathrm{cm}^{3}$]", fontsize=fontsize)
    # post process and show
    plt.tight_layout()
    plt.show()


def combined_plot(folder_name, file_name="eval_data"):
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
    print(f"Took {len(df)} of {file_num} files ({100 * len(df) / file_num:.2f}% remaining)")


    fig, axes = plt.subplots(2, 2, gridspec_kw={"width_ratios": [2, 1]})
    fig.set_size_inches(15, 7)
    # create violin plot
    # sns.boxplot(data=eval_data, ax=ax, showfliers=False)
    # sns.scatterplot(ax=axes[0], data=df, x="volume", y="dist")
    sns.regplot(
        ax=axes[0][0],
        x=df["volume"].astype(float),
        y=df["dist"].astype(float),
        order=1,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "black"},
    )
    sns.boxplot(ax=axes[0][1], data=df, y="dist", showfliers=False)
    # axes[0][1].set_ylim(axes[0][0].get_ylim())
    # axes[0][1].get_yaxis().set_visible(False)
    axes[0][1].axes.get_yaxis().set_ticklabels([])
    axes[0][1].get_shared_y_axes().join(axes[0][1], axes[0][0])
    
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # compute percentile
    axes[0][0].set(xscale="log")
    # axes[0].set(yscale="log")

    axes[0][0].set_ylabel(r"\textbf{Mean distance [cm]}")
    axes[0][1].set_ylabel("")
    axes[0][0].set_xlabel("")
    # post process and show
    # axes[1][0].get_shared_x_axes().join(axes[1][0], axes[0][0])
    axes[0][0].axes.get_xaxis().set_ticklabels([])

    # create violin plot
    sns.regplot(
        ax=axes[1][0],
        x=df["volume"].astype(float),
        y=df["completeness"].astype(float),
        order=1,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "black"},
    )
    sns.boxplot(ax=axes[1][1], data=df, y="completeness", showfliers=False)
    axes[1][1].axes.get_yaxis().set_ticklabels([])
    axes[1][1].get_shared_y_axes().join(axes[1][1], axes[1][0])

    # axes[0][0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    x = 0.7
    axes[0][0].set_title(r"\textbf{(a)}", x=x)
    axes[1][0].set_title(r"\textbf{(b)}", x=x)

    axes[1][0].set(xscale="log")
    axes[1][0].set_ylabel(r"\textbf{Reconstruction rate [-]}")
    axes[1][1].set_ylabel("")
    axes[1][0].set_xlabel(r"\textbf{Volume [$\mathbf{cm^3}$]}")

    fig.align_ylabels(axes)
    plt.tight_layout()
    plt.show()


def display_recon(base_name):
    exp_paths = os.listdir(f"{DATA_DIR}/{base_name}")

    idcs = [5]
    for f_idx, folder in enumerate(exp_paths):
        # if f_idx not in idcs:
        #     continue
        exp_dir = f"{DATA_DIR}/{base_name}/{folder}"
        if os.path.isdir(exp_dir):
            brick_id = folder.split("_id-")[-1]

            # load reconstructed point cloud
            pc_source = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")

            # load model
            scale_factor = 0.1
            mesh = load_stl(brick_id)
            mesh = prepare_mesh(mesh, scale_factor=scale_factor, random=bool(seed), seed=seed)
            mesh.compute_triangle_normals()
            pc_target = mesh.sample_points_uniformly(100_000)

            # correct wrong scaling
            pc_source.scale(scale_factor, pc_source.get_center())
            pc_source.scale(1 / scale_factor, np.zeros(3))
            mesh.scale(1 / scale_factor, np.zeros(3))

            # pc_source.estimate_normals()

            # convert to point cloud

            # compute distance error
            rot_center = mesh.get_center()

            rot = rot_mat((1., 0., 0.), -45.)

            mesh.rotate(rot, rot_center)
            pc_source.rotate(rot, rot_center)

            dist = np.array(pc_source.compute_point_cloud_distance(pc_target))
            dist = m2c_dist_rough(mesh, pc_source)
            dist *= 0.1
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(mesh.vertices), 3)) * 0.5)
            mesh.scale(0.99, mesh.get_center())
            display_dist(dist, pc_source, mesh, lightness=0.9, coord_axes=False)
            print(brick_id)
            # compute completeness
            # min_dist = 0.1
            # t2s_dists = np.array(pc_target.compute_point_cloud_distance(pc_source))
            # idcs_recon = np.where(t2s_dists < min_dist)[0]
            # completeness = len(idcs_recon) / len(t2s_dists)

            # visualize completeness
            # colors = np.full((num_samples, 3), np.array([1., 0, 0]))
            # colors[idcs_recon, :] = np.array([0., 1., 0.])
            # pc_target.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pc_source, mesh], mesh_show_wireframe=False)


if __name__ == "__main__":
    folder_name = "eval_brick_recon"
    file_name = "eval_data_mean"
    seed = 2
    base_name = f"eval_retrieval/seed_{str(seed).zfill(4)}"
    # main(base_name, folder_name, out_file=file_name, seed=seed)
    # plot_error(folder_name, file_name=file_name)
    # plot_results(folder_name, file_name=file_name)
    # combined_plot(folder_name, file_name=file_name)
    display_recon(base_name)

