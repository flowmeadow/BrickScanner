#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : eval_pca_obb.py
@Project   : BrickScanner
@Time      : 02.08.22 15:03
@Author    : flowmeadow
"""

import pickle
import time

import cv2
import numpy as np
import pandas as pd
import yaml
from glpg_flowmeadow.transformations.methods import rotate_vec
from matplotlib import pyplot as plt

from lib.helper.cloud_operations import draw_point_clouds, rotate_random
from lib.helper.lego_bricks import get_base_bricks, load_stl
from lib.retrieval.bounding_box import compute_obb_edges, PCA_based_alignment, compute_obb_volume
from lib.retrieval.cloud_alignment import prepare_cloud, find_model
from scripts.sim_recon_setup import prepare_mesh, double_side_recon

import open3d as o3d
from definitions import *
import seaborn as sns


def generate_pca_data(folder_name: str, files: list):
    directory = f"{DATA_DIR}/{folder_name}"
    tries = 10

    methods = ["all", "min", "max"]
    data_dict = dict()
    for f_idx, file in enumerate(files):
        brick_id = file[:-4]
        with open(f"{directory}/results.pkl", "rb") as f:
            data_dict: dict = pickle.load(f)

        if file in data_dict.keys() and data_dict[file]["completed"]:
            continue

        print(f"processing file {file} ({f_idx + 1}|{len(files)})")
        data_dict[file] = dict(completed=False)
        mesh = load_stl(brick_id)
        mesh.compute_vertex_normals()
        orig_edges = compute_obb_edges(mesh)

        # TODO: filter out 2D models moe efficient
        try:
            mesh, _ = mesh.compute_convex_hull()
        except Exception as e:
            del data_dict[file]
            continue

        for method in methods:
            for source in ["_sampled", "_hull", "_hull_sampled"]:

                times = []
                edges = []
                for t_idx in range(tries):
                    if "hull" in source:
                        mesh, _ = mesh.compute_convex_hull()
                    if "sampled" in source:
                        pc = mesh.sample_points_uniformly(100_000)
                    else:
                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(mesh.vertices)

                    start = time.time()
                    pc = PCA_based_alignment(pc, method=method)
                    times.append(time.time() - start)

                    edges.append(compute_obb_edges(pc))

                data_dict[file][f"pca_{method}{source}_edges"] = np.mean(np.array(edges), axis=0)
                data_dict[file][f"pca_{method}{source}_time"] = np.mean(np.array(times))

                err = (np.mean(np.array(edges), axis=0) / orig_edges - 1.0) ** 2
                err = np.sum(err)
                t = np.mean(np.array(times)) * 1000
                print(f"-- {f'pca_{method}{source}:'.ljust(25)} Error: {err:.3E}\t Time: {t:.2f} ms")
        data_dict[file]["completed"] = True
        with open(f"{directory}/results.pkl", "wb") as f:
            pickle.dump(data_dict, f)


def evaluate_pca_data(folder_name, cloud_files):
    directory = f"{DATA_DIR}/{folder_name}"
    with open(f"{directory}/results.pkl", "rb") as f:
        data_dict: dict = pickle.load(f)

    errors = dict()
    for id, path in cloud_files.items():
        print(f"evaluating brick {id}")

        target_data = data_dict[f"{id}.stl"]
        pc_source = o3d.io.read_point_cloud(path)

        methods = ["all", "min", "max"]
        for method in methods:
            pc_s = PCA_based_alignment(pc_source, method=method)
            edges_s = compute_obb_edges(pc_s)
            for source in ["_sampled", "_hull", "_hull_sampled"]:
                edges_t = target_data[f"pca_{method}{source}_edges"]
                err = (edges_t / edges_s - 1.0) ** 2
                err = np.sum(err)

                if f"{method}{source}" in errors.keys():
                    errors[f"{method}{source}"].append(err)
                else:
                    errors[f"{method}{source}"] = [err]
    with open(f"{directory}/errors.pkl", "wb") as f:
        pickle.dump(errors, f)
    return


def plot_times(folder_name):
    sns.set()
    sns.set(font_scale=1.5)
    params = {'text.usetex': True,
              'font.family': 'serif',
              }
    plt.rcParams.update(params)

    directory = f"{DATA_DIR}/{folder_name}"

    with open(f"{directory}/results.pkl", "rb") as f:
        data = pickle.load(f)

    t_dict = dict()
    for file, f_dict in data.items():
        # compute volume error
        for key, item in f_dict.items():
            if key.endswith("_time"):
                new_key = key[4 : -len("_time")]
                if new_key in t_dict.keys():
                    t_dict[new_key].append(item * 1000)
                else:
                    t_dict[new_key] = [item * 1000]

    fig, axes = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(15, 4)
    axes[1].set_title(r"\textbf{(a)}")
    # create violin plot
    for idx, (ax, source, ax_label) in enumerate(
        zip(axes, ["_sampled", "_hull", "_hull_sampled"], ["Sampled", "Hull", "Sampled from hull"])
    ):
        df = pd.DataFrame()
        for key, val in t_dict.items():
            print(key)
            if key[3:] == source:
                label = f"{key[:-len(source)].upper()}-PCA"
                df[label] = val
        print(df)
        sns.boxplot(data=df, ax=ax, showfliers=False)
        ax.set_xticks([])
        # ax.set_xlabel(ax_label)
        if idx == 0:
            ax.set_ylabel(r"\textbf{Computation time [ms]}")

    # post process and show
    plt.tight_layout()
    plt.show()


def plot_volumes(folder_name):
    sns.set()
    sns.set(font_scale=1.5)
    params = {'text.usetex': True,
              'font.family': 'serif',
              }
    plt.rcParams.update(params)

    directory = f"{DATA_DIR}/{folder_name}"

    with open(f"{directory}/results.pkl", "rb") as f:
        data = pickle.load(f)

    v_dict = dict()
    for file, f_dict in data.items():

        # find minimum volume
        min_v = np.inf
        for key, item in f_dict.items():
            if key.endswith("_edges"):
                v = np.prod(item)
                if v < min_v:
                    min_v = v

        # compute volume error
        for key, item in f_dict.items():
            if key.endswith("_edges"):
                new_key = key[4 : -len("_edges")]
                v = np.prod(item)
                v = v / min_v - 1
                if new_key in v_dict.keys():
                    v_dict[new_key].append(v)
                else:
                    v_dict[new_key] = [v]

    fontsize = 16
    fig, axes = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(15, 4)
    axes[1].set_title(r"\textbf{(b)}")

    # create violin plot
    for idx, (ax, source, ax_label) in enumerate(
        zip(axes, ["_sampled", "_hull", "_hull_sampled"], ["Sampled", "Hull", "Sampled from hull"])
    ):
        df = pd.DataFrame()
        for key, val in v_dict.items():
            print(key)
            if key[3:] == source:
                label = f"{key[:-len(source)].upper()}-PCA"
                df[label] = val
        sns.boxplot(data=df, ax=ax, showfliers=False)

        ax.set_xticks([])
        # ax.set_xlabel(ax_label)
        if idx == 0:
            ax.set_ylabel(r"\textbf{Normalized volume error [-]}")

    # post process and show
    plt.tight_layout()
    plt.show()


def plot_errors(folder_name):
    sns.set()
    sns.set(font_scale=1.5)
    params = {'text.usetex': True,
              'font.family': 'serif',
              }
    plt.rcParams.update(params)

    directory = f"{DATA_DIR}/{folder_name}"

    with open(f"{directory}/errors.pkl", "rb") as f:
        errors = pickle.load(f)

    fig, axes = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(15, 4)
    axes[1].set_title(r"\textbf{(c)}")

    # create violin plot
    for idx, (ax, source, ax_label) in enumerate(
        zip(axes, ["_sampled", "_hull", "_hull_sampled"], ["Sampled", "Hull", "Sampled from hull"])
    ):
        df = pd.DataFrame()
        for key, val in errors.items():
            if key[3:] == source:
                label = f"{key[:-len(source)].upper()}-PCA"
                df[label] = val

        sns.boxplot(data=df, ax=ax, showfliers=False)
        # ax.set_xticks([])
        ax.set_xlabel(r"\textbf{" + ax_label + r"}")
        if idx == 0:
            ax.set_ylabel(r"\textbf{Summed edge error [-]}")

    # post process and show
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    base_name = f"eval_pca_obb"
    # files = get_base_bricks()
    #
    # generate_pca_data(base_name, files)
    # seed = 2
    # recon_dir = f"{DATA_DIR}/eval_retrieval/seed_{str(seed).zfill(4)}"
    # cloud_files = dict()
    # for directory in os.listdir(recon_dir):
    #     if os.path.isdir(f"{recon_dir}/{directory}"):
    #         cloud_files[directory[-4:]] = f"{recon_dir}/{directory}/recon_pc.pcd"
    # evaluate_pca_data(base_name, cloud_files)
    plot_errors(base_name)
    plot_volumes(base_name)
    plot_times(base_name)


    # mesh = load_stl("3148")
    # mesh.vertices = o3d.utility.Vector3dVector(rotate_vec(np.array(mesh.vertices), (1.0, 0.5, 0.0), 235.0))
    # mesh.compute_vertex_normals()
    # pc = mesh.sample_points_uniformly(1_000)
    #
    # def draw_it(pc, mesh):
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()
    #     vis.add_geometry(pc)
    #     vis.add_geometry(mesh)
    #     settings = vis.get_render_option()
    #     settings.point_size = 10.0
    #     settings.mesh_show_wireframe = True
    #     vis.run()
    #     vis.capture_screen_image("test.png")
    #     vis.destroy_window()
    #     img = np.array(o3d.io.read_image("test.png"))
    #     os.remove("test.png")
    #     return img
    #
    # img_1 = draw_it(pc, mesh)
    # mesh, _ = mesh.compute_convex_hull()
    # mesh.compute_vertex_normals()
    #
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(mesh.vertices)
    # a = o3d.visualization.RenderOption()
    # a.point_size = 44
    # img_2 = draw_it(pc, mesh)
    #
    # pc = mesh.sample_points_uniformly(1_000)
    # img_3 = draw_it(pc, mesh)
    #
    # def crop(img, x_min, x_max, y_min, y_max):
    #     return img[y_min:y_max, x_min:x_max]
    # imgs = []
    # for img in [img_1, img_2, img_3]:
    #     imgs.append(crop(img, 650, 1400, 170, 960))
    #
    # a = cv2.hconcat(imgs)
    # cv2.imshow("frame", a)
    # cv2.waitKey()
