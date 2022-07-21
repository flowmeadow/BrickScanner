#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : evaluate data from recon_test script
@File      : recon_test_eval.py
@Project   : BrickScanner
@Time      : 16.07.22 18:14
@Author    : flowmeadow
"""
from typing import Tuple

import numpy as np

from scripts.sim_recon_test import recon_test
from definitions import *
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def generate_data(num_experiments: int):
    """
    Generates random recon_test data for evaluation
    :param num_experiments: number of experiments
    """
    for idx in range(num_experiments):
        print(f"\n\nGenerating data {idx + 1} of {num_experiments}")
        recon_test(folder_name=f"recon_test/eval/exp_{str(idx).zfill(5)}", automated=True, gen_rand=True)


def compute_angle(T_W1: np.ndarray, T_W2: np.ndarray) -> float:
    """
    Computes the angle between two z axes from transformation matrices
    :param T_W1: transformation matrix (4, 4)
    :param T_W2: transformation matrix (4, 4)
    :return: angle
    """
    axis = np.array([0.0, 0.0, 1.0])
    return np.arccos(np.dot(T_W1[:3, :3] @ axis, T_W2[:3, :3] @ axis)) * (180.0 / np.pi)


def plot_data(df_angles: pd.DataFrame, df_dists: pd.DataFrame) -> plt.Figure:
    """
    Plots data from data frames in one figure
    :param df_angles: dataframe containing angles between z axes and the corresponding mean error distances
    :param df_dists: dataframe containing error distances
    :return: matplotlib figure object
    """
    # init figure
    sns.set()
    fontsize = 16
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)

    # create violin plot
    sns.violinplot(data=df_dists, x="dists", ax=ax1)
    ax1.set_xlabel("error distances [-]", fontsize=fontsize)

    # create regression plot
    sns.regplot(
        data=df_angles,
        x="angles",
        y="mean_dists",
        order=2,
        scatter_kws={"alpha": 0.1},
        line_kws={"color": "black"},
        ax=ax2,
    )
    ax2.set_xlabel("angle $\gamma$ [°]", fontsize=fontsize)
    ax2.set_ylabel("mean error distances [-]", fontsize=fontsize)

    # post process and show
    plt.tight_layout()
    plt.show()


def prepare_data(angle_thresh=5, max_dist=0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and prepares the generated data for plotting
    :param angle_thresh: consider only angles that are between angle_thresh and 180 - angle_thresh
    :param max_dist: remove distance values below this value for plotting
    :return:
    """
    directory = f"{DATA_DIR}/recon_test/eval"
    experiments = sorted(os.listdir(directory))

    # load and compute data
    dists_total = []
    dists_mean = []
    angles = []
    for idx, exp in enumerate(experiments):
        errors = np.load(f"{directory}/{exp}/errors.npy")
        T_W1 = np.load(f"{directory}/{exp}/T_W1.npy")
        T_W2 = np.load(f"{directory}/{exp}/T_W2.npy")

        # take square root to get distances from errors
        dists = np.sqrt(errors)

        # compute angle
        angle = compute_angle(T_W1, T_W2)

        # append data
        angles.append(angle)
        dists_mean.append(np.mean(dists))  # compute mean distance for each experiment
        dists_total.append(dists)

    # convert to numpy
    angles = np.array(angles)
    dists_mean = np.array(dists_mean)
    dists_total = np.concatenate(dists_total)

    # remove data with angles outside the threshold
    idcs = np.argwhere(np.logical_and(angle_thresh < angles, 180 - angle_thresh > angles)).flatten()
    angles = angles[idcs]
    dists_mean = dists_mean[idcs]

    # remove distances higher than max_dist for violin plot
    idcs = np.argwhere(dists_total < max_dist).flatten()
    dists_total = dists_total[idcs]

    # prepare data frames
    df_angles = pd.DataFrame(dict(angles=angles, mean_dists=dists_mean))
    df_dists = pd.DataFrame(dict(dists=dists_total))
    return df_angles, df_dists


def show_exp_data(idx: int):
    """
    Shows a saved experimental setup
    :param idx: index of the experiment
    """
    directory = f"{DATA_DIR}/recon_test/eval"
    exp = f"exp_{str(idx).zfill(5)}"
    T_W1 = np.load(f"{directory}/{exp}/T_W1.npy")
    T_W2 = np.load(f"{directory}/{exp}/T_W2.npy")
    print(f"Angle: {compute_angle(T_W1, T_W2)}")
    recon_test(folder_name=f"recon_test/tmp", T_W1=T_W1, T_W2=T_W2)


if __name__ == "__main__":
    # generate_data(10000)
    # prepare_data()
    # show_exp_data(3044)
    plot_data(*prepare_data())
