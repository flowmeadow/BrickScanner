#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : eval_retrievel.py
@Project   : BrickScanner
@Time      : 25.07.22 15:53
@Author    : flowmeadow
"""
import pickle

import numpy as np
import yaml

from lib.helper.cloud_operations import draw_point_clouds
from lib.helper.lego_bricks import get_base_bricks, load_stl
from lib.retrieval.bounding_box import compute_obb_edges
from lib.retrieval.cloud_alignment import prepare_cloud, find_model
from scripts.sim_recon_setup import prepare_mesh, double_side_recon

import open3d as o3d
from definitions import *


def generate_retrieval_data(folder_name: str, brick_id: str, settings: dict, seed: int = None):
    directory = f"{DATA_DIR}/{folder_name}"
    # recon settings
    scale_factor = 0.1

    # generate mesh
    mesh = load_stl(brick_id)
    mesh = prepare_mesh(mesh, scale_factor=scale_factor, random=bool(seed), seed=seed)

    # (create images and) reconstruct point cloud
    pc = double_side_recon(folder_name, mesh, **settings)

    # remove outlier
    pc, _ = pc.remove_radius_outlier(nb_points=8, radius=0.01)

    # rescale point cloud back to cm and save point cloud
    pc.scale(1 / scale_factor, pc.get_center())
    o3d.io.write_point_cloud(f"{directory}/recon_pc.pcd", pc)

    # prepare it for alignment
    pc = prepare_cloud(pc)

    # find the best alignment matches
    files, errors, transformations, percentages = ret = find_model(pc, threshold=0.2, max_best=100)

    # save results
    results = dict(files=files, errors=errors, transformations=transformations, percentages=percentages)
    with open(f"{directory}/results.pkl", "wb") as f:
        pickle.dump(results, f)

    return f"{brick_id}.stl" in files


def generate_experiments(settings: dict, seed: int, base_name: str, num_experiments=100):
    os.makedirs(f"{DATA_DIR}/{base_name}", exist_ok=True)
    with open(f"{DATA_DIR}/{base_name}/settings.yml", "w") as f:
        yaml.dump([settings], f)

    # remove large bricks
    max_size = 5.0  # in cm
    with open(f"{DATA_DIR}/base_obb_data.pkl", "rb") as f:
        obb_data = pickle.load(f)
    max_edges = np.array(obb_data["edges"])[:, -1]
    idcs = np.where(max_edges < max_size)[0]
    files = list(np.array(obb_data["files"])[idcs])

    # make a random selection
    np.random.seed(seed)
    np.random.shuffle(files)
    files = files[:num_experiments]

    successes = 0
    for idx, file in enumerate(files):
        brick_id = file[:-4]

        folder_name = f"{base_name}/exp-{str(idx).zfill(5)}_id-{brick_id}"
        if os.path.exists(f"{DATA_DIR}/{folder_name}"):
            if "results.pkl" in os.listdir(f"{DATA_DIR}/{folder_name}"):
                with open(f"{DATA_DIR}/{folder_name}/results.pkl", "rb") as f:
                    results = pickle.load(f)
                if file in results["files"]:
                    successes += 1
                continue

        ret = generate_retrieval_data(folder_name, brick_id, settings, seed=seed)
        if ret:
            successes += 1
        print(f"Retrieval of file {file} {'PASSED' if ret else 'FAILED'}")
        print(f"Success rate: ({successes} of {idx + 1}) {100 * successes / (idx + 1):.2f}")


def evaluate_results(base_name: str):
    with open(f"{DATA_DIR}/base_obb_data.pkl", "rb") as f:
        obb_data = pickle.load(f)

    success = 0
    for folder in os.listdir(f"{DATA_DIR}/{base_name}"):
        exp_dir = f"{DATA_DIR}/{base_name}/{folder}"
        if os.path.isdir(exp_dir):
            brick_id = folder.split("_id-")[-1]
            if "results.pkl" in os.listdir(exp_dir):
                with open(f"{exp_dir}/results.pkl", "rb") as f:
                    results = pickle.load(f)
                if f"{brick_id}.stl" in results["files"]:
                    idcs = np.argsort(results["errors"]).flatten()
                    if f"{brick_id}.stl" == results["files"][idcs[0]]:
                        success += 1
                    else:
                        continue
                        # print(results["percentages"])
                        idcs = np.argsort(results["errors"])
                        best_files = list(np.array(results["files"])[idcs])
                        correct_pos = best_files.index(f'{brick_id}.stl')
                        print(f"Correct guess is at {correct_pos + 1}. position")
                        if correct_pos < 3:
                            print(results["percentages"][idcs][:correct_pos + 1])
                            for g_idx in idcs[:correct_pos + 1]:
                                mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{results['files'][g_idx]}")
                                transf = results["transformations"][g_idx]
                                mesh = mesh.transform(transf)
                                pc_source = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
                                pc_source = prepare_cloud(pc_source)
                                o3d.visualization.draw_geometries([pc_source, mesh], mesh_show_wireframe=True, left=1000, height=800)
                else:
                    print(f"OBB search failed for model {brick_id}")
                    pc_source = o3d.io.read_point_cloud(f"{exp_dir}/recon_pc.pcd")
                    pc_source = prepare_cloud(pc_source)
                    source_edges = compute_obb_edges(pc_source)
                    idx = obb_data["files"].index(f"{brick_id}.stl")
                    target_edges = obb_data["edges"][idx]
                    print(f"\t source edges: {source_edges}")
                    print(f"\t target edges: {target_edges}")

                    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{brick_id}.stl")
                    # mesh, _ = mesh.compute_convex_hull()

                    pc_target: o3d.geometry.PointCloud = mesh.sample_points_uniformly(100_000)
                    pc_target = prepare_cloud(pc_target)

                    draw_point_clouds(pc_source, pc_target)

                    # mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(f"{STL_DIR}/{brick_id}.stl")
                    # mesh, _ = mesh.compute_convex_hull()
                    #
                    # pc_target: o3d.geometry.PointCloud = mesh.sample_points_uniformly(100_000)
                    # pc_target = prepare_cloud(pc_target)
                    #
                    # draw_point_clouds(pc_source, pc_target)

                    files, errors, transformations, percentages = ret = find_model(
                        pc_source, threshold=0.2, max_best=None
                    )
                    best_files = list(np.array(files)[np.argsort(errors)])
                    best_match = best_files[0]
                    print("-----------------------------------")
                    if f"{brick_id}.stl" not in best_files:
                        print(f"WARNING: file not in OBB selection")
                    else:
                        print(f"Correct guess is at {best_files.index(f'{brick_id}.stl') + 1}. position")
                    print(f"Source:     {brick_id}")
                    print(f"Best Guess: {best_match[:-4]}")
                    print("-----------------------------------")

                # errors = results["errors"]
    print(success)

if __name__ == "__main__":
    seed = 2
    settings = dict(
        automated=True,
        generate_new=True,
        cam_dist=1.2,
        cam_alpha=10.0,
        cam_beta=45.0,
        step=0.01,
        gap_window=10,
        y_extension=1.0,
    )
    base_name = f"eval_retrieval/seed_{str(seed).zfill(4)}"
    # generate_experiments(settings, seed, base_name, num_experiments=1000)
    evaluate_results(base_name)
