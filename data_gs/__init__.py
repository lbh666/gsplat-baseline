#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from data_gs.dataset_readers import sceneLoadTypeCallbacks
from dataclasses import dataclass
from typing import Literal
from .base_dataset import BaseDataset

@dataclass
class DataConfig:
    data_dir: str
    downscale_factor: int = 1
    auto_scale_poses: bool = True
    split: Literal["train", "val"] = "train"
    type: Literal["colmap", "bender", "dtu", "nerfies", "plenopticVideo"] = "colmap"
    images:str = "images"
    white_background: bool = True

def get_train_val_datasets(
    data_cfg: DataConfig, load_val: bool = True, scene_norm: bool = True
) -> BaseDataset:
    if data_cfg.type == "colmap" or os.path.exists(os.path.join(data_cfg.data_dir, "sparse")):
        scene_info = sceneLoadTypeCallbacks.Colmap(data_cfg.data_dir, data_cfg.images, load_val)
    elif data_cfg.type == "bender" or os.path.exists(os.path.join(data_cfg.data_dir, "transforms_train.json")):
        print("Assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks.Blender(data_cfg.data_dir, data_cfg.white_background, load_val)
    elif data_cfg.type == "dtu" or os.path.exists(os.path.join(data_cfg.data_dir, "cameras_sphere.npz")):
        print("Assuming DTU data set!")
        scene_info = sceneLoadTypeCallbacks.DTU(data_cfg.data_dir, "cameras_sphere.npz", "cameras_sphere.npz")
    elif data_cfg.type == "nerfies" or os.path.exists(os.path.join(data_cfg.data_dir, "dataset.json")):
        print("Assuming Nerfies data set!")
        scene_info = sceneLoadTypeCallbacks.nerfies(data_cfg.data_dir, load_val)
    elif data_cfg.type == "plenopticVideo" or os.path.exists(os.path.join(data_cfg.data_dir, "poses_bounds.npy")):
        print("Assuming Neu3D data set!")
        scene_info = sceneLoadTypeCallbacks.plenopticVideo(data_cfg.data_dir, load_val, 24)
    else:
        assert False, "Could not recognize scene type!"

    nerf_normalization = scene_info.nerf_normalization if scene_norm else None
    train_cameras = BaseDataset(scene_info.train_cameras, scene_info.point_cloud, data_cfg.downscale_factor, nerf_normalization=nerf_normalization)
    test_cameras = BaseDataset(scene_info.test_cameras, data_cfg.downscale_factor, nerf_normalization=nerf_normalization)

    return train_cameras, test_cameras
