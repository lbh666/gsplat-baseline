from abc import abstractmethod

import torch
from torch.utils.data import Dataset, default_collate
from dataclasses import dataclass
from typing import Literal, cast
from .dataset_readers import CameraInfo
import math
from utils.utils import get_viewmat
from loguru import logger as guru
from tqdm import tqdm
import imageio.v3 as iio
import torch.nn.functional as F
import numpy as np
import re
import os.path as osp
from .dataset_readers import BasicPointCloud

class BaseDataset(Dataset):
    def __init__(
            self, 
            data:list[CameraInfo], 
            pcd: BasicPointCloud = None,
            downscale_factor: int = 1,
            keep_original_world_coordinate: bool = False,
            nerf_normalization: dict = None
            ):
        self.data = data
        self.pcd = None
        if pcd:
            self.pcd = pcd
        applied_transform = np.eye(4, dtype=np.float32)
        if not keep_original_world_coordinate:
            applied_transform = applied_transform[np.array([0, 2, 1, 3]), :]
            applied_transform[2, :] *= -1
        self.applied_transform = applied_transform
        
        # load cameras
        Ks, c2ws = [], []
        for viewpoint_camera in self.data:
            tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
            tanfovy = math.tan(viewpoint_camera.FovY * 0.5)
            focal_length_x = viewpoint_camera.width / (2 * tanfovx)
            focal_length_y = viewpoint_camera.height / (2 * tanfovy)
            Ks.append(np.array(
                [
                    [focal_length_x, 0, viewpoint_camera.width / 2.0],
                    [0, focal_length_y, viewpoint_camera.height / 2.0],
                    [0, 0, 1],
                ],
            ))
            w2c = np.eye(4)
            w2c[:3, :3] = viewpoint_camera.R.transpose()
            w2c[:3, 3] = viewpoint_camera.T
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            c2w = applied_transform @ c2w
            c2ws.append(c2w)
        
        self.c2ws, self.Ks = torch.tensor(c2ws).float(), torch.tensor(Ks).float()

        self.scale_factor = 1.0
        self.translate = torch.from_numpy(applied_transform)[:3, :3] @ torch.tensor([[0, 0, 0]], dtype=torch.float32).T
        if nerf_normalization:
            self.translate = torch.from_numpy(applied_transform)[:3, :3] @ torch.from_numpy(nerf_normalization['translate'])[..., None]
            self.c2ws[:, :3, 3] += self.translate[..., 0]
            guru.info(f"Auto center scene with translation {self.translate=}")
            self.scale_factor /= float(torch.max(torch.abs(self.c2ws[:, :3, 3])))
            guru.info(f"Auto scene scale factor of {self.scale_factor=}")
        self.c2ws[:, :3, 3] *= self.scale_factor
        self.Ks[:, :2] /= downscale_factor

        self.fx, self.fy, self.cx, self.cy = self.Ks[0][0,0], self.Ks[0][1,1], self.Ks[0][0,2], self.Ks[0][1,2]
        self.viewmats = get_viewmat(self.c2ws)
        # load imgs
        imgs = torch.from_numpy(
                np.array(
                    [
                        viewpoint_camera.image
                        for viewpoint_camera in tqdm(
                            self.data,
                            desc=f"Loading images",
                            leave=False,
                        )
                    ],
                )
            )
        if downscale_factor != 1:
            imgs = F.interpolate(imgs.permute(0,3,1,2), scale_factor=1./downscale_factor).permute(0,2,3,1)
        self.imgs = imgs[..., :3] / 255.0

        self.frame_names = [viewpoint_camera.image_name for viewpoint_camera in self.data]
        # load metadata
        pattern = r"(?:frame_)?(\d+)"
        time_ids = [float(re.search(pattern, osp.basename(viewpoint_camera.image_path).split('.')[0]).group(1)) for viewpoint_camera in self.data]
        self.time_ids = torch.tensor(time_ids) - time_ids[0]
        guru.info(f"{self.time_ids.min()=} {self.time_ids.max()=}")
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data = {
            "frame_names": self.frame_names[index],
            "ts": self.time_ids[index],
            "c2ws": self.c2ws[index],
            "Ks": self.Ks[index],
            "imgs": self.imgs[index],
            "viewmats": self.viewmats[index]
        }

        return data

    def get_pcd(self):

        points3D = torch.from_numpy(np.asarray(self.pcd.points, dtype=np.float32)) * self.scale_factor
        if hasattr(self, "applied_transform"):
            points3D = (torch.from_numpy(self.applied_transform).float()[:3, :3] @ points3D[..., None]).squeeze(-1)
        points3D_normal = torch.ones_like(points3D) / 3 ** (1/2)
        points3D_rgb = torch.from_numpy(np.array(self.pcd.colors, dtype=np.float32))

        return points3D, points3D_normal, points3D_rgb
    