import torch
from torch.utils.data import Dataset, default_collate
from typing import Literal
import imageio.v3 as iio
from utils.utils import get_viewmat
import os.path as osp
import numpy as np
from utils.camera_utils import get_distortion_params
from loguru import logger as guru
from tqdm import tqdm
from skimage.transform import resize
import torch.nn.functional as F
import re
from .base_dataset import BaseDataset
from dataclasses import dataclass
import open3d as o3d
from utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_points3D_text,
    parse_colmap_camera_params
)


@dataclass
class ColmapDataConfig:
    data_dir: str
    downscale_factor: int = 1
    auto_scale_poses: bool = True
    split: Literal["train", "val"] = "train"


class ColmapDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str, 
        downscale_factor: int = 1,
        auto_scale_poses: bool = True,
        split: Literal["train", "val"] = "train",
        keep_original_world_coordinate: bool = False
    ):
        self.data_dir = data_dir
        self.downscale_factor = downscale_factor
        if self.downscale_factor != 1:
            guru.info(f"Image downscale factor of {self.downscale_factor=}")
        cam_id_to_camera = read_cameras_binary(osp.join(self.data_dir, "sparse/0/cameras.bin"))
        im_id_to_image = read_images_binary(osp.join(self.data_dir, "sparse/0/images.bin"))

        assert set(cam_id_to_camera.keys()) == {1}, "only support single camera"

        meta = parse_colmap_camera_params(cam_id_to_camera[1])

        assert meta["camera_model"] == "OPENCV", "only support OPENCV camera model currently"

        # load camera inrinsic
        self.fx = float(meta["fl_x"])
        self.fy = float(meta["fl_y"])
        self.cx = float(meta["cx"])
        self.cy = float(meta["cy"])
        height = int(meta["h"])
        width = int(meta["w"])

        applied_transform = np.eye(4)
        if not keep_original_world_coordinate:
            applied_transform = applied_transform[np.array([0, 2, 1, 3]), :]
            applied_transform[2, :] *= -1
        self.applied_transform = applied_transform

        fnames = []
        Ks, c2ws = [], []
        for im_id, im_data in im_id_to_image.items():
            Ks.append(
                    [
                        [self.fx, 0.0, self.cx],
                        [0.0, self.fy, self.cy],
                        [0.0, 0.0, 1.0],
                    ]
                )

            rotation = qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            c2w = applied_transform @ c2w
            
            if osp.exists(osp.join(self.data_dir, 'images', im_data.name)):
                fnames.append(im_data.name)
                c2ws.append(c2w)
        # sort by file names
        inds = np.argsort(fnames)
        self.frame_names = [osp.join(self.data_dir, "images", fnames[ind]) for ind in inds]
        c2ws = [c2ws[ind] for ind in inds]

        self.Ks = torch.tensor(Ks)
        self.Ks[:, :2] /= downscale_factor
        self.c2ws = torch.from_numpy(np.array(c2ws)).float()

        self.scale_factor = 1.0
        if auto_scale_poses:
            self.scale_factor /= float(torch.max(torch.abs(self.c2ws[:, :3, 3])))
            guru.info(f"Auto scene scale factor of {self.scale_factor=}")
        self.c2ws[:, :3, 3] *= self.scale_factor
        self.w2cs = self.c2ws.inverse()
        self.viewmats = get_viewmat(self.c2ws)

        # load images
        imgs = torch.from_numpy(
                np.array(
                    [
                        iio.imread(frame_name)
                        for frame_name in tqdm(
                            self.frame_names,
                            desc=f"Loading images",
                            leave=False,
                        )
                    ],
                )
            )
        if downscale_factor != 1:
            imgs = F.interpolate(imgs.permute(0,3,1,2), scale_factor=1./downscale_factor).permute(0,2,3,1)
        self.imgs = imgs[..., :3] / 255.0
        self.valid_masks = None
        if self.imgs.shape[-1] == 4:
            self.valid_masks = imgs[..., 3] / 255.0
        guru.info(f"{self.imgs.shape=}  {self.c2ws.shape}")

        # load metadata
        pattern = r"(?:frame_)?(\d+)"
        time_ids = [float(re.search(pattern, osp.basename(fname).split('.')[0]).group(1)) for fname in self.frame_names]
        self.time_ids = torch.tensor(time_ids) - time_ids[0]
        self.time_ids /= time_ids[-1]
        guru.info(f"{self.time_ids.min()=} {self.time_ids.max()=}")

    def get_pcd(
        self, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if osp.exists(osp.join(self.data_dir, "sparse/0/points3D.bin")):
            pcd = read_points3D_binary(osp.join(self.data_dir, "sparse/0/points3D.bin"))
        elif osp.exists(osp.join(self.data_dir, "sparse/0/points3D.txt")):
            pcd = read_points3D_text(osp.join(self.data_dir, "sparse/0/points3D.txt"))

        points3D = torch.from_numpy(np.array([p.xyz for p in pcd.values()], dtype=np.float32)) * self.scale_factor
        if hasattr(self, "applied_transform"):
            points3D = (torch.from_numpy(self.applied_transform).float()[:3, :3] @ points3D[..., None]).squeeze(-1)
        points3D_normal = torch.ones_like(points3D) / 3 ** (1/2)
        points3D_rgb = torch.from_numpy(np.array([p.rgb for p in pcd.values()], dtype=np.float32)) / 255.

        return points3D, points3D_normal, points3D_rgb
    
    def __len__(self):
        return self.imgs.shape[0]
    
    def get_c2ws(self) -> torch.Tensor:
        return self.c2ws
    
    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs
    
    def get_Ks(self) -> torch.Tensor:
        return self.Ks
    
    def num_frames(self) -> int:
        return self.imgs.shape[0]

    def get_image(self, index: int) -> torch.Tensor:
        return self.imgs
    
    def __getitem__(self, index: int):
        data = {
            "frame_names": self.frame_names[index],
            "ts": self.time_ids[index],
            "c2ws": self.c2ws[index],
            "w2cs": self.w2cs[index],
            "Ks": self.Ks[index],
            "imgs": self.imgs[index],
            "viewmats": self.viewmats[index]
        }

        return data

            


