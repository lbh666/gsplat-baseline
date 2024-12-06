import torch
from torch.utils.data import Dataset, default_collate
from typing import Literal
import imageio.v3 as iio
from utils.utils import load_from_json
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

@dataclass
class SteroBlurDataConfig:
    data_dir: str
    downscale_factor: int = 1
    auto_scale_poses: bool = True
    split: Literal["train", "val"] = "train"

class SteroBlurDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str, 
        downscale_factor: int = 1,
        auto_scale_poses: bool = False,
        split: Literal["train", "val"] = "train",
    ):
        self.data_dir = data_dir
        self.downscale_factor = downscale_factor
        if self.downscale_factor != 1:
            guru.info(f"Image downscale factor of {self.downscale_factor=}")
        meta = load_from_json(osp.join(data_dir, 'transforms.json'))

        if "camera_model" in meta:
            assert meta["camera_model"] == "OPENCV", "only support OPENCV camera model currently"
        

        # load camera inrinsic
        self.fx = float(meta["fl_x"])
        self.fy = float(meta["fl_y"])
        self.cx = float(meta["cx"])
        self.cy = float(meta["cy"])
        height = int(meta["h"])
        width = int(meta["w"])

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = frame["file_path"]
            fname = osp.basename(filepath)
            if osp.exists(osp.join(self.data_dir, filepath)):
                fnames.append(fname)
        inds = np.argsort(fnames)
        inds = [ind for ind in inds if osp.exists(osp.join(self.data_dir, meta['frames'][ind]['file_path']))]
        self.frame_names = [(meta['frames'][ind]['file_path']) for ind in inds]

        # load cameras
        Ks, c2ws = [], []
        for ind in inds:
            Ks.append(
                    [
                        [self.fx, 0.0, self.cx],
                        [0.0, self.fy, self.cy],
                        [0.0, 0.0, 1.0],
                    ]
                )
            c2ws.append(
                np.array(meta["frames"][ind]["transform_matrix"], dtype=np.float32)
            )
        
        self.Ks = torch.tensor(Ks)
        self.Ks[:, :2] /= downscale_factor
        self.c2ws = torch.from_numpy(np.array(c2ws))

        self.scale_factor = 1.0
        if auto_scale_poses:
            self.scale_factor /= float(torch.max(torch.abs(self.c2ws[:, :3, 3])))
            guru.info(f"Auto scene scale factor of {self.scale_factor=}")
        self.c2ws[:, :3, 3] *= self.scale_factor
        self.w2cs = self.c2ws.inverse()

        # load images
        imgs = torch.from_numpy(
                np.array(
                    [
                        iio.imread(
                            osp.join(self.data_dir, frame_name)
                        )
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
        guru.info(f"{self.time_ids.min()=} {self.time_ids.max()=}")
    
    def get_pcd(
        self, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pcd = o3d.io.read_point_cloud(osp.join(self.data_dir, 'sparse_pc.ply'))

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)) * self.scale_factor
        points3D_normal = torch.ones_like(points3D) / 3 ** (1/2)
        points3D_rgb = torch.from_numpy(np.asarray(pcd.colors, dtype=np.float32))

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
        }

        return data
        




if __name__ == "__main__":
    dataset = SteroBlurDataset('/home/DybluGS/dataset/basketball')

