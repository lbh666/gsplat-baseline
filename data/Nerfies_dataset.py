import torch
from torch.utils.data import Dataset, default_collate
from typing import Literal
import imageio.v3 as iio
from utils.utils import load_from_json, get_viewmat
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
class NerfiesDataConfig:
    data_dir: str
    downscale_factor: int = 1
    auto_scale_poses: bool = True
    split: Literal["train", "val"] = "train"

