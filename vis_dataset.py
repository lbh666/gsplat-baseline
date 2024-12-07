import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated

import numpy as np
import torch
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2 as cv
from configs import LossesConfig, OptimizerConfig, SceneLRConfig
from data import (
    BaseDataset,
    get_train_val_datasets,
    SteroBlurDataConfig,
    SteroBlurDataset
)
from utils.utils import to_device
from utils.init_utils import init_gs

from scene.scene_model import SceneModel
from utils.tensor_dataclass import StaticObservations
from trainer import Trainer
from vis.utils import get_server
from vis.viewer import DynamicViewer, debug_render_fn, VISER_NERFSTUDIO_SCALE_RATIO

torch.set_float32_matmul_precision("high")


cfg = SteroBlurDataConfig('/home/dyblurGS/data/nerfstudio/poster')
dataset = SteroBlurDataset(**asdict(cfg))

server = get_server(port=1000)
viewer = DynamicViewer(
    server, debug_render_fn, 100, 'tmp', mode="training"
)
viewer.init_scene(dataset, 'training')
