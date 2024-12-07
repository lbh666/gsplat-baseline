from dataclasses import asdict, replace

from torch.utils.data import Dataset

from .base_dataset import BaseDataset, CustomDataConfig
from .stero_blur_dataset import SteroBlurDataset, SteroBlurDataConfig
from .colmap_dataset import ColmapDataset, ColmapDataConfig


def get_train_val_datasets(
    data_cfg: CustomDataConfig, load_val: bool
) -> BaseDataset:
    if isinstance(data_cfg, SteroBlurDataConfig):
        train_dataset = SteroBlurDataset(**asdict(data_cfg))
    elif isinstance(data_cfg, ColmapDataConfig):
        train_dataset = ColmapDataset(**asdict(data_cfg))
    else:
        raise NotImplementedError("Not supported dataset type")
    return train_dataset