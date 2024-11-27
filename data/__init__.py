from dataclasses import asdict, replace

from torch.utils.data import Dataset

from .base_dataset import BaseDataset, CustomDataConfig
from .stero_blur_dataset import SteroBlurDataset, SteroBlurDataConfig


def get_train_val_datasets(
    data_cfg: CustomDataConfig, load_val: bool
) -> BaseDataset:
    assert isinstance(data_cfg, SteroBlurDataConfig), "Only support SteroBlurDataset yet"
    train_dataset = SteroBlurDataset(**asdict(data_cfg))
    return train_dataset