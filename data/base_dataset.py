from abc import abstractmethod

import torch
from torch.utils.data import Dataset, default_collate
from dataclasses import dataclass
from typing import Literal, cast


class BaseDataset(Dataset):
    @property
    @abstractmethod
    def num_frames(self) -> int: ...

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return torch.arange(self.num_frames)

    @abstractmethod
    def get_w2cs(self) -> torch.Tensor: ...

    @abstractmethod
    def get_Ks(self) -> torch.Tensor: ...

    @abstractmethod
    def get_image(self, index: int) -> torch.Tensor: ...

    def get_img_wh(self) -> tuple[int, int]: ...

    @staticmethod
    def train_collate_fn(batch):
        collated = {}
        for k in batch[0]:
            if k not in [
                "query_tracks_2d",
                "target_ts",
                "target_w2cs",
                "target_Ks",
                "target_tracks_2d",
                "target_visibles",
                "target_track_depths",
                "target_invisibles",
                "target_confidences",
            ]:
                collated[k] = default_collate([sample[k] for sample in batch])
            else:
                collated[k] = [sample[k] for sample in batch]
        return collated
    
    @abstractmethod
    def get_pcd(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:...

@dataclass
class CustomDataConfig:
    seq_name: str
    root_dir: str
    start: int = 0
    end: int = -1
    res: str = ""
    image_type: str = "images"
    mask_type: str = "masks"
    depth_type: Literal[
        "aligned_depth_anything",
        "aligned_depth_anything_v2",
        "depth_anything",
        "depth_anything_v2",
        "unidepth_disp",
    ] = "aligned_depth_anything"
    camera_type: Literal["droid_recon"] = "droid_recon"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 7
    num_targets_per_frame: int = 4
    load_from_cache: bool = False