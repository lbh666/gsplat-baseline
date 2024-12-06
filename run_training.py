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
)
from utils.utils import to_device
from utils.init_utils import init_gs

from scene.scene_model import SceneModel
from utils.tensor_dataclass import StaticObservations
from trainer import Trainer

torch.set_float32_matmul_precision("high")


def set_seed(seed):
    # Set the seed for generating random numbers
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    work_dir: str
    data: (Annotated[SteroBlurDataConfig, tyro.conf.subcommand(name="stero")]
           | Annotated[SteroBlurDataConfig, tyro.conf.subcommand(name="stero")]
           | Annotated[SteroBlurDataConfig, tyro.conf.subcommand(name="stero")]
    )
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_fg: int = 40_000
    num_bg: int = 100_000
    iterations: int = 30_000
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 1
    num_dl_workers: int = 8
    validate_every: int = 50
    save_videos_every: int = 50


def main(cfg: TrainConfig):
    set_seed(42)
    backup_code(cfg.work_dir)
    train_dataset = (
        get_train_val_datasets(cfg.data, load_val=True)
    )
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save config
    os.makedirs(cfg.work_dir, exist_ok=True)
    with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    # if checkpoint exists
    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    initialize_and_checkpoint_model(
        cfg,
        train_dataset,
        device,
        ckpt_path,
        vis=cfg.vis_debug,
        port=cfg.port,
    )

    trainer, start_epoch = Trainer.init_from_checkpoint(
        ckpt_path,
        device,
        cfg.lr,
        cfg.loss,
        cfg.optim,
        work_dir=cfg.work_dir,
        port=cfg.port,
    )

    trainer.viewer.init_scene(train_dataset, "training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dl_workers,
        persistent_workers=True,
        collate_fn=BaseDataset.train_collate_fn,
        pin_memory=True,
        shuffle=True
    )

    val_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dl_workers,
        persistent_workers=True,
        collate_fn=BaseDataset.train_collate_fn,
        pin_memory=True
    )

    guru.info(f"Starting training from {trainer.global_step=}")
    load_data = iter(train_loader)
    for iters in (
        pbar := tqdm(
            range(trainer.global_step, cfg.iterations),
            initial=trainer.global_step,
            total=cfg.iterations,
        )
    ):
        try:
            batch = next(load_data)
        except:
            load_data = iter(train_loader)
            batch = next(load_data)
            trainer.set_epoch(iters // len(train_loader))
        batch = to_device(batch, device)
        loss = trainer.train_step(batch)
        pbar.set_description(f"Loss: {loss:.6f}")
    psnr_ = []
    for idx, data in enumerate(val_loader):
        data = to_device(data, device)
        loss, stats, _, _, rendered = trainer.compute_losses(data)
        psnr_.append(stats["train/psnr"].item())
        cv.imwrite(f'outputs/{idx:03d}.png', (rendered.squeeze().detach().cpu().numpy()[..., ::-1]*255).astype(np.uint8))
    print("Avg PSNR", np.array(psnr_).mean())




def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_dataset: BaseDataset,
    device: torch.device,
    ckpt_path: str,
    vis: bool = False,
    port: int | None = None,
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return
    
    points = StaticObservations(*train_dataset.get_pcd())
    gs_params= init_gs(points)
    # run initial optimization
    Ks = train_dataset.get_Ks().to(device)
    w2cs = train_dataset.get_w2cs().to(device)
    model = SceneModel(Ks, w2cs, gs_params)

    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)


def backup_code(work_dir):
    root_dir = osp.abspath(osp.join(osp.dirname(__file__)))
    tracked_dirs = [osp.join(root_dir, dirname) for dirname in ["flow3d", "scripts"]]
    dst_dir = osp.join(work_dir, "code", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    for tracked_dir in tracked_dirs:
        if osp.exists(tracked_dir):
            shutil.copytree(tracked_dir, osp.join(dst_dir, osp.basename(tracked_dir)))


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))