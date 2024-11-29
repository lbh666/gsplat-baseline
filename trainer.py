import functools
import time
from dataclasses import asdict
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState
from pytorch_msssim import SSIM
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from configs import LossesConfig, OptimizerConfig, SceneLRConfig
from loss_utils import (
    compute_gradient_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    masked_l1_loss,
    ssim, l1_loss
)
from metrics import PCK, mLPIPS, mPSNR, mSSIM
from scene_model import SceneModel
from vis.utils import get_server
from vis.viewer import DynamicViewer, debug_render_fn, VISER_NERFSTUDIO_SCALE_RATIO
import viser.transforms as vtf
from utils import get_viewmat

def save_img(rendered, gt):
    import cv2 as cv
    rendered = (rendered.detach().cpu().numpy()[..., ::-1]*255).astype(np.uint8)
    gt = (gt.detach().cpu().numpy()[..., ::-1]*255).astype(np.uint8)
    for idx, (i1, i2) in enumerate(zip(rendered, gt)):
        cv.imwrite(f'render_{idx}.png', i1)
        cv.imwrite(f'gt_{idx}.png', i2)


class Trainer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        lr_cfg: SceneLRConfig,
        losses_cfg: LossesConfig,
        optim_cfg: OptimizerConfig,
        # Logging.
        work_dir: str,
        port: int | None = None,
        log_every: int = 10,
        checkpoint_every: int = 2000,
        validate_every: int = 500,
        validate_video_every: int = 1000,
        validate_viewer_assets_every: int = 100,
    ):
        self.device = device
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every
        self.validate_video_every = validate_video_every
        self.validate_viewer_assets_every = validate_viewer_assets_every

        self.model = model

        self.lr_cfg = lr_cfg
        self.losses_cfg = losses_cfg
        self.optim_cfg = optim_cfg

        self.reset_opacity_every = (
            self.optim_cfg.reset_opacity_every_n_controls * self.optim_cfg.control_every
        )
        self.optimizers, self.scheduler = self.configure_optimizers()

        # running stats for adaptive density control
        self.running_stats = {
            "xys_grad_norm_acc": torch.zeros(self.model.num_gaussians, device=device),
            "vis_count": torch.zeros(
                self.model.num_gaussians, device=device, dtype=torch.int64
            ),
            "max_radii": torch.zeros(self.model.num_gaussians, device=device),
        }

        self.work_dir = work_dir
        self.writer = SummaryWriter(log_dir=work_dir)
        self.global_step = 0
        self.epoch = 0

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, 100, work_dir, mode="training"
            )


        # metrics
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS()
        self.pck_metric = PCK()

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def save_checkpoint(self, path: str):
        model_dict = self.model.state_dict()
        optimizer_dict = {k: v.state_dict() for k, v in self.optimizers.items()}
        scheduler_dict = {k: v.state_dict() for k, v in self.scheduler.items()}
        ckpt = {
            "model": model_dict,
            "optimizers": optimizer_dict,
            "schedulers": scheduler_dict,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        torch.save(ckpt, path)
        guru.info(f"Saved checkpoint at {self.global_step=} to {path}")

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> tuple["Trainer", int]:
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        trainer = Trainer(model, device, *args, **kwargs)
        if "optimizers" in ckpt:
            trainer.load_checkpoint_optimizers(ckpt["optimizers"])
        if "schedulers" in ckpt:
            trainer.load_checkpoint_schedulers(ckpt["schedulers"])
        trainer.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        trainer.set_epoch(start_epoch)
        return trainer, start_epoch

    def load_checkpoint_optimizers(self, opt_ckpt):
        for k, v in self.optimizers.items():
            v.load_state_dict(opt_ckpt[k])

    def load_checkpoint_schedulers(self, sched_ckpt):
        for k, v in self.scheduler.items():
            v.load_state_dict(sched_ckpt[k])

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        W, H = img_wh
        # rescale scene 
        camera_state.c2w[:3, 3] /= VISER_NERFSTUDIO_SCALE_RATIO

        # # convert COLMAP coordinate to nerfstudio coordinate
        # R = vtf.SO3.from_matrix(camera_state.c2w[:3, :3])
        # R = R @ vtf.SO3.from_x_radians(np.pi)
        # R = torch.tensor(R.as_matrix())
        # camera_state.c2w[:3, :3] = R

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
            )
        self.model.training = False
        img = self.model.render(w2c[None], K[None], img_wh)["img"][0]
        return (img.cpu().numpy() * 255.0).astype(np.uint8)

    def train_step(self, batch):
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.1)
            self.viewer.lock.acquire()

        loss, stats, num_rays_per_step, num_rays_per_sec, _ = self.compute_losses(batch)
        if loss.isnan():
            guru.info(f"Loss is NaN at step {self.global_step}!!")
            import ipdb

            ipdb.set_trace()
        loss.backward()

        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in self.scheduler.values():
            sched.step()

        self.log_dict(stats)
        self.global_step += 1
        self.run_control_steps()

        if self.viewer is not None:
            self.viewer.lock.release()
            self.viewer.state.num_train_rays_per_sec = num_rays_per_sec
            if self.viewer.mode == "training":
                self.viewer.update(self.global_step, num_rays_per_step)

        if self.global_step % self.checkpoint_every == 0:
            self.save_checkpoint(f"{self.work_dir}/checkpoints/last.ckpt")

        return loss.item()

    def compute_losses(self, batch):
        self.model.training = True
        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        # (B, 4, 4).
        c2ws = batch["c2ws"]
        viewmats = get_viewmat(c2ws)
        # (B, 3, 3).
        Ks = batch["Ks"]
        # (B, H, W, 3).
        imgs = batch["imgs"]
        # (B, H, W).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))

        _tic = time.time()

        loss = 0.0

        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []
        for i in range(B):
            rendered = self.model.render(
                viewmats[None, i],
                Ks[None, i],
                img_wh
            )
            rendered_all.append(rendered)
            if (
                self.model._current_xys is not None
                and self.model._current_radii is not None
                and self.model._current_img_wh is not None
            ):
                self._batched_xys.append(self.model._current_xys)
                self._batched_radii.append(self.model._current_radii)
                self._batched_img_wh.append(self.model._current_img_wh)

        # Necessary to make viewer work.
        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        # (B, H, W, N, *).
        rendered_all = {
            key: (
                torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }

        # Compute losses.
        # RGB loss.
        rendered_imgs = cast(torch.Tensor, rendered_all["img"])
        # save_img(rendered_imgs, imgs)
        rgb_loss = 0.8 * l1_loss(rendered_imgs, imgs) + 0.2 * (
            1 - ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2))
        )
        loss += rgb_loss * self.losses_cfg.w_rgb



        # Prepare stats for logging.
        stats = {
            "train/loss": loss.item(),
            "train/rgb_loss": rgb_loss.item(),
            "train/num_gaussians": self.model.num_gaussians,
        }

        # Compute metrics.
        with torch.no_grad():
            psnr = self.psnr_metric(
                rendered_imgs, imgs, valid_masks
            )
            self.psnr_metric.reset()
            stats["train/psnr"] = psnr

        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )

        return loss, stats, num_rays_per_step, num_rays_per_sec, rendered_imgs

    def log_dict(self, stats: dict):
        for k, v in stats.items():
            self.writer.add_scalar(k, v, self.global_step)

    def run_control_steps(self):
        global_step = self.global_step
        # Adaptive gaussian control.
        cfg = self.optim_cfg
        ready = self._prepare_control_step()
        if (
            ready
            and global_step > cfg.warmup_steps
            and global_step % cfg.control_every == 0
            and global_step < cfg.stop_control_steps
        ):
            if (
                global_step < cfg.stop_densify_steps
            ):
                self._densify_control_step(global_step)
            if global_step % self.reset_opacity_every ==1000:
                self._cull_control_step(global_step)
            if global_step % self.reset_opacity_every == 0:
                self._reset_opacity_control_step()

            # Reset stats after every control.
            for k in self.running_stats:
                self.running_stats[k].zero_()

    @torch.no_grad()
    def _prepare_control_step(self) -> bool:
        # Prepare for adaptive gaussian control based on the current stats.
        if not (
            self.model._current_radii is not None
            and self.model._current_xys is not None
        ):
            guru.warning("Model not training, skipping control step preparation")
            return False

        batch_size = len(self._batched_xys)
        # these quantities are for each rendered view and have shapes (C, G, *)
        # must be aggregated over all views
        for _current_xys, _current_radii, _current_img_wh in zip(
            self._batched_xys, self._batched_radii, self._batched_img_wh
        ):
            sel = _current_radii > 0
            gidcs = torch.where(sel)[1]
            # normalize grads to [-1, 1] screen space
            xys_grad = _current_xys.grad.clone()
            xys_grad[..., 0] *= _current_img_wh[0] / 2.0 * batch_size
            xys_grad[..., 1] *= _current_img_wh[1] / 2.0 * batch_size
            self.running_stats["xys_grad_norm_acc"].index_add_(
                0, gidcs, xys_grad[sel].norm(dim=-1)
            )
            self.running_stats["vis_count"].index_add_(
                0, gidcs, torch.ones_like(gidcs, dtype=torch.int64)
            )
            max_radii = torch.maximum(
                self.running_stats["max_radii"].index_select(0, gidcs),
                _current_radii[sel] / max(_current_img_wh),
            )
            self.running_stats["max_radii"].index_put((gidcs,), max_radii)
        return True

    @torch.no_grad()
    def _densify_control_step(self, global_step):
        assert (self.running_stats["vis_count"] > 0).any()

        cfg = self.optim_cfg
        xys_grad_avg = self.running_stats["xys_grad_norm_acc"] / self.running_stats[
            "vis_count"
        ].clamp_min(1)
        is_grad_too_high = xys_grad_avg > cfg.densify_xys_grad_threshold
        # Split gaussians.
        scales = self.model.get_scales_all()
        is_scale_too_big = scales.amax(dim=-1) > cfg.densify_scale_threshold
        if global_step < cfg.stop_control_by_screen_steps:
            is_radius_too_big = (
                self.running_stats["max_radii"] > cfg.densify_screen_threshold
            )
        else:
            is_radius_too_big = torch.zeros_like(is_grad_too_high, dtype=torch.bool)

        should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
        should_dup = is_grad_too_high & ~is_scale_too_big

        param_map = self.model.gs_params.densify_params(should_split, should_dup)
        num_splits = int(should_split.sum().item())
        num_dups = int(should_dup.sum().item())
        for param_name, new_params in param_map.items():
            full_param_name = f"gs_params.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            dup_in_optim(
                optimizer,
                [new_params],
                should_split,
                num_splits * 2 + num_dups,
            )


        # update running stats
        for k, v in self.running_stats.items():
            new_v = torch.cat(
                [
                    v[~should_split],
                    v[should_dup],
                    v[should_split].repeat(2),
                ],
                dim=0,
            )
            self.running_stats[k] = new_v
        guru.info(
            f"Split {should_split.sum().item()} gaussians, "
            f"Duplicated {should_dup.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _cull_control_step(self, global_step):
        # Cull gaussians.
        cfg = self.optim_cfg
        opacities = self.model.get_opacities_all()
        device = opacities.device
        is_opacity_too_small = opacities < cfg.cull_opacity_threshold
        is_radius_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        is_scale_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        cull_scale_threshold = (
            torch.ones(len(is_scale_too_big), device=device) * cfg.cull_scale_threshold
        )
        if global_step > self.reset_opacity_every:
            scales = self.model.get_scales_all()
            is_scale_too_big = scales.amax(dim=-1) > cull_scale_threshold
            if global_step < cfg.stop_control_by_screen_steps:
                is_radius_too_big = (
                    self.running_stats["max_radii"] > cfg.cull_screen_threshold
                )
        should_cull = is_opacity_too_small | is_radius_too_big | is_scale_too_big

        param_map = self.model.gs_params.cull_params(should_cull)
        for param_name, new_params in param_map.items():
            full_param_name = f"gs_params.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_cull)


        # update running stats
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[~should_cull]

        guru.info(
            f"Culled {should_cull.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _reset_opacity_control_step(self):
        # Reset gaussian opacities.
        new_val = torch.logit(torch.tensor(0.8 * self.optim_cfg.cull_opacity_threshold))
        for part in ["gs_params"]:
            part_params = getattr(self.model, part).reset_opacities(new_val)
            # Modify optimizer states by new assignment.
            for param_name, new_params in part_params.items():
                full_param_name = f"{part}.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                reset_in_optim(optimizer, [new_params])
        guru.info("Reset opacities")

    def configure_optimizers(self):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}
        # named parameters will be [part].params.[field]
        # e.g. fg.params.means
        # lr config is a nested dict for each fg/bg part
        for name, params in self.model.named_parameters():
            part, _, field = name.split(".")
            lr = lr_dict[part][field]
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            if "scales" in name:
                fnc = functools.partial(_exponential_decay, lr_final=0.1 * lr)
            else:
                fnc = lambda _, **__: 1.0

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )
        return optimizers, schedulers


def dup_in_optim(optimizer, new_params: list, should_dup: torch.Tensor, num_dups: int):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            p = param_state[key]
            param_state[key] = torch.cat(
                [p[~should_dup], p.new_zeros(num_dups, *p.shape[1:])],
                dim=0,
            )
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def remove_from_optim(optimizer, new_params: list, _should_cull: torch.Tensor):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            param_state[key] = param_state[key][~_should_cull]
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def reset_in_optim(optimizer, new_params: list):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            param_state[key] = torch.zeros_like(param_state[key])
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()