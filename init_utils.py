import time
from typing import Literal

import cupy as cp
import imageio.v3 as iio
import numpy as np

# from pytorch3d.ops import sample_farthest_points
import roma
import torch, math
import torch.nn.functional as F
from cuml import HDBSCAN, KMeans
from loguru import logger as guru
from matplotlib.pyplot import get_cmap
from tqdm import tqdm
from viser import ViserServer

from loss_utils import (
    compute_accel_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    get_weights_for_procrustes,
    knn,
    masked_l1_loss,
)
from params import GaussianParams, num_sh_bases, RGB2SH
from tensor_dataclass import StaticObservations


def init_gs(
    points: StaticObservations, sh_degree: int = 3
) -> GaussianParams:
    """
    using dataclasses instead of individual tensors so we know they're consistent
    and are always masked/filtered together
    """
    num_init_gaussians = points.xyz.shape[0]
    scene_center = points.xyz.mean(0)
    points_centered = points.xyz - scene_center
    min_scale = points_centered.quantile(0.05, dim=0)
    max_scale = points_centered.quantile(0.95, dim=0)
    scene_scale = torch.max(max_scale - min_scale).item() / 2.0
    dim_sh = num_sh_bases(sh_degree)

    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    dists, _ = knn(points.xyz, 3)
    dists = torch.from_numpy(dists)
    scales = dists.mean(dim=-1, keepdim=True)
    scales = torch.log(scales.repeat(1, 3))

    means = points.xyz

    # Initialize gaussian colors
    shs = torch.zeros((points.xyz.shape[0], dim_sh, 3)).float()
    if sh_degree > 0:
        shs[:, 0, :3] = RGB2SH(points.colors)
        shs[:, 1:, 3:] = 0.0
    else:
        guru.info("use color only optimization with sigmoid activation")
        shs[:, 0, :3] = torch.logit(points.colors, eps=1e-10)
    features_dc = torch.nn.Parameter(shs[:, 0, :])
    features_rest = torch.nn.Parameter(shs[:, 1:, :])

    # Initialize gaussian orientations by normals.
    quats = random_quat_tensor(means.shape[0])
    opacities = torch.logit(torch.full((num_init_gaussians,), 0.1))
    gaussians = GaussianParams(
        means,
        quats,
        scales,
        features_dc,
        features_rest,
        opacities,
        scene_center=scene_center,
        scene_scale=scene_scale,
    )
    return gaussians

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def random_quats(N: int) -> torch.Tensor:
    u = torch.rand(N, 1)
    v = torch.rand(N, 1)
    w = torch.rand(N, 1)
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * np.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * np.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * np.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * np.pi * w),
        ],
        -1,
    )
    return quats

def interp_masked(vals: cp.ndarray, mask: cp.ndarray, pad: int = 1) -> cp.ndarray:
    """
    hacky way to interpolate batched with cupy
    by concatenating the batches and pad with dummy values
    :param vals: [B, M, *]
    :param mask: [B, M]
    """
    assert mask.ndim == 2
    assert vals.shape[:2] == mask.shape

    B, M = mask.shape

    # get the first and last valid values for each track
    sh = vals.shape[2:]
    vals = vals.reshape((B, M, -1))
    D = vals.shape[-1]
    first_val_idcs = cp.argmax(mask, axis=-1)
    last_val_idcs = M - 1 - cp.argmax(cp.flip(mask, axis=-1), axis=-1)
    bidcs = cp.arange(B)

    v0 = vals[bidcs, first_val_idcs][:, None]
    v1 = vals[bidcs, last_val_idcs][:, None]
    m0 = mask[bidcs, first_val_idcs][:, None]
    m1 = mask[bidcs, last_val_idcs][:, None]
    if pad > 1:
        v0 = cp.tile(v0, [1, pad, 1])
        v1 = cp.tile(v1, [1, pad, 1])
        m0 = cp.tile(m0, [1, pad])
        m1 = cp.tile(m1, [1, pad])

    vals_pad = cp.concatenate([v0, vals, v1], axis=1)
    mask_pad = cp.concatenate([m0, mask, m1], axis=1)

    M_pad = vals_pad.shape[1]
    vals_flat = vals_pad.reshape((B * M_pad, -1))
    mask_flat = mask_pad.reshape((B * M_pad,))
    idcs = cp.where(mask_flat)[0]

    cx = cp.arange(B * M_pad)
    out = cp.zeros((B * M_pad, D), dtype=vals_flat.dtype)
    for d in range(D):
        out[:, d] = cp.interp(cx, idcs, vals_flat[idcs, d])

    out = out.reshape((B, M_pad, *sh))[:, pad:-pad]
    return out


def batched_interp_masked(
    vals: cp.ndarray, mask: cp.ndarray, batch_num: int = 4096, batch_time: int = 64
):
    assert mask.ndim == 2
    B, M = mask.shape
    out = cp.zeros_like(vals)
    for b in tqdm(range(0, B, batch_num), leave=False):
        for m in tqdm(range(0, M, batch_time), leave=False):
            x = interp_masked(
                vals[b : b + batch_num, m : m + batch_time],
                mask[b : b + batch_num, m : m + batch_time],
            )  # (batch_num, batch_time, *)
            out[b : b + batch_num, m : m + batch_time] = x
    return out
