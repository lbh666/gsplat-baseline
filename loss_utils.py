import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable
from math import exp


def masked_mse_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_mse_loss(pred, gt, quantile)
    else:
        sum_loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_huber_loss(pred, gt, delta, mask=None, normalize=True):
    if mask is None:
        return F.huber_loss(pred, gt, delta=delta)
    else:
        sum_loss = F.huber_loss(pred, gt, delta=delta, reduction="none")
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def trimmed_mse_loss(pred, gt, quantile=0.9):
    loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def compute_gradient_loss(pred, gt, mask, quantile=0.98):
    """
    Compute gradient loss
    pred: (batch_size, H, W, D) or (batch_size, H, W)
    gt: (batch_size, H, W, D) or (batch_size, H, W)
    mask: (batch_size, H, W), bool or float
    """
    # NOTE: messy need to be cleaned up
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
    pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
    gt_grad_x = gt[:, :, 1:] - gt[:, :, :-1]
    gt_grad_y = gt[:, 1:, :] - gt[:, :-1, :]
    loss = masked_l1_loss(
        pred_grad_x[mask_x][..., None], gt_grad_x[mask_x][..., None], quantile=quantile
    ) + masked_l1_loss(
        pred_grad_y[mask_y][..., None], gt_grad_y[mask_y][..., None], quantile=quantile
    )
    return loss


def knn(x: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = x.cpu().numpy()
    knn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x)
    distances, indices = knn_model.kneighbors(x)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def get_weights_for_procrustes(clusters, visibilities=None):
    clusters_median = clusters.median(dim=-2, keepdim=True)[0]
    dists2clusters_center = torch.norm(clusters - clusters_median, dim=-1)
    dists2clusters_center /= dists2clusters_center.median(dim=-1, keepdim=True)[0]
    weights = torch.exp(-dists2clusters_center)
    weights /= weights.mean(dim=-1, keepdim=True) + 1e-6
    if visibilities is not None:
        weights *= visibilities.float() + 1e-6
    invalid = dists2clusters_center > np.quantile(
        dists2clusters_center.cpu().numpy(), 0.9
    )
    invalid |= torch.isnan(weights)
    weights[invalid] = 0
    return weights


def compute_z_acc_loss(means_ts_nb: torch.Tensor, w2cs: torch.Tensor):
    """
    :param means_ts (G, 3, B, 3)
    :param w2cs (B, 4, 4)
    return (float)
    """
    camera_center_t = torch.linalg.inv(w2cs)[:, :3, 3]  # (B, 3)
    ray_dir = F.normalize(
        means_ts_nb[:, 1] - camera_center_t, p=2.0, dim=-1
    )  # [G, B, 3]
    # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, B, 3]
    # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    acc_loss = (
        ((means_ts_nb[:, 1] - means_ts_nb[:, 0]) * ray_dir).sum(dim=-1) ** 2
    ).mean() + (
        ((means_ts_nb[:, 2] - means_ts_nb[:, 1]) * ray_dir).sum(dim=-1) ** 2
    ).mean()
    return acc_loss


def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (K, T, 3)
    :param motion_rots (K, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    accel = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    loss = accel.norm(dim=-1).mean()
    return loss


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
