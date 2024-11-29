import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianParams(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        scene_center: torch.Tensor | None = None,
        scene_scale: torch.Tensor | float = 1.0,
    ):
        super().__init__()
        if not check_gaussian_sizes(
            means, quats, scales, colors, opacities
        ):
            import ipdb

            ipdb.set_trace()
        params_dict = {
            "means": nn.Parameter(means),
            "quats": nn.Parameter(quats),
            "scales": nn.Parameter(scales),
            "colors": nn.Parameter(colors),
            "opacities": nn.Parameter(opacities),
        }
        self.params = nn.ParameterDict(params_dict)
        self.quat_activation = lambda x: F.normalize(x, dim=-1, p=2)
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid

        if scene_center is None:
            scene_center = torch.zeros(3, device=means.device)
        self.register_buffer("scene_center", scene_center)
        self.register_buffer("scene_scale", torch.as_tensor(scene_scale))

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        req_keys = ["means", "quats", "scales", "colors", "opacities"]
        assert all(f"{prefix}{k}" in state_dict for k in req_keys)
        args = {
            "scene_center": torch.zeros(3),
            "scene_scale": torch.tensor(1.0),
        }
        for k in req_keys + list(args.keys()):
            if f"{prefix}{k}" in state_dict:
                args[k] = state_dict[f"{prefix}{k}"]
        return GaussianParams(**args)

    @property
    def num_gaussians(self) -> int:
        return self.params["means"].shape[0]

    def get_colors(self) -> torch.Tensor:
        return self.color_activation(self.params["colors"])

    def get_scales(self) -> torch.Tensor:
        return self.scale_activation(self.params["scales"])

    def get_opacities(self) -> torch.Tensor:
        return self.opacity_activation(self.params["opacities"])

    def get_quats(self) -> torch.Tensor:
        return self.quat_activation(self.params["quats"])


    def densify_params(self, should_split, should_dup):
        """
        densify gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_dup = x[should_dup]
            x_split = x[should_split].repeat([2] + [1] * (x.ndim - 1))
            if name == "scales":
                x_split -= math.log(1.6)
            x_new = nn.Parameter(torch.cat([x[~should_split], x_dup, x_split], dim=0))
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def cull_params(self, should_cull):
        """
        cull gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_new = nn.Parameter(x[~should_cull])
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def reset_opacities(self, new_val):
        """
        reset all opacities to new_val
        """
        self.params["opacities"].data.fill_(new_val)
        updated_params = {"opacities": self.params["opacities"]}
        return updated_params


def check_gaussian_sizes(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
) -> bool:
    dims = means.shape[:-1]
    leading_dims_match = (
        quats.shape[:-1] == dims
        and scales.shape[:-1] == dims
        and colors.shape[:-1] == dims
        and opacities.shape == dims
    )
    dims_correct = (
        means.shape[-1] == 3
        and (quats.shape[-1] == 4)
        and (scales.shape[-1] == 3)
        and (colors.shape[-1] == 3)
    )
    return leading_dims_match and dims_correct