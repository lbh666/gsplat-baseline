import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat.rendering import rasterization
from torch import Tensor

from params import GaussianParams


class SceneModel(nn.Module):
    def __init__(
        self,
        Ks: Tensor,
        w2cs: Tensor,
        gs_params: GaussianParams,
    ):
        super().__init__()
        self.gs_params = gs_params
        scene_scale = 1.0
        self.register_buffer("bg_scene_scale", torch.as_tensor(scene_scale))
        self.register_buffer("Ks", Ks)
        self.register_buffer("w2cs", w2cs)

        self._current_xys = None
        self._current_radii = None
        self._current_img_wh = None

    @property
    def num_gaussians(self) -> int:
        return self.gs_params.num_gaussians


    def get_colors_all(self) -> torch.Tensor:
        return self.gs_params.get_colors()

    def get_scales_all(self) -> torch.Tensor:
        return self.gs_params.get_scales()

    def get_opacities_all(self) -> torch.Tensor:
        """
        :returns colors: (G, 3), scales: (G, 3), opacities: (G, 1)
        """
        return self.gs_params.get_opacities()

    @staticmethod
    def init_from_state_dict(state_dict, prefix=""):
        gs_params = None
        gs_params = GaussianParams.init_from_state_dict(
            state_dict, prefix=f"{prefix}gs_params.params."
        )
        Ks = state_dict[f"{prefix}Ks"]
        w2cs = state_dict[f"{prefix}w2cs"]
        return SceneModel(Ks, w2cs, gs_params)

    def render(
        self,
        w2cs: torch.Tensor,  # (C, 4, 4)
        Ks: torch.Tensor,  # (C, 3, 3)
        img_wh: tuple[int, int],
        bg_color: torch.Tensor | float = 1.0,
        colors_override: torch.Tensor | None = None,
        means: torch.Tensor | None = None,
        quats: torch.Tensor | None = None,
        return_depth: bool = False,
        return_mask: bool = False,
    ) -> dict:
        device = w2cs.device
        C = w2cs.shape[0]

        W, H = img_wh

        if means is None or quats is None:
            means, quats = self.gs_params.params['means'], self.gs_params.get_quats()

        if colors_override is None:
            colors_override = self.gs_params.get_colors()

        D = colors_override.shape[-1]

        scales = self.gs_params.get_scales()
        opacities = self.gs_params.get_opacities()

        if isinstance(bg_color, float):
            bg_color = torch.full((C, D), bg_color, device=device)
        assert isinstance(bg_color, torch.Tensor)

        mode = "RGB"
        ds_expected = {"img": D}

        if return_mask:
            mask_values = torch.ones((self.num_gaussians, 1), device=device)
            colors_override = torch.cat([colors_override, mask_values], dim=-1)
            bg_color = torch.cat([bg_color, torch.zeros(C, 1, device=device)], dim=-1)
            ds_expected["mask"] = 1

        B = 0

        assert colors_override.shape[-1] == sum(ds_expected.values())
        assert bg_color.shape[-1] == sum(ds_expected.values())

        if return_depth:
            mode = "RGB+ED"
            ds_expected["depth"] = 1

        render_colors, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_override,
            backgrounds=bg_color,
            viewmats=w2cs,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=W,
            height=H,
            packed=False,
            render_mode=mode,
        )

        # Populate the current data for adaptive gaussian control.
        if self.training and info["means2d"].requires_grad:
            self._current_xys = info["means2d"]
            self._current_radii = info["radii"]
            self._current_img_wh = img_wh
            # We want to be able to access to xys' gradients later in a
            # torch.no_grad context.
            self._current_xys.retain_grad()

        assert render_colors.shape[-1] == sum(ds_expected.values())
        outputs = torch.split(render_colors, list(ds_expected.values()), dim=-1)
        out_dict = {}
        for i, (name, dim) in enumerate(ds_expected.items()):
            x = outputs[i]
            assert x.shape[-1] == dim, f"{x.shape[-1]=} != {dim=}"
            out_dict[name] = x
        out_dict["acc"] = alphas
        return out_dict
