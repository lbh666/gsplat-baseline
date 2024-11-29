import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState

from scene_model import SceneModel
from vis.utils import draw_tracks_2d_th, get_server
from vis.viewer import DynamicViewer, VISER_NERFSTUDIO_SCALE_RATIO


class Renderer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        # Logging.
        work_dir: str,
        port: int | None = None,
    ):
        self.device = device

        self.model = model

        self.work_dir = work_dir
        self.global_step = 0
        self.epoch = 0

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, 100, work_dir, mode="rendering"
            )

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> "Renderer":
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        renderer = Renderer(model, device, *args, **kwargs)
        renderer.global_step = ckpt.get("global_step", 0)
        renderer.epoch = ckpt.get("epoch", 0)
        return renderer

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh
        # rescale scene 
        camera_state.c2w[:3, 3] /= VISER_NERFSTUDIO_SCALE_RATIO

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