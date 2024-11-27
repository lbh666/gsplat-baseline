from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from jaxtyping import Float32, UInt8
from nerfview import CameraState, Viewer
from viser import Icon, ViserServer
import viser
from typing import TYPE_CHECKING, Dict, List, Literal, Optional
from vis.playback_panel import add_gui_playback_group
from vis.render_panel import populate_render_tab
from data.stero_blur_dataset import SteroBlurDataset
import torch
import viser.transforms as vtf

VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0


def debug_render_fn(
    camera_state: CameraState, img_wh: Tuple[int, int]
) -> np.ndarray:
    # Parse camera state for camera-to-world matrix (c2w) and intrinsic (K) as
    # float64 numpy arrays.
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    # Do your things and get an image as a uint8 numpy array.
    img = np.ones((128, 128, 3)) * 255
    return img.astype(np.uint8)

class DynamicViewer(Viewer):
    def __init__(
        self,
        server: ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        num_frames: int,
        work_dir: str,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        self.num_frames = num_frames
        self.work_dir = Path(work_dir)
        super().__init__(server, render_fn, mode)

    def _define_guis(self):
        super()._define_guis()
        server = self.server
        self._time_folder = server.gui.add_folder("Time")
        with self._time_folder:
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.num_frames,
                initial_fps=15.0,
            )
            self._playback_guis[0].on_update(self.rerender)
            self._canonical_checkbox = server.gui.add_checkbox("Canonical", False)
            self._canonical_checkbox.on_update(self.rerender)

            _cached_playback_disabled = []

            def _toggle_gui_playing(event):
                if event.target.value:
                    nonlocal _cached_playback_disabled
                    _cached_playback_disabled = [
                        gui.disabled for gui in self._playback_guis
                    ]
                    target_disabled = [True] * len(self._playback_guis)
                else:
                    target_disabled = _cached_playback_disabled
                for gui, disabled in zip(self._playback_guis, target_disabled):
                    gui.disabled = disabled

            self._canonical_checkbox.on_update(_toggle_gui_playing)

        self._render_track_checkbox = server.gui.add_checkbox("Render tracks", False)
        self._render_track_checkbox.on_update(self.rerender)

        tabs = server.gui.add_tab_group()
        with tabs.add_tab("Render", Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                server, Path(self.work_dir) / "camera_paths", self._playback_guis[0]
            )

        # Add buttons to toggle training image visibility
        self.hide_images = self.server.gui.add_button(
            label="Hide Train Cams", disabled=False, icon=viser.Icon.EYE_OFF, color=None
        )
        self.hide_images.on_click(lambda _: self.set_camera_visibility(False))
        self.hide_images.on_click(lambda _: self.toggle_cameravis_button())
        self.show_images = self.server.gui.add_button(
            label="Show Train Cams", disabled=False, icon=viser.Icon.EYE, color=None
        )
        self.show_images.on_click(lambda _: self.set_camera_visibility(True))
        self.show_images.on_click(lambda _: self.toggle_cameravis_button())
        self.show_images.visible = False

    def set_camera_visibility(self, visible: bool) -> None:
        """Toggle the visibility of the training cameras."""
        with self.server.atomic():
            for idx in self.camera_handles:
                self.camera_handles[idx].visible = visible
                self.point_cloud_handle.visible = visible

    def toggle_cameravis_button(self) -> None:
        self.hide_images.visible = not self.hide_images.visible
        self.show_images.visible = not self.show_images.visible


    def init_scene(
        self,
        train_dataset: SteroBlurDataset,
        train_state: Literal["training", "paused", "completed"],
    ) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            train_state: Current status of training
        """
        # draw the training cameras and images
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}
        fx, fy, cx, cy = train_dataset.fx, train_dataset.fy, train_dataset.cx, train_dataset.cy
        for idx in range(len(train_dataset)):
            data = train_dataset[idx]
            image = data["imgs"]
            c2w = data["c2ws"].cpu().numpy()
            image_uint8 = (image * 255).detach().type(torch.uint8)
            image_uint8 = image_uint8.permute(2, 0, 1)

            # torchvision can be slow to import, so we do it lazily.
            import torchvision

            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)  # type: ignore
            image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handle = self.server.scene.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.1,
                aspect=float(cx / cy),
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
            )

            def create_on_click_callback(capture_idx):
                def on_click_callback(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                    with event.client.atomic():
                        event.client.camera.position = event.target.position
                        event.client.camera.wxyz = event.target.wxyz

                return on_click_callback

            camera_handle.on_click(create_on_click_callback(idx))

            self.camera_handles[idx] = camera_handle
            self.original_c2w[idx] = c2w

        self.train_state = train_state
        self.train_util = 0.9

        pcd, _, pcd_color = train_dataset.get_pcd()

        self.point_cloud_handle = self.server.scene.add_point_cloud(
            name="/colmap/pcd",
            points=pcd.numpy() * VISER_NERFSTUDIO_SCALE_RATIO,
            colors=pcd_color.numpy(),
            point_size=0.005,
        )
    
