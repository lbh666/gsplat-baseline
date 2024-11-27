from vis.utils import get_server
from data.stero_blur_dataset import SteroBlurDataset
import cv2 as cv
from vis.viewer import DynamicViewer, debug_render_fn


dataset = SteroBlurDataset('/home/dyblurGS/data/nerfstudio/poster', downscale_factor=3)
server = get_server(port=1000)
viewer = DynamicViewer(
    server, debug_render_fn, 100, './', mode="training"
)
viewer.init_scene(dataset, "training")
while True:
    x = 1