import open3d as o3d
import numpy as np
import json

def create_camera_axes(scale=1.0):
    """Create a set of axes to represent the camera pose."""
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=[0, 0, 0])
    return axes

def visualize_camera_poses(camera_poses):
    """
    Visualize camera poses using Open3D.
    Args:
        camera_poses: List of 4x4 numpy arrays representing camera extrinsics.
    """
    geometries = []
    for pose in camera_poses:
        camera_axes = create_camera_axes(scale=0.2)
        camera_axes.transform(pose)  # Apply the transformation to the axes
        geometries.append(camera_axes)
    camera_axes = create_camera_axes(scale=0.8)
    camera_axes.transform(np.eye(4))  # Apply the transformation to the axes
    geometries.append(camera_axes)

    # Visualize all the camera poses
    o3d.visualization.draw_geometries(geometries)

# 读取JSON文件
with open('/home/dyblurGS/data/nerfstudio/poster/transforms.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

poses = np.array([frame['transform_matrix'] for frame in data['frames']][:200])

visualize_camera_poses(poses)