import open3d as o3d
import numpy as np
pcd = o3d.io.read_point_cloud('/home/DybluGS/dataset/basketball/sparse_pc.ply')

points3D = np.asarray(pcd.points, dtype=np.float32)

points3D_rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)

print(f"{points3D} {points3D_rgb}")