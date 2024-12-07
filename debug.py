from data.colmap_dataset import ColmapDataset


dataset = ColmapDataset('/home/dyblurGS/data/nerfstudio/poster')
pcd = dataset.get_pcd()