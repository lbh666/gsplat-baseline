import glob, subprocess, os, shutil
import os.path as osp

scenes = os.listdir('/home/DybluGS/stereo_blur_dataset')


for scene in scenes:
    cmd = ["ns-process-data", "images", "--data",
        f"/home/DybluGS/stereo_blur_dataset/{scene}/dense",
        "--output-dir", f"dataset/{scene}", "--skip-image-processing",
        "--skip-colmap", "--colmap-model-path", f"/home/DybluGS/stereo_blur_dataset/{scene}/dense/sparse"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    for folder_name in os.listdir(f"/home/DybluGS/stereo_blur_dataset/{scene}/dense"):
        if osp.isdir(f"/home/DybluGS/stereo_blur_dataset/{scene}/dense/{folder_name}") and 'images' in folder_name:
            shutil.copytree(f"/home/DybluGS/stereo_blur_dataset/{scene}/dense/{folder_name}", f"dataset/{scene}/{folder_name}", dirs_exist_ok=True)


    if result.stderr:
        print(result.stderr)
        