# The following code will only execute
# successfully when compression is complete

import kagglehub

# Download latest version
# path = kagglehub.dataset_download("ductrongle/kitti-bin")
# path = kagglehub.dataset_download("ductrongle/nyu-sync-and-official-splits")
path = kagglehub.dataset_download("ductrongle/mde-opengl-dataset")

print("Path to dataset files:", path)

# /root/.cache/kagglehub/datasets/