import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

def is_valid_image_path(file_path, valid_extensions=None):
    if valid_extensions is None:
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    return os.path.splitext(file_path)[1].lower() in valid_extensions

class DepthRGBDataset(Dataset):
    def __init__(self, data_dir_root):
        """
        Args:
            data_dir_root (str): Path to the dataset
        """
        self.data_dir_root = data_dir_root
        self.transform = transforms.ToTensor()  # Convert to tensor

        self.depth_dir = os.path.join(data_dir_root, "depth")
        self.rgb_dir = os.path.join(data_dir_root, "rgb")
        self.seg_dir = os.path.join(data_dir_root, "semantic_segmentation")

        self.min_depth = 1e-5
        self.max_depth = 10
        # Collect image paths
        self.image_pairs = []
        for scene in os.listdir(self.depth_dir):
            depth_scene_path = os.path.join(self.depth_dir, scene)
            rgb_scene_path = os.path.join(self.rgb_dir, scene)
            seg_scene_path = os.path.join(self.seg_dir, scene)

            if os.path.isdir(depth_scene_path) and os.path.isdir(rgb_scene_path):
                for img_name in os.listdir(depth_scene_path):
                    depth_img_path = os.path.join(depth_scene_path, img_name)
                    rgb_img_path = os.path.join(rgb_scene_path, img_name.replace("depth", "rgb"))
                    seg_img_path = os.path.join(seg_scene_path, img_name.replace("depth", "seg"))
                    if not(is_valid_image_path(depth_img_path) and is_valid_image_path(rgb_img_path)):
                        continue
                    if os.path.exists(rgb_img_path):  # Ensure corresponding RGB image exists
                        if os.path.exists(seg_img_path):
                            self.image_pairs.append((rgb_img_path, depth_img_path, seg_img_path))
                        else:    
                            self.image_pairs.append((rgb_img_path, depth_img_path, None))
                            

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path, seg_path = self.image_pairs[idx]

        rgb_image = Image.open(rgb_path)
        # depth_image = Image.open(depth_path).convert("L")
        depth_image = Image.open(depth_path)
        if seg_path is not None:
            seg_image = Image.open(seg_path)
        
        rgb_image = self.transform(rgb_image)
        depth_image = np.asarray(depth_image, dtype=np.float32)
        depth_image = depth_image / 1000.0
        depth_image = torch.from_numpy(depth_image).unsqueeze(0)
        if seg_path is not None:
            seg_image = np.asarray(seg_image, dtype=np.int64)
            seg_image = torch.from_numpy(seg_image).unsqueeze(0)
        else:
            seg_image = None
        gt = depth_image.clone()
        mask = torch.logical_and(depth_image > self.min_depth,
                              depth_image < self.max_depth)
        # depth_image[mask] = 1./(depth_image[mask] + 1e-6)
        # depth_image[mask] = (depth_image[mask] - depth_image[mask].min()) / (depth_image[mask].max() - depth_image[mask].min()) # normalize to [0, 1] 

        return dict(image=rgb_image, 
                    depth=depth_image, 
                    segmentation=seg_image, 
                    gt=gt,
                    mask=mask, 
                    image_path=rgb_path, 
                    depth_path=depth_path, 
                    seg_path=seg_path, 
                    dataset='our-data')

def get_our_data_loader(data_dir_root, batch_size=1, distributed=False, num_workers=1, mode = "train", **kwargs):
    if distributed:
        if mode == "train":
            training_samples = DepthRGBDataset(data_dir_root)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                training_samples)
    
            dataloader = DataLoader(training_samples,
               batch_size=batch_size,
               shuffle=(train_sampler is None),
               num_workers=num_workers,
               pin_memory=True,
               persistent_workers=True,
            #    prefetch_factor=2,
               sampler=train_sampler)
        elif mode == "online_eval" or mode == "test":
            testing_samples = DepthRGBDataset(data_dir_root)
            dataloader = DataLoader(testing_samples, 1,
                                    shuffle=kwargs.get("shuffle_test", False),
                                    num_workers=1,
                                    pin_memory=False,
                                    sampler=None)
    else:
        dataloader = DataLoader(DepthRGBDataset(data_dir_root), batch_size=batch_size, **kwargs)
        # dataloader = DataLoader(DepthRGBDataset(data_dir_root), batch_size=batch_size, num_workers=0, persistent_workers=False, pin_memory=False, **kwargs)
    return dataloader

if __name__ == '__main__':
    from torchvision.utils import save_image
    # Initialize dataset and dataloader
    # dataset = DepthRGBDataset(data_dir_root="dataset/our-data/v1.2/test")
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    data_dir_root = "/mrsach_cao_chanh_tri_102/workspace/DepthEstimation/dataset/our-data/v4.1/test"
    dataloader = get_our_data_loader(batch_size=16, data_dir_root=data_dir_root, shuffle=True)
    
    # Test the dataloader
    for i, batch in enumerate(dataloader):
        batch = dataloader.dataset[119]
        print(batch['image_path'])
        print(f'RGB Batch Shape: {batch["image"].shape}, Depth Batch Shape: {batch["depth"].shape}, Mask Batch Shape: {batch["mask"].shape}')
        save_image(batch["image"], "rgb.jpg")
        save_image(batch["mask"].float(), "mask.jpg")
        save_image(batch["depth"], "depth.jpg")
        break

    
