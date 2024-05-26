import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from torchvision import transforms
import PIL.Image as Image

class MammographyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_size=(512, 512)):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self._get_path(self.annotations.iloc[idx]['input_image']))
        mask_path = os.path.join(self.root_dir, 'masks_processed', self._get_path(self.annotations.iloc[idx]['mask_image']))
        
        image = pydicom.dcmread(img_path).pixel_array
        mask = pydicom.dcmread(mask_path).pixel_array

        # Convert images and masks to float32 and normalize image
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize image
        mask = (mask > 0).astype(np.float32)  # Binary mask

        # Resize images and masks to target size using PIL
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        image = np.array(image)
        mask = np.array(mask)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['mask']

    def _get_path(self, filename):
        """
        Get the full path to the file, ensuring to check in both benign and malignant folders.
        """
        for pathology in ['benign', 'malignant']:
            potential_path = os.path.join(pathology, filename)
            if os.path.exists(os.path.join(self.root_dir, 'images', potential_path)):
                return potential_path
            elif os.path.exists(os.path.join(self.root_dir, 'masks_processed', potential_path)):
                return potential_path
        raise FileNotFoundError(f"File {filename} not found in either benign or malignant folders.")

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # Add a channel dimension (C, H, W)
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]

        return {'image': torch.from_numpy(image), 'mask': torch.from_numpy(mask)}

def get_dataloader(csv_file, root_dir, batch_size=4, transform=None, shuffle=True):
    dataset = MammographyDataset(csv_file, root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



