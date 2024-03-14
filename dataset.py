import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CombinedDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Args:
            root_dirs (list): List of directories containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = []
        self.labels = []
        self.transform = transform
        
        for label, root_dir in enumerate(root_dirs):
            for foldername in ['train', 'test', 'valid']:
                folder_path = os.path.join(root_dir, foldername)
                if not os.path.exists(folder_path):
                    continue
                for subdir, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('.jpg'):
                            image_path = os.path.join(subdir, file)
                            self.data.append(image_path)
                            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        label = torch.tensor(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label







