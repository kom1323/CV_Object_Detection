import os
import shutil
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv
from torch.nn.utils.rnn import pad_sequence
import skimage
import skimage.segmentation
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import selectivesearch
from torchvision.ops import box_iou
from random import shuffle




class CombinedDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Args:
            root_dirs (list): List of directories containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.img_data_all = []
        self.gt_classes_all = []
        self.gt_boxes_all = []
        self.transform = transform

        for label, root_dir in enumerate(root_dirs):
                for foldername in ['train', 'test', 'valid']:
                    folder_path = os.path.join(root_dir, foldername)
                    if not os.path.exists(folder_path):
                        continue
                    # Open the CSV file
                    csv_file = os.path.join(folder_path, '_annotations.csv')
                    if os.path.exists(csv_file):
                        with open(csv_file, newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            filename_data = {}
                            
                            for row in reader:
                                filename = os.path.join(folder_path, row['filename'])
                                if filename not in filename_data:
                                     filename_data[filename] = {'classes': [], 'boxes': []}                                
                                xmin = int(row['xmin'])
                                ymin = int(row['ymin'])
                                xmax = int(row['xmax'])
                                ymax = int(row['ymax'])
                                filename_data[filename]['boxes'].append([xmin, ymin, xmax, ymax])
                                filename_data[filename]['classes'].append(label)
                            # Append data for each unique filename
                            for filename, data in filename_data.items():
                                self.img_data_all.append(filename)
                                self.gt_classes_all.append(data['classes'])
                                self.gt_boxes_all.append(data['boxes'])

    




    def __len__(self):
        return len(self.img_data_all)





    def __getitem__(self, idx):
        image_path = self.img_data_all[idx]
        image = Image.open(image_path)
        labels = torch.tensor(self.gt_classes_all[idx], dtype=torch.int64)
        boxes = torch.tensor(self.gt_boxes_all[idx], dtype=torch.float32)
        scale_factor_width = 224 / image.width
        scale_factor_height = 224 / image.height
        
        if self.transform:
             # Apply transformations to the image
            image = self.transform(image)
        
        for box in boxes:
            box[0] = int(box[0] * scale_factor_width)  # Adjust x1
            box[1] = int(box[1] * scale_factor_height)  # Adjust y1
            box[2] = int(box[2] * scale_factor_width)  # Adjust x2
            box[3] = int(box[3] * scale_factor_height)  # Adjust y2

        # Prepare targets
        targets = {
            'boxes': boxes,
            'labels': labels
        }




        # Return image tensor and targets
        return image, [targets]
  

    
