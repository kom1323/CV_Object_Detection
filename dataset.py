import os
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

        # pad bounding boxes and classes so they are of the same size
        self.gt_boxes_all = pad_sequence([torch.tensor(bboxes) for bboxes in self.gt_boxes_all], batch_first=True, padding_value=-1)
        self.gt_classes_all = pad_sequence([torch.tensor(classes) for classes in self.gt_classes_all], batch_first=True, padding_value=-1)




    def __len__(self):
        return len(self.img_data_all)



        # Define a function to perform selective search on an image
    def selective_search(self, image):
        # Convert image to skimage format
        image_np = image.permute(1, 2, 0).cpu().numpy() 
        # Perform selective search
        _, regions = selectivesearch.selective_search(image_np, scale=300, sigma=0.8, min_size=700)

        
        # Convert regions to RoI format
        rois = []
        for r in regions: 
            
            x, y, w, h = r['rect']
            rois.append([x, y, w, h])  # [x1, y1, x2, y2]
        
        return rois
    


    def __getitem__(self, idx):
        image_path = self.img_data_all[idx]
        image = Image.open(image_path)
        labels = torch.tensor(self.gt_classes_all[idx])
        boxes = torch.tensor(self.gt_boxes_all[idx])
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
          
        rois = torch.tensor(self.selective_search(image))

        return (image, rois), (labels, boxes) 
    

