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
        _, regions = selectivesearch.selective_search(image_np, scale=500, sigma=0.8, min_size=100)

        
        # Convert regions to RoI format
        rois = []
        for r in regions: 
            
            x, y, w, h = r['rect']
            rois.append([x, y, w, h])  # [x1, y1, x2, y2]
        
        return rois
    


    def __getitem__(self, idx):
        image_path = self.img_data_all[idx]
        image = Image.open(image_path)
        labels = self.gt_classes_all[idx].clone().detach()
        boxes = self.gt_boxes_all[idx].clone().detach()
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
        

            # Convert boxes to torchvision format [x1, y1, x2, y2]
        torchvision_boxes = torch.stack([
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        ], dim=1)
        
        # Calculate IoU between each ROI and each ground truth box
        iou_matrix = box_iou(rois, torchvision_boxes)
        
        #Select the top 64 ROIs based on IoU with ground truth boxes
        top_indices = torch.argsort(iou_matrix, descending=True, dim=0)
        selected_rois = []
        selected_indices = set()
        for index in top_indices:
            for idx in index:
                if idx.item() not in selected_indices:
                    selected_indices.add(idx.item())
                    selected_rois.append(rois[idx])
                    if len(selected_rois) == 8:
                        break
            if len(selected_rois) == 8:
                break
        selected_rois = torch.stack(selected_rois)

        # # Sample 25% RoIs from object proposals with IoU >= 0.5
        # foreground_indices = torch.nonzero(iou_matrix >= 0.5)
        # foreground_indices = foreground_indices[torch.randperm(len(foreground_indices))[:int(len(foreground_indices) * 0.25)]]

        # # Sample remaining RoIs from object proposals with IoU in [0.1, 0.5)
        # background_indices = torch.nonzero((iou_matrix >= 0.1) & (iou_matrix < 0.5))
        # background_indices = background_indices[torch.randperm(len(background_indices))[:64 - len(foreground_indices)]]

        # # Combine foreground and background indices
        # selected_indices = torch.cat((foreground_indices, background_indices), dim=0)

        # # Shuffle the selected indices
        # selected_indices = selected_indices[torch.randperm(len(selected_indices))]

        # # Select corresponding RoIs based on selected indices
        # selected_rois = rois[selected_indices.squeeze()]

        # # Ensure that we have 64 RoIs
        # if len(selected_rois) < 64:
        #     # If there are fewer than 64 RoIs, randomly select additional RoIs from the top-ranked RoIs
        #     remaining_indices = torch.nonzero(~iou_matrix.any(dim=1))
        #     remaining_indices = remaining_indices[torch.randperm(len(remaining_indices))[:64 - len(selected_rois)]]
        #     remaining_rois = rois[remaining_indices.squeeze()]
        #     remaining_rois = remaining_rois.unsqueeze(0)  # Unsqueeze to match dimensions
        #     selected_rois = torch.cat((selected_rois, remaining_rois), dim=0)

        # # Ensure that we have exactly 64 RoIs
        # selected_rois = selected_rois[:64]

        return (image, selected_rois), (labels, boxes)
  

    
