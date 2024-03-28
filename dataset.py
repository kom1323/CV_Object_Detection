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
        _, regions = selectivesearch.selective_search(image_np, scale=600, sigma=0.8, min_size=50)

        
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

        # selected_rois = []
        # selected_labels = []

        # num_foreground_rois = 0
        # num_background_rois = 0

        # # Select foreground ROIs with IoU >= 0.5
        # for i in range(len(rois)):
        #     max_iou = torch.max(iou_matrix[i])
        #     if max_iou >= 0.5:
        #         selected_rois.append(rois[i])
        #         selected_labels.append(i)  # Save index of the foreground ROI
        #         num_foreground_rois += 1

        # # Select background ROIs with IoU in [0.1, 0.5)
        # for i in range(len(rois)):
        #     max_iou = torch.max(iou_matrix[i])
        #     if max_iou >= 0.1 and max_iou < 0.5 and num_background_rois < (len(rois) - num_foreground_rois):
        #         num_background_rois += 1

        # # Shuffle selected indices of foreground ROIs
        # shuffle(selected_rois)

        # # Convert selected ROIs to tensor
        # selected_rois = torch.stack(selected_rois)

        # # Convert selected indices to tensor
        # foreground_rois_indices = torch.tensor(selected_labels)


        return (image, selected_rois), (labels, boxes)
  

    
