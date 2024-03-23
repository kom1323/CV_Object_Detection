import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv
from torch.nn.utils.rnn import pad_sequence


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

    def __getitem__(self, idx):
        image_path = self.img_data_all[idx]
        image = Image.open(image_path)
        labels = torch.tensor(self.gt_classes_all[idx])
        boxes = torch.tensor(self.gt_boxes_all[idx])

        if self.transform:
             # Apply transformations to the image
            image = self.transform(image)
           
        return image, labels, boxes
    
