import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dataset
import miscs
import anchor_box

# transforms
# Create transform function
transforms_train = transforms.Compose([
    #transforms.Resize((224, 224)),
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),                     # data augmentation
    transforms.ToTensor(),
   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])



root_dirs = ['poker-cards-2', 'chip-detection-and-counting-v2-1']
trainset = dataset.CombinedDataset(root_dirs=root_dirs, transform=transforms_train)


# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)

# constant for classes
classes = ('Card','Chip')

print(f" is cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


img,labels,boxes = trainset[0]
print(f"{img=}")
print(f"{labels=}")
print(f"{boxes=}")
miscs.show_image_with_boxes(img,labels,boxes)
 