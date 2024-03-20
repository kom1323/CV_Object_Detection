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
# from torch.utils.tensorboard import SummaryWriter

# # default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('logs')


# transforms
# Create transform function
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),                     # data augmentation
    transforms.ToTensor(),
   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    #transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


root_dirs = ['poker-cards-2', 'chip-detection-and-counting-v2-1']
trainset = dataset.CombinedDataset(root_dirs=root_dirs, transform=transforms_train)


# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)


# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                         shuffle=False, num_workers=2)

# constant for classes
classes = ('Card','Chip')

print(f" is cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Display a sample of 5 images from the combined_dataset
#miscs.show_images(trainset, num_samples=5)

img,labels,boxes = trainset[1]
h, w = img.shape[1:3]

print(f"{img=}")
print(f"{labels=}")
print(f"{boxes=}")


# print(h, w)
# X = torch.rand(size=(1, 3, h, w))  # Construct input data
# Y = anchor_box.multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# print(Y.shape)


# boxes = Y.reshape(h, w, 5, 4)
# print(boxes[150, 150, 0, :])


# fig, ax = plt.subplots()
# # Assuming w, h are the width and height of the image respectively
# bbox_scale = torch.tensor((w, h, w, h))
# anchor_box.show_bboxes(ax, boxes[150, 150, :, :] * bbox_scale,
#             ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
#              's=0.75, r=0.5'])


miscs.show_image_with_boxes(img,labels,boxes)


if __name__ == "__main__":
    pass

    