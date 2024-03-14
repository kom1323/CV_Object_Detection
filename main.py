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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
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
miscs.show_images(trainset, num_samples=5)





if __name__ == "__main__":
    pass

    