import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import time
import numpy as np
from misc import eval_forward, get_model, collate_fn, move_to
from dataset import CombinedDataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import random_split


import cv2
import model_applier
import sys


checkpoint_dir = 'model_checkpoint/'
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter('logs')
TESTING = 0

def write_images_to_tensorboard(model, data_loader, device, iteration):
    images, targets = next(iter(data_loader))   
    images = move_to(images, device)
    targets = move_to(targets, device)
    losses, detections = eval_forward(model, images, targets)
        
        #only loop once and display images in tensorboard
    processed_images = []
    for i in range(4):
      img_display = images[i]
      img_display = (img_display * 255).to(torch.uint8)
      output = detections[i]
      print(output['labels'])
      mapping = {0: "Background", 1: "Card", 2: "Chip"}
      replaced_list = [mapping[value.item()] if value.item() in mapping else str(value) for value in output['labels']]
      img_display = draw_bounding_boxes(img_display, output['boxes'], replaced_list)  
      processed_images.append(img_display)          
    processed_images = move_to(processed_images, device)
    img_grid = torchvision.utils.make_grid(processed_images)  # Creating grid of images
    writer.add_image(f'images', img_grid, global_step=iteration)  
    model.train()
     

def evaluate_valid_loss(model, data_loader, device):
    val_loss = 0
    with torch.no_grad():
      for images, targets in data_loader:
          images = move_to(images, device)
          targets = move_to(targets, device)
          losses, detections = eval_forward(model, images, targets)
          val_loss += losses['loss_classifier']

    validation_loss = val_loss / len(data_loader)
    model.train()
    return validation_loss




if __name__ == "__main__":
    
    if TESTING:
      num_classes = 3  
      model = get_model(num_classes)

      print(f" is cuda available: {torch.cuda.is_available()}")
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print(device)
      model.to(device)
      model_applier.activate_model(model, device)
      sys.exit()
       
       

#   Define transformation to apply to the images
    transforms_train = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor()
    ])
      
      # Albumentations

    root_dirs = ['poker-cards-2', 'chip-detection-and-counting-v2-1']
    print("Creating Dataset...")
    dataset = CombinedDataset(root_dirs=root_dirs, transform=transforms_train)

    validation_split = 0.2
    dataset_size = len(dataset)
    # Define the sizes of train and validation sets
    train_size = int((1 - validation_split) * dataset_size)
    val_size = dataset_size - train_size
    torch.manual_seed(1)

    # Split the dataset randomly
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for train and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                            collate_fn=collate_fn)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=8,
                                            collate_fn=collate_fn)
  



    # Define the number of classes (including background)
    num_classes = 3  
    model = get_model(num_classes)

    print(f" is cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Print the model architecture
    #print(model)

    # Define optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    model.to(device)
    # Training loop
    for epoch in range(15):
            print("Starting Epoch #", epoch+1)
            model.train()
            running_loss = 0.0
            for i, (images, targets) in enumerate(train_loader):
                images = move_to(images, device)
                targets = move_to(targets, device)

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                if i % 50 == 0:  # every x mini-batches...
                  print(f"{i}/{len(train_loader)}")
                  global_step = epoch * len(train_loader) + i

                  # ...log the running loss
                  writer.add_scalar(
                      "loss/train", running_loss / len(train_loader) , global_step
                  )
                  write_images_to_tensorboard(model, val_loader, device, i+ (epoch * len(train_loader)))
                
                  running_loss = 0.0
            checkpoint_path = checkpoint_dir + f'model_epoch_{epoch+1}.pth'
            torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
            


    valid_loss = evaluate_valid_loss(model,val_loader,device)
    print(valid_loss)
                # writer.add_scalar(
                #     "loss/validation", valid_loss , global_step
                # )




