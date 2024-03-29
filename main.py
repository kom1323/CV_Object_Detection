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
from misc import eval_forward
from dataset import CombinedDataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision.utils import draw_bounding_boxes
import cv2


random_seed = 24
checkpoint_dir = 'model_checkpoint/'
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter('logs')

def collate_fn(batch):
    # Separate images and targets from the batch
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")

def get_model(num_classes):
    # Load pre-trained ResNet-18 model
    backbone = resnet_fpn_backbone('resnet18', weights=ResNet18_Weights.DEFAULT)
    backbone.out_channels = 256

    for p in backbone.parameters():
            p.requires_grad = False

    # Define anchor sizes and aspect ratios for the Region Proposal Network (RPN)
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                       aspect_ratios=(0.5, 1.0, 2.0))

    # Define the region of interest (RoI) pooling method
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                     output_size=7,
                                                     sampling_ratio=2)

    # Create Faster R-CNN model with ResNet-18 backbone
    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                      box_detections_per_img=20)

    return model



def evaluate_loss(model, data_loader, device, iteration):
    val_loss = 0
    with torch.no_grad():
      for images, targets in data_loader:
          images = move_to(images, device)
          targets = move_to(targets, device)
          losses, detections = eval_forward(model, images, targets)
          
          #only loop once and display images in tensorboard
          if val_loss == 0:
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
            break 

          val_loss += losses['loss_classifier']

    validation_loss = val_loss / len(data_loader)
    model.train()
    return validation_loss




if __name__ == "__main__":

# Define transformation to apply to the images
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    root_dirs = ['poker-cards-2', 'chip-detection-and-counting-v2-1']
    print("Creating Dataset...")
    dataset = CombinedDataset(root_dirs=root_dirs, transform=transforms_train)



    # Define the split ratio
    validation_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(validation_split * dataset_size)

    # Shuffle the indices if needed
    #np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Split the dataset into train and validation sets
    train_indices, val_indices = indices[split:], indices[:split]

    # Define samplers for train and validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SequentialSampler(val_indices)

    # Create data loaders for train and validation sets
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                            sampler=train_sampler,
                                            collate_fn=collate_fn)
    val_loader =  torch.utils.data.DataLoader(dataset, batch_size=8,
                                            sampler=val_sampler,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)
    # Training loop
for epoch in range(10):
        start_time_epoch=time.time()
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
            if i % 10 == 0:  # every x mini-batches...
              print(f"{i}/{len(train_loader)}")
              global_step = epoch * len(train_loader) + i

              # ...log the running loss
              writer.add_scalar(
                  "loss/train", running_loss / len(train_loader) , global_step
              )
              valid_loss=evaluate_loss(model, val_loader, device, i+ (epoch * len(train_loader)))
              writer.add_scalar(
                  "loss/validation", valid_loss , global_step
              )
              running_loss = 0.0



        checkpoint_path = checkpoint_dir + f'model_epoch_{epoch+1}.pth'
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)




