import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from test_dataset import CombinedDataset
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet18_Weights
import sys
import os


checkpoint_dir = 'model_checkpoint/'
os.makedirs(checkpoint_dir, exist_ok=True)




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
    print(backbone)

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
                       box_roi_pool=roi_pooler)

    return model



if __name__ == "__main__":

# Define transformation to apply to the images
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    root_dirs = ['poker-cards-2', 'poker-chips']
    
    print("Creating Dataset...")
    trainset = CombinedDataset(root_dirs=root_dirs, transform=transforms_train)


    # dataloader-s|
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                            shuffle=True,
                                            collate_fn=collate_fn)
    num_batches = len(dataloader)
    # yes = next(iter(dataloader))
    # print("heeyyy", len(yes[1]))
    # sys.exit()

    
    # Example usage:
    # Define the number of classes (including background)
    num_classes = 3  # For example, if you have 1 class + background
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
        print("Starting Epoch #", epoch)
        model.train()
        running_loss = 0.0
        for i, (images, targets) in enumerate(dataloader):
            print(f"Processing batch {i+1} out of {num_classes}")
            images = move_to(images, device)
            targets = move_to(targets, device)
                
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
            checkpoint_path = checkpoint_dir + f'model_epoch_{epoch+1}.pth'

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")


