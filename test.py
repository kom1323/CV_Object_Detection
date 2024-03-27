import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from test_dataset import CombinedDataset
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet18_Weights


def get_model(num_classes):
    # Load pre-trained ResNet-18 model
    backbone = resnet_fpn_backbone('resnet18', weights=ResNet18_Weights.DEFAULT)

    # Define anchor sizes and aspect ratios for the Region Proposal Network (RPN)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

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


    # dataloaders|
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=True)




    # Example usage:
    # Define the number of classes (including background)
    num_classes = 3  # For example, if you have 1 class + background
    model = get_model(num_classes)

    # Print the model architecture
    #print(model)

# Define optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    print("Starting Epoch #", epoch)
    model.train()
    running_loss = 0.0
    for images, targets in dataloader:
        images = [image.float() for image in images]  # Convert images to float
        targets = [{k: v.float() for k, v in target.items()} for target in targets]  # Convert targets to float
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")


