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
import model as m

# Create transform function
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def main():

    root_dirs = ['poker-cards-2', 'chip-detection-and-counting-v2-1']
    trainset = dataset.CombinedDataset(root_dirs=root_dirs, transform=transforms_train)


    # dataloaders
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                            shuffle=True, num_workers=2)

    # constant for classes
    classes = ('Card','Chip')

    print(f" is cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Define the model
    num_classes = 2  # Specify the number of classes in your dataset
    model = m.FastRCNNResNet(num_classes)
    model.to(device)

    # Define loss function and optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # Training loop
    num_epochs = 10  # Specify the number of epochs
    for epoch in range(num_epochs):
        print("Starting Epoch: ", epoch)
        
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for images, targets in dataloader:  # Iterate over batches of data

            images = images.to(device)
            targets = [target.to(device) for target in targets]

            # Forward pass
            print("Forward pass")
            cls_scores, bbox_preds = model(images)
            
            # Compute the loss
            print("Compute Loss")
            loss_cls = criterion_cls(cls_scores, targets[0])
            loss_bbox = criterion_bbox(bbox_preds, targets[1])
            loss = loss_cls + loss_bbox

            # Backward pass and optimization
            print("Backprop")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print statistics
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    print('Finished Training')



    def print_example():
        (img,rois),(labels,boxes) = trainset[1]


        miscs.show_image_with_boxes(img,labels=labels,boxes=boxes)
        miscs.show_image_with_boxes(img,boxes=rois)
    

if __name__ == "__main__":
    main()






