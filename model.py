import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.resnet import ResNet18_Weights


# Define RoI pooling layer
class RoIPooling(nn.Module):
    def __init__(self, output_size):
        super(RoIPooling, self).__init__()
        self.roi_pool = nn.AdaptiveMaxPool2d(output_size)
        feature_map_size = 7  # Assuming the final feature map size
        original_image_size = 224  # Assuming the size of the original image
        self.scale_factor = original_image_size / feature_map_size


    def forward(self, features, rois):
        pooled_features = []
        for roi in rois:
            # Iterate over each bounding box in the batch
            print(features.shape)
            for box in roi:
                # Convert RoI from image coordinates to feature map coordinates
                x1_fm = int(box[0] / self.scale_factor)
                y1_fm = int(box[1] / self.scale_factor)
                x2_fm = int(box[2] / self.scale_factor)
                y2_fm = int(box[3] / self.scale_factor)

                
                # Perform RoI pooling for each region
                pooled_features.append(self.roi_pool(features[:, :, x1_fm:x2_fm, y1_fm:y2_fm]))
        return torch.cat(pooled_features, dim=0)


# Define classification and regression heads
class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNN, self).__init__()
        self.roi_pooling = RoIPooling(output_size=(7, 7))
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)  # 4 for bounding box coordinates

    def forward(self, features, rois):
        pooled_features = self.roi_pooling(features, rois)
        x = pooled_features.view(pooled_features.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred

# Define the Fast R-CNN model
class FastRCNNResNet(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNNResNet, self).__init__()
        resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet18 = nn.Sequential(*list(resnet18.children())[:-4])
        self.backbone = resnet18
        self.fast_rcnn = FastRCNN(num_classes + 1)  # +1 for background

    def forward(self, images, rois):
        features = self.backbone(images)
        return self.fast_rcnn(features, rois)

# Define loss function (e.g., cross-entropy loss for classification and smooth L1 loss for bounding box regression)

# Training loop

# Inference
