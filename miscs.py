import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
import itertools
import numpy as np



base_sizes = [16, 32, 64]  # Base sizes of anchor boxes
aspect_ratios = [0.5, 1.0, 2.0]  # Aspect ratios of anchor boxes
num_anchors = len(base_sizes) * len(aspect_ratios)

# Define a function to display images and bounding boxes
def show_image_with_boxes(image, boxes,labels=None):
    fig, ax = plt.subplots(1)
    # Convert the tensor image to a PIL image for display
    img_pil = to_pil_image(image)
    ax.imshow(img_pil)

    # Iterate through each bounding box
    if labels is not None:
        for box, label in zip(boxes, labels):
            # Extract coordinates
            xmin, ymin, xmax, ymax = box

            # Create a Rectangle patch
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
            
            # Add label text
            ax.text(xmin, ymin, f'Class: {label}', color='r')

        plt.show()

    else:
        for box in boxes:
            # Extract coordinates
            xmin, ymin, xmax, ymax = box

            # Create a Rectangle patch
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()


        num_boxes = len(gt_boxes)
        num_anchors = len(anchor_boxes)
        labels = torch.zeros(num_anchors, dtype=torch.long)
        bbox_targets = torch.zeros((num_anchors, 4), dtype=torch.float32)

        if num_boxes == 0:
            return labels, bbox_targets

        ious = torch.zeros((num_anchors, num_boxes))
        for i in range(num_anchors):
            for j in range(num_boxes):
                ious[i, j] = calculate_iou(anchor_boxes[i], gt_boxes[j])

        max_ious, max_inds = ious.max(dim=1)
        gt_max_ious, gt_max_inds = ious.max(dim=0)

        # Assign positive labels to anchor boxes with IoU greater than pos_iou_threshold
        labels[max_ious >= pos_iou_threshold] = 1
        # Assign negative labels to anchor boxes with IoU less than neg_iou_threshold
        labels[max_ious < neg_iou_threshold] = 0

        # Match each ground truth box to the anchor box with the highest IoU
        for j in range(num_boxes):
            i = gt_max_inds[j]
            bbox_targets[i] = calculate_bbox_target(anchor_boxes[i], gt_boxes[j])

        return labels, bbox_targets