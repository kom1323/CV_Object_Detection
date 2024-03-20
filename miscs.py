import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image

# Define a function to display a sample of images
def show_images(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i in range(num_samples):
        # Get a random index
        idx = torch.randint(0, len(dataset), (1,))
        image, label = dataset[idx]

        # Display the image
        ax = axes[i] if num_samples > 1 else axes
        ax.imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax.set_title(f"Label: {label.item()}")
        ax.axis('off')

    plt.show()


# Define a function to display images and bounding boxes
def show_image_with_boxes(image, labels, boxes):
    fig, ax = plt.subplots(1)
    # Convert the tensor image to a PIL image for display
    img_pil = to_pil_image(image)
    ax.imshow(img_pil)

    # Iterate through each bounding box
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

