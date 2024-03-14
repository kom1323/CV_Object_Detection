import matplotlib.pyplot as plt
import torch

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


