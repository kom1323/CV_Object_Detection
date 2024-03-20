import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard's SummaryWriter
from PIL import Image

# Check if CUDA is available
print(f"Is CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load pre-trained ResNet18 model
model = resnet18(pretrained=True)
model.to(device)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter()

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# File path of the image to test
image_path = "pic.jpg"

# Load and preprocess the image
image_tensor = load_and_preprocess_image(image_path)

# Function to make predictions on the image
def predict_image(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Make prediction
prediction = predict_image(image_tensor)
print(prediction)

# Log the prediction to TensorBoard
writer.add_scalar("Predictions", prediction)  # Log prediction to TensorBoard

# Close the SummaryWriter
writer.close()
