import torch
import torch.nn as nn
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import numpy as np
from torchvision.models.detection import FasterRCNN
from misc import move_to, eval_forward




def draw_on_frame(model, frame):
    print(frame.shape)
    predictions = model([frame])
    img_display = frame
    img_display = (img_display * 255).to(torch.uint8)
    mapping = {0: "Background", 1: "Card", 2: "Chip"}
    replaced_list = [mapping[value.item()] if value.item() in mapping else str(value) for value in predictions[0]['labels']]
    img_display = draw_bounding_boxes(img_display, predictions[0]['boxes'], replaced_list) 
    return img_display








def activate_model(model, device):


    # Define the checkpoint path
    checkpoint_dir = 'model_checkpoint/'
    checkpoint_path = checkpoint_dir + f'model_epoch_9.pth'

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    video_path = "video.mp4"
    video = cv2.VideoCapture(video_path)
    while True:
        success, frame = video.read()
        if not success:
            break

        # Convert frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #resized_frame = cv2.resize(frame, (224, 224))


        #frame = np.transpose(resized_frame, (2, 0, 1))

        
    

        # Transform frame to tensor type of PIL image
        transform = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            
        ])
       
        
        #frame_tensor = torch.tensor(resized_frame).permute(2, 0, 1)
        
        frame_tensor = transform(frame)
        frame_tensor = move_to(frame_tensor, device)


        output_frame = draw_on_frame(model, frame_tensor)


        # Display frame using OpenCV
        cv2.imshow('Frame', output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
