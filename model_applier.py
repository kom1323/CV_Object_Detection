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
    
    predictions = model(frame)
    print(predictions)
    img_display = frame[0]
    img_display = (img_display * 255).to(torch.uint8)

    mapping = {0: "Background", 1: "Card", 2: "Chip"}
    replaced_list = [mapping[value.item()] if value.item() in mapping else str(value) for value in predictions[0]['labels']]
    img_display = draw_bounding_boxes(img_display, predictions[0]['boxes'], replaced_list) 

    return img_display








def activate_model(model, device):


    # Define the checkpoint path
    checkpoint_path = r'model_checkpoint\run_500x500_3\model_epoch_5.pth'

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()


    img = Image.open(r"a.jpg").convert("RGB")
    transform = transforms.Compose([   
            transforms.ToTensor(),
            transforms.Resize((500, 500)),
        ])
    img = transform(img)
    frame_tensor = move_to([img], device)

    output_frame = draw_on_frame(model, frame_tensor)

    output_frame = output_frame.cpu().numpy()
    output_frame = np.transpose(output_frame, (1, 2, 0))
    
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Tensor Image', output_frame)
    cv2.waitKey(0)  # Wait for any key to be pressed

    # video_path = "video.mp4"
    # video = cv2.VideoCapture(video_path)
    # i = 0
    # while video.isOpened():
    #     success, frame = video.read()
    #     if not success:
    #         break

    #     # Convert frame to RGB format
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     #resized_frame = cv2.resize(frame, (224, 224))
    #     frame = Image.fromarray(frame)
    #     frame = frame.convert("RGB")
    #     #frame = np.transpose(resized_frame, (2, 0, 1))

        
    

    #     # Transform frame to tensor type of PIL image
    #     transform = transforms.Compose([

            
    #         transforms.ToTensor(),
    #         transforms.Resize((500, 500)),
            
    #     ])
               
    #     frame_tensor = transform(frame)
    #     frame_tensor = move_to([frame_tensor], device)

    #     output_frame = draw_on_frame(model, frame_tensor)
    


    #     output_frame = output_frame.cpu().numpy()
    #     output_frame = np.transpose(output_frame, (1, 2, 0))
    #     output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

    #     cv2.imshow('Tensor Image', output_frame)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
   
    # video.release()
    # cv2.destroyAllWindows()
