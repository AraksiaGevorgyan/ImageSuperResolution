import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# Load pre-trained SRCNN from PyTorch Hub
model = torch.hub.load('https://github.com/krasserm/super-resolution', 'srcnn', pretrained=True)
model.eval()  # Set the model to evaluation mode

def preprocess_image(image):
    # Convert the PIL image to a numpy array
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Resize the image to a lower resolution (for testing)
    low_res = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    # Normalize and convert the image to a tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    low_res_tensor = transform(low_res).unsqueeze(0)  # Add batch dimension
    return low_res, low_res_tensor

def super_resolve(low_res_tensor):
    with torch.no_grad():
        output_tensor = model(low_res_tensor)

    # Convert tensor back to image
    output_image = output_tensor.squeeze(0).cpu().numpy()  # Remove batch dimension
    output_image = np.transpose(output_image, (1, 2, 0))  # Convert from CHW to HWC
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)  # Convert to 8-bit

    return output_image
