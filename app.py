import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# Define the SRCNN model architecture (this should match the architecture of the saved model)
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load the saved model
def load_model(model_path):
    model = SRCNN()  # Initialize the model (same architecture as when you saved it)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to process the input image and perform super-resolution
def process_image(image, model):
    # Preprocess image: Convert to tensor and normalize
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # No need to calculate gradients during inference
        output_tensor = model(input_tensor)
    
    # Convert tensor to image
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Remove batch dimension and change order
    output_image = (output_image * 255).clip(0, 255).astype(np.uint8)  # Convert to valid image range

    return Image.fromarray(output_image)

# Streamlit app
def main():
    st.title("Image Super-Resolution")
    
    # Load model once at the beginning
    model = load_model("srcnn_model.pth")  # Replace with the correct model path
    
    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Open the image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image with model
        processed_image = process_image(image, model)
        
        # Show processed image
        st.image(processed_image, caption="Super-Resolution Image", use_column_width=True)
        
        # Option to download the processed image
        processed_image_bytes = processed_image.tobytes()
        st.download_button("Download Processed Image", data=processed_image_bytes, file_name="super_resolved_image.png")
