import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import io

# Define the SRCNN model architecture (this should match the architecture of the saved model)
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()

        # Non-linear mapping
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()

        # Reconstruction
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load the saved model with caching to avoid reloading it on every interaction
@st.cache_resource
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
    
    # Move tensor to the same device as model (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model.to(device)
    
    with torch.no_grad():  # No need to calculate gradients during inference
        output_tensor = model(input_tensor)
    
    # Convert tensor to image
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Remove batch dimension and change order
    output_image = (output_image * 255).clip(0, 255).astype(np.uint8)  # Convert to valid image range

    return Image.fromarray(output_image)

# Function to convert image to bytes for downloading
def pil_image_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# Streamlit app
def main():
    st.title("Image Super-Resolution")
    
    # Load model once at the beginning (caching ensures it's only loaded once)
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
        processed_image_bytes = pil_image_to_bytes(processed_image)
        st.download_button("Download Processed Image", data=processed_image_bytes, file_name="super_resolved_image.png")

if __name__ == "__main__":
    main()
