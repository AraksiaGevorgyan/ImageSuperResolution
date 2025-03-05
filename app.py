import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import time

# Load pre-trained SRCNN from PyTorch Hub
model = torch.hub.load('https://github.com/krasserm/super-resolution', 'srcnn', pretrained=True)
model.eval()  # Set the model to evaluation mode

def preprocess_image(image):
    # Convert PIL Image to OpenCV format (BGR)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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


# Set page config
st.set_page_config(page_title="Super-Resolution App", page_icon="✨", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stButton>button {
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .uploaded-file { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar menu
st.sidebar.title("Navigation")
st.sidebar.markdown("Choose an option:")
page = st.sidebar.radio("", ["Home", "Upload Image", "Settings"])

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Super-Resolution Image Enhancer</h1>", unsafe_allow_html=True)

# Home Page
if page == "Home":
    st.write("Welcome to the Image Super-Resolution App! 🎨 Upload an image, and our AI model will enhance its resolution.")

# Upload Image Page
elif page == "Upload Image":
    st.subheader("Upload Your Low-Resolution Image")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="🖼️ Original Image", use_column_width=True)

        with col2:
            st.write("🔄 **Enhancing Image...**")
            time.sleep(2)  # Simulate processing time

            # Preprocess and perform super-resolution on the image
            low_res_image, low_res_tensor = preprocess_image(image)
            high_res_image = super_resolve(low_res_tensor)

            # Display the enhanced image
            st.image(high_res_image, caption="✨ Enhanced Image", use_column_width=True)

        st.success("Processing complete!")
        st.download_button("📥 Download Enhanced Image", high_res_image, file_name="enhanced.png")

# Settings Page (Future)
elif page == "Settings":
    st.write("⚙️ Settings will be available soon!")
