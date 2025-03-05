import streamlit as st
from PIL import Image

st.title("Super-Resolution Image Enhancer")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image")
