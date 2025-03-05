import streamlit as st
from PIL import Image
import time
import io
from model import preprocess_image, super_resolve

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

            # Save the enhanced image to a buffer so it can be downloaded
            img_pil = Image.fromarray(high_res_image)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            buf.seek(0)

        st.success("Processing complete!")
        st.download_button("📥 Download Enhanced Image", buf, file_name="enhanced.png", mime="image/png")

# Settings Page (Future)
elif page == "Settings":
    st.write("⚙️ Settings will be available soon!")