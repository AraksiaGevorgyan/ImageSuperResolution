# pages/1_Home.py
import streamlit as st

# Academic-style landing page for SRGAN application
st.title("SRGAN Image Super-Resolution Application")

st.markdown("""
**Abstract:** This web application implements a state-of-the-art Super-Resolution Generative Adversarial Network (SRGAN) to enhance the spatial resolution of low-quality images by a factor of four. The model reconstructs high-frequency textures and fine details, achieving perceptually convincing results that exceed traditional interpolation methods.

---

## Overview
1. **Input:** Provide a low-resolution (LR) photograph.  
2. **Inference:** The SRGAN generator upscales the LR image by 4× using residual blocks and sub-pixel convolutions.  
3. **Output:** Retrieve a high-resolution (HR) image with enhanced sharpness and texture.

The underlying network was trained on the DIV2K dataset with **16 residual blocks** and incorporates a perceptual loss derived from a pre-trained VGG19 network. This ensures fidelity to the original image content while promoting realistic detail reconstruction.

---

## Key Features
- **4× Upscaling**: Single-pass super-resolution to quadruple image dimensions.  
- **Residual Learning**: 16 ResBlocks for deep feature extraction.  
- **Perceptual Loss**: VGG-based feature matching for texture preservation.  
- **Fast Inference**: Approximately 0.3 seconds per image on GPU.

---

## Usage Example
Below is a representative example of a low-resolution input and its corresponding SRGAN output:
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
col1.image("assets/test_lr.png", caption="Low-Resolution Input", use_container_width=True)
col2.image("assets/test_sr.png", caption="SRGAN-Enhanced Output", use_container_width=True)

st.markdown("""
---

**References:**  
Ledig, Christian, et al. *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*. CVPR 2017.
""", unsafe_allow_html=True)
