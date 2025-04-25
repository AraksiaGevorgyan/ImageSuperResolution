# pages/3_Settings.py
import streamlit as st
import torch

st.set_page_config(page_title="Settings", layout="wide")
st.subheader("⚙️ Settings")

# 1) Download format selector
st.markdown("**Download format**")
download_fmt = st.radio("Choose output file type", ["PNG", "JPEG"], index=0)
jpeg_quality = None
if download_fmt == "JPEG":
    jpeg_quality = st.slider("JPEG quality", min_value=10, max_value=100, value=90)
    st.write(f"JPEG quality set to **{jpeg_quality}**")

st.markdown("---")

# 2) Inference device toggle
st.markdown("**Inference device**")
use_gpu = st.checkbox("Enable GPU acceleration", value=torch.cuda.is_available())
device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
st.write(f"Running on: **{device}**")

# Persist into session state for other pages to read
st.session_state["DOWNLOAD_FORMAT"] = download_fmt
if jpeg_quality is not None:
    st.session_state["JPEG_QUALITY"] = jpeg_quality
st.session_state["DEVICE"] = device
