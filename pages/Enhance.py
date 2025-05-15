import streamlit as st
import io, time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import load_generator, DEVICE
from utils import pad_to_multiple, tensor_to_pil
from skimage.metrics import structural_similarity


# Load SRGAN model (4×)
G = load_generator("generator_weights.pth")
device = st.session_state.get("DEVICE", "cpu")
G = G.to(device)


st.subheader("🚀 Enhance: Super-Resolve Your Image(s)")
st.markdown("Upload one or more low-resolution images and obtain their 4× super-resolved outputs. Optionally, upload corresponding high-res images to compute PSNR metrics.")

# Multiple file uploaders
lr_files = st.file_uploader(
    "Select Low-Res Image(s)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="lr_files"
)
hr_files = st.file_uploader(
    "Select HR Image(s) for PSNR (optional)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="hr_files"
)

# Reset function to clear uploads
def clear_all():
    st.session_state.pop("lr_files", None)
    st.session_state.pop("hr_files", None)

st.button("🔄 Reset All", on_click=clear_all)


if "lr_files" in st.session_state and st.session_state.lr_files:
    lr_files = st.session_state["lr_files"]
    hr_files = st.session_state.get("hr_files", [])
    progress = st.progress(0)

    for idx, lr_file in enumerate(lr_files):
        st.markdown(f"### Image {idx+1}")
        # Load LR
        lr_img = Image.open(lr_file).convert("RGB")
        st.image(lr_img, caption="Low-Res Input", use_column_width=True)

        # Preprocess
        # 1) Read the device toggle from Settings
        device = st.session_state.get("DEVICE", "cpu")
        # 2) Move the model, too
        G = G.to(device)

        # 3) Preprocess and move the input tensor to the chosen device
        canvas, (w, h) = pad_to_multiple(lr_img, factor=4)
        inp = transforms.ToTensor()(canvas).unsqueeze(0).to(device)

        # Inference
        with st.spinner("Enhancing image…"):
            start = time.time()
            with torch.no_grad():
                out = G(inp).clamp(0,1)
            elapsed = time.time() - start

        # Crop to original ×4
        sr_tensor = out[0, :h*4, :w*4].cpu()
        sr_img = tensor_to_pil(sr_tensor)

        # Display
        c1, c2 = st.columns(2)
        c1.image(lr_img, caption="LR Input", use_column_width=True)
        c2.image(sr_img, caption=f"SR Output (×4) — {elapsed:.2f}s", use_column_width=True)

                # PSNR if HR provided
        if hr_files and len(hr_files) == len(lr_files):
            hr_img = Image.open(hr_files[idx]).convert("RGB")
            # Match HR to SR dimensions directly
            sr_w, sr_h = sr_img.size
            hr_resized = hr_img.resize((sr_w, sr_h), Image.BICUBIC)

            # Convert to arrays
            hr_arr = np.array(hr_resized).astype(np.float32) / 255.0
            sr_arr = np.array(sr_img).astype(np.float32) / 255.0

            # PSNR
            mse = np.mean((hr_arr - sr_arr) ** 2)
            psnr_sr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            # SSIM
            ssim_sr = structural_similarity(hr_arr, sr_arr, data_range=1.0, channel_axis=2)

            # # Bicubic baseline
            # bic_img = lr_img.resize((sr_w, sr_h), Image.BICUBIC)
            # bic_arr = np.array(bic_img).astype(np.float32) / 255.0
            # mse_bic = np.mean((hr_arr - bic_arr) ** 2)
            # psnr_bic = 10 * np.log10(1.0 / mse_bic) if mse_bic > 0 else float('inf')
            # ssim_bic = structural_similarity(hr_arr, bic_arr, data_range=1.0, channel_axis=2)

            # Display metrics
            st.metric(label="SRGAN PSNR (dB)", value=f"{psnr_sr:.2f}")
            st.metric(label="SRGAN SSIM", value=f"{ssim_sr:.4f}")
            # st.metric(label="Bicubic SSIM", value=f"{ssim_bic:.4f}")
            # st.metric(label="Bicubic PSNR", value=f"{psnr_bic:.2f}")

         # Download
        fmt     = st.session_state.get("DOWNLOAD_FORMAT", "PNG")
        quality = st.session_state.get("JPEG_QUALITY", 90)

        buf = io.BytesIO()
        if fmt == "JPEG":
            sr_img.save(buf, format="JPEG", quality=quality)
        else:
            sr_img.save(buf, format="PNG")

        st.download_button(
            label=f"📥 Download SR Image {idx+1}",
            data=buf.getvalue(),
            file_name=f"sr_{idx+1}.{fmt.lower()}",
            mime=f"image/{fmt.lower()}"
        )

        # Progress
        progress.progress((idx+1) / len(lr_files))
    # if st.button("🔄 Reset All"):
    #     st.success("App state cleared. Please refresh the page to restart.")
