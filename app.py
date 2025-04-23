import os
import io
import time
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# 1) Page config & CSS
st.set_page_config(page_title="✨ SRGAN Super-Resolution", layout="wide")
st.markdown("""
    <style>
    .stButton>button {
        font-size:18px;
        padding:10px 24px;
        border-radius:10px;
        background:#4CAF50;
        color:#FFF;
    }
    .stButton>button:hover { background:#45a049; }
    </style>
""", unsafe_allow_html=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Model loader (cached)
@st.cache(allow_output_mutation=True)
def load_model(path: str):
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
            )
        def forward(self, x):
            return x + self.block(x)

    class Generator(nn.Module):
        def __init__(self, num_residual_blocks=16, upscale_factor=4):
            super().__init__()
            self.initial = nn.Sequential(
                nn.Conv2d(3, 64, 9, padding=4),
                nn.ReLU(inplace=True)
            )
            # Residual blocks
            layers = [ResidualBlock(64) for _ in range(num_residual_blocks)]
            self.res_blocks = nn.Sequential(*layers)
            # Mid-layer
            self.mid_conv = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64)
            )
            # Upsampling
            up_layers = []
            for _ in range(upscale_factor // 2):
                up_layers += [
                    nn.Conv2d(64, 256, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True)
                ]
            self.upsample = nn.Sequential(*up_layers)
            # Final conv
            self.final = nn.Conv2d(64, 3, 9, padding=4)

        def forward(self, x):
            x1 = self.initial(x)
            x2 = self.res_blocks(x1)
            x3 = self.mid_conv(x2)
            x = x1 + x3
            x = self.upsample(x)
            return self.final(x)

    model = Generator(num_residual_blocks=16, upscale_factor=4).to(DEVICE)
    sd = torch.load(path, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# **UPDATE THIS** to your actual Kaggle Models path
MODEL_PATH = "generator_final.pth"
G = load_model(MODEL_PATH)

# 3) Sidebar
st.sidebar.title("Menu")
page = st.sidebar.radio("", ["Home", "Enhance", "Settings"])

# 4) Home
if page == "Home":
    st.title("🖼️ SRGAN Super-Resolution")
    st.write("Upload a low-res image and see it magically upscaled!")

# 5) Enhance
elif page == "Enhance":
    st.subheader("Upload Your Low-Res Image")
    uploaded = st.file_uploader("", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Low-Res Input", use_column_width=True)

        # Pad to multiple of 4
        w,h = img.size
        nw, nh = ((w+3)//4)*4, ((h+3)//4)*4
        canvas = Image.new("RGB", (nw,nh))
        canvas.paste(img, (0,0))
        inp = transforms.ToTensor()(canvas).unsqueeze(0).to(DEVICE)

        with st.spinner("Enhancing image…"):
            start = time.time()
            with torch.no_grad():
                out = G(inp).clamp(0,1)
            elapsed = time.time() - start

        # Un-pad and convert
        sr = out[0,:,:h,:w].cpu()
        sr_img = transforms.ToPILImage()(sr)

        col1, col2 = st.columns(2)
        col1.image(img, caption="Original LR", use_column_width=True)
        col2.image(sr_img, caption=f"SRGAN Output (took {elapsed:.1f}s)", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        sr_img.save(buf, format="PNG")
        st.download_button("📥 Download SR Image", buf.getvalue(), file_name="srgan_output.png",
                           mime="image/png")

# 6) Settings
elif page == "Settings":
    st.subheader("⚙️ Settings")
    scale = st.slider("Upscale factor", min_value=2, max_value=8, value=4)
    st.write("Model path:", MODEL_PATH)
    st.write("Running on:", DEVICE)
