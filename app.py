# import os
# import io
# import time
# import torch
# import streamlit as st
# from PIL import Image
# from torchvision import transforms


# # 1) Page config & CSS
# st.set_page_config(page_title="✨ SRGAN Super-Resolution", layout="wide")
# st.markdown("""
#     <style>
#     .stButton>button {
#         font-size:18px;
#         padding:10px 24px;
#         border-radius:10px;
#         background:#4CAF50;
#         color:#FFF;
#     }
#     .stButton>button:hover { background:#45a049; }
#     </style>
# """, unsafe_allow_html=True)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # 2) Model loader (cached_resource)
# @st.cache_resource
# def load_model(path: str):
#     import torch.nn as nn

#     class ResidualBlock(nn.Module):
#         def __init__(self, channels):
#             super().__init__()
#             self.block = nn.Sequential(
#                 nn.Conv2d(channels, channels, 3, padding=1),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channels, channels, 3, padding=1),
#                 nn.BatchNorm2d(channels),
#             )
#         def forward(self, x):
#             return x + self.block(x)

#     class Generator(nn.Module):
#         def __init__(self, num_residual_blocks=16, upscale_factor=4):
#             super().__init__()
#             self.initial = nn.Sequential(
#                 nn.Conv2d(3, 64, 9, padding=4),
#                 nn.ReLU(inplace=True)
#             )
#             layers = [ResidualBlock(64) for _ in range(num_residual_blocks)]
#             self.res_blocks = nn.Sequential(*layers)
#             self.mid_conv = nn.Sequential(
#                 nn.Conv2d(64, 64, 3, padding=1),
#                 nn.BatchNorm2d(64)
#             )
#             up_layers = []
#             for _ in range(upscale_factor // 2):
#                 up_layers += [
#                     nn.Conv2d(64, 256, 3, padding=1),
#                     nn.PixelShuffle(2),
#                     nn.ReLU(inplace=True)
#                 ]
#             self.upsample = nn.Sequential(*up_layers)
#             self.final = nn.Conv2d(64, 3, 9, padding=4)

#         def forward(self, x):
#             x1 = self.initial(x)
#             x2 = self.res_blocks(x1)
#             x3 = self.mid_conv(x2)
#             x = x1 + x3
#             x = self.upsample(x)
#             return self.final(x)

#     model = Generator(num_residual_blocks=16, upscale_factor=4).to(DEVICE)
#     sd = torch.load(path, map_location=DEVICE)
#     model.load_state_dict(sd, strict=False)
#     model.eval()
#     return model

# # ***** UPDATE THIS ********
# MODEL_PATH = "generator_best.pth"
# G = load_model(MODEL_PATH)

# # 3) Sidebar
# st.sidebar.title("Menu")
# page = st.sidebar.radio("", ["Home", "Enhance", "Settings"])

# # 4) Home
# if page == "Home":
#     st.title("🖼️ SRGAN Super-Resolution")
#     st.write("Upload a low-res image and see it magically upscaled 4×!")

# # 5) Enhance
# elif page == "Enhance":
#     st.subheader("Upload Your Low-Res Image")
#     uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"])
#     if uploaded:
#         img = Image.open(uploaded).convert("RGB")
#         st.image(img, caption="Low-Res Input", use_container_width=True)

#         # Ask user for a scale factor (but only ×4 is supported)
#         scale = st.slider("Select Upscale Factor", 2, 8, 4)
#         if scale != 4:
#             st.warning("Currently only a 4× model is supported under the hood. Using 4× regardless.")
#             scale = 4

#         # Pad to multiple of scale
#         w, h = img.size
#         nw, nh = ((w + (scale-1)) // scale) * scale, ((h + (scale-1)) // scale) * scale
#         canvas = Image.new("RGB", (nw, nh))
#         canvas.paste(img, (0, 0))
#         inp = transforms.ToTensor()(canvas).unsqueeze(0).to(DEVICE)

#         with st.spinner("Enhancing image…"):
#             start = time.time()
#             with torch.no_grad():
#                 out = G(inp).clamp(0, 1)
#             elapsed = time.time() - start

#         # Slice out exactly scale× original size
#         sr_tensor = out[0, : h*scale, : w*scale].cpu()
#         sr_img = transforms.ToPILImage()(sr_tensor)

#         col1, col2 = st.columns(2)
#         col1.image(img, caption="Original LR", use_container_width=True)
#         col2.image(sr_img, caption=f"SRGAN Output (took {elapsed:.1f}s)", use_container_width=True)

#         buf = io.BytesIO()
#         sr_img.save(buf, format="PNG")
#         st.download_button(
#             "📥 Download Enhanced Image",
#             buf.getvalue(),
#             file_name="srgan_output.png",
#             mime="image/png"
#         )

# # 6) Settings
# elif page == "Settings":
#     st.subheader("⚙️ Settings")
#     st.write("**Model path:**", MODEL_PATH)
#     st.write("**Running on device:**", DEVICE)
#     st.markdown("> _Note: Only a 4× model is currently supported._")

# app.py
import os
# Disable the complex module watcher before importing streamlit
# os.environ["STREAMLIT_WATCHER_SIMPLE"] = "true"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


import streamlit as st

st.set_page_config(page_title="✨ SRGAN Super-Resolution", layout="wide")
st.title("✨ SRGAN Super-Resolution Application")
st.markdown(
    """
    Welcome to the SRGAN Super-Resolution Application!  
    Use the sidebar to navigate:
    - **Home**: Overview & academic details  
    - **Enhance**: Upload & super-resolve images  
    - **Settings**: View model/device info  
    """,
    unsafe_allow_html=True,
)
