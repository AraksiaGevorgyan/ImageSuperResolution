import torch
import torch.nn as nn
import streamlit as st

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
            nn.Conv2d(3,64,9,padding=4), nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.mid_conv   = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64))
        up_layers = []
        for _ in range(upscale_factor//2):
            up_layers += [nn.Conv2d(64,256,3,padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
        self.upsample = nn.Sequential(*up_layers)
        self.final    = nn.Conv2d(64,3,9,padding=4)

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.res_blocks(x1)
        x3 = self.mid_conv(x2)
        x = x1 + x3
        x = self.upsample(x)
        return self.final(x)

@st.cache_resource
def load_generator(path: str):
    G = Generator(num_residual_blocks=16, upscale_factor=4).to(DEVICE)
    sd = torch.load(path, map_location=DEVICE)
    G.load_state_dict(sd, strict=False)
    G.eval()
    return G
