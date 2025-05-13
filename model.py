# model.py

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=8, upscale_factor=4):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )

        res_blocks = [ResidualBlock(64) for _ in range(num_residual_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        upsample_layers = []
        for _ in range(upscale_factor // 2):
            upsample_layers += [
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        self.final = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        initial_out = self.initial(x)
        res_out = self.res_blocks(initial_out)
        mid = self.mid_conv(res_out)
        combined = initial_out + mid
        upsampled = self.upsample(combined)
        return self.final(upsampled)

# ✅ Use this to load the model from weights
def load_generator(weights_path="generator_weights.pth", device="cpu"):
    model = Generator(num_residual_blocks=8, upscale_factor=4)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)

# Optional device constant
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
