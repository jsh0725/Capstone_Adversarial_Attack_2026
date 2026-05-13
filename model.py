import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class MobileUNetGenerator(nn.Module):
    def __init__(self):
        super(MobileUNetGenerator, self).__init__()
        self.enc1 = self.conv_block(3, 32) # RGB 3채널
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)
        self.final = nn.Sequential(nn.Conv2d(32, 3, kernel_size=1), nn.Tanh())

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2, mode='bilinear'), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear'), e1], dim=1))
        # Tanh ∈ (-1,1) → scale to max noise budget; train.py clamps per G_v / G_f
        return self.final(d1) * max(config.EPSILON_V, config.EPSILON_F)