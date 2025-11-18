import torch
import torch.nn as nn 

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        
        # Layer 1: No InstanceNorm
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)
        )
        
        # Layer 2
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        )
        self.norm1 = nn.InstanceNorm2d(128)
        
        # Layer 3
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        )
        self.norm2 = nn.InstanceNorm2d(256)
        
        # Layer 4
        self.conv4 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)
        )
        self.norm3 = nn.InstanceNorm2d(512)
        
        # Layer 5: Output layer, no norm, no activation
        self.conv5 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)
        )
        
        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x):
        # Layer 1
        out = self.conv1(x)
        out = self.leaky_relu(out)
        
        # Layer 2
        out = self.conv2(out)
        out = self.norm1(out)
        out = self.leaky_relu(out)
        
        # Layer 3
        out = self.conv3(out)
        out = self.norm2(out)
        out = self.leaky_relu(out)
        
        # Layer 4
        out = self.conv4(out)
        out = self.norm3(out)
        out = self.leaky_relu(out)
        
        # Layer 5 (output)
        out = self.conv5(out)
        
        return out