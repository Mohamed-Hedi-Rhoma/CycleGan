import torch 
import torch.nn as nn


class Resblock(nn.Module) : 
    def __init__(self, n_chan = 256, dropout = 0.2, angle_embd = 256):
        super(Resblock,self).__init__()
        self.conv1 = nn.Conv2d(n_chan, n_chan, kernel_size=3, stride=1, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(
            num_features=n_chan,      
            affine=False,         
            track_running_stats=False  
        )
        self.film1 = nn.Linear(angle_embd, n_chan * 2)
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        
        self.conv2 = nn.Conv2d(n_chan, n_chan, kernel_size=3, stride=1, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(
            num_features=n_chan,      
            affine=False,          
            track_running_stats=False  
        )
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.film2 = nn.Linear(angle_embd, n_chan * 2)
        
    def forward(self, x, angles): 
        out = self.conv1(x)
        out = self.instance_norm1(out)
        films_params = self.film1(angles)
        gamma, beta = films_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        out = gamma * out + beta
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.instance_norm2(out)
        
        film_params = self.film2(angles)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        out = gamma * out + beta
        out = torch.relu(out)
        
        return out + x 
    

class Generator_S2L(nn.Module):
    """Generator: Sentinel (384x384) → Landsat (128x128)"""
    def __init__(self, in_channels=6, n_angles=4):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding=3)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.norm1 = nn.InstanceNorm2d(num_features=64)
        
        # PixelUnshuffle 3x downsampling
        self.pixelunshuffle = nn.PixelUnshuffle(downscale_factor=3)
        
        self.conv2 = nn.Conv2d(in_channels=576, out_channels=256, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.norm2 = nn.InstanceNorm2d(num_features=256)
        
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        self.norm3 = nn.InstanceNorm2d(num_features=256)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        self.norm4 = nn.InstanceNorm2d(num_features=256)

        # Angle MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_angles, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # ResBlocks with FiLM
        self.resblocks = nn.ModuleList([
            Resblock(n_chan=256, angle_embd=256) for _ in range(9)
        ])

        # Decoder
        self.transposeconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        nn.init.kaiming_normal_(self.transposeconv1.weight, nonlinearity='relu')
        self.norm5 = nn.InstanceNorm2d(num_features=128)
        
        self.transposeconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        nn.init.kaiming_normal_(self.transposeconv2.weight, nonlinearity='relu')
        self.norm6 = nn.InstanceNorm2d(num_features=64)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')
        self.norm7 = nn.InstanceNorm2d(num_features=64)
        
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='relu')
        self.norm8 = nn.InstanceNorm2d(num_features=32)
        
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=6, kernel_size=7, padding=3, stride=1)
        nn.init.kaiming_normal_(self.conv7.weight, nonlinearity='tanh')

    def forward(self, x, angles):
        # x: [B, 6, 384, 384]
        # angles: [B, 4]
        
        # Process angles
        angles_embd = self.mlp(angles)  # [B, 256]
        
        # Encoder
        out = self.conv1(x)  # [B, 64, 384, 384]
        out = self.norm1(out)
        out = torch.relu(out)
        
        out = self.pixelunshuffle(out)  # [B, 576, 128, 128]
        
        out = self.conv2(out)  # [B, 256, 128, 128]
        out = self.norm2(out)
        out = torch.relu(out)
        
        out = self.conv3(out)  # [B, 256, 64, 64]
        out = self.norm3(out)
        out = torch.relu(out)
        
        out = self.conv4(out)  # [B, 256, 32, 32]
        out = self.norm4(out)
        out = torch.relu(out)

        # ResBlocks with FiLM
        for resblock in self.resblocks:
            out = resblock(out, angles_embd)  # [B, 256, 32, 32]

        # Decoder
        out = self.transposeconv1(out)  # [B, 128, 64, 64]
        out = self.norm5(out)
        out = torch.relu(out)

        out = self.transposeconv2(out)  # [B, 64, 128, 128]
        out = self.norm6(out)
        out = torch.relu(out)

        out = self.conv5(out)  # [B, 64, 128, 128]
        out = self.norm7(out)
        out = torch.relu(out)

        out = self.conv6(out)  # [B, 32, 128, 128]
        out = self.norm8(out)
        out = torch.relu(out)

        out = self.conv7(out)  # [B, 6, 128, 128]
        out = torch.tanh(out)

        return out