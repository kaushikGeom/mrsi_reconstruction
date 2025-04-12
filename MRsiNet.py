import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import math


class MRSI3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channels: 2 (real + imaginary)
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 2, kernel_size=3, padding=1)  # No activation for complex output
        )

    def forward(self, real, imag):
        """
        real: Tensor of shape [batch, 22, 22, 21] (real part)
        imag: Tensor of shape [batch, 22, 22, 21] (imaginary part)
        """
        x_in = torch.stack([real, imag], dim=1)  # Shape: [batch, 2, 22, 22, 21]
        out = self.encoder(x_in)  # Output: [batch, 2, 22, 22, 21]
        out_real = out[:, 0, :, :, :]  # First channel as real part
        out_imag = out[:, 1, :, :, :]  # Second channel as imaginary part

        return torch.complex(out_real, out_imag)  # Convert back to complex tensor

class MRSI3D_WB(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channels: 2 (real + imaginary)
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 2, kernel_size=3, padding=1)
        )
        

    def forward(self, real, imag):
        """
        real: Tensor of shape [batch, 22, 22, 21] (real part)
        imag: Tensor of shape [batch, 22, 22, 21] (imaginary part)
        """
        x_in = torch.stack([real, imag], dim=1)  # Shape: [batch, 2, 22, 22, 21]
        out = self.encoder(x_in)  # Output: [batch, 2, 22, 22, 21]
        out = self.decoder(out)  # Output: [batch, 2, 22, 22, 21]
        
        out_real = out[:, 0, :, :, :]  # First channel as real part
        out_imag = out[:, 1, :, :, :]  # Second channel as imaginary part

        return torch.complex(out_real, out_imag)  # Convert back to complex tensor


# Example usage

""" batch_size = 100
real_part = torch.randn(batch_size, 22, 22, 21, dtype=torch.float32)  # Real part
imag_part = torch.randn(batch_size, 22, 22, 21, dtype=torch.float32)  # Imaginary part

model = MRSI3D_WB()
output = model(real_part, imag_part)

print("Output shape:", output.shape)  # Should be [batch, 22, 22, 21] complex
 """



