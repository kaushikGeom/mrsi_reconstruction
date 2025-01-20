import torch
import torch.nn as nn

# Define the complex 3D CNN model
class MRSI3D(nn.Module):
    def __init__(self):
        super(MRSI3D, self).__init__()
        
        # Define the 3D Convolutional layers for the real part and imaginary part
        self.conv1_real = nn.Conv3d(in_channels=768, out_channels=128, kernel_size=3, padding=1)  # Real part input
        self.conv1_imag = nn.Conv3d(in_channels=768, out_channels=128, kernel_size=3, padding=1)  # Imaginary part input
        
        self.conv2_real = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)   # Output channels: 64
        self.conv2_imag = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)   # Output channels: 64
        
        self.conv3_real = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)    # Output channels: 32
        self.conv3_imag = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)    # Output channels: 32
        
        # Define the output convolutional layer for both real and imaginary parts
        self.conv4_real = nn.Conv3d(in_channels=32, out_channels=768, kernel_size=3, padding=1)  # Output channels: 768 (real part)
        self.conv4_imag = nn.Conv3d(in_channels=32, out_channels=768, kernel_size=3, padding=1)  # Output channels: 768 (imaginary part)
        
        # Batch Normalization layers for intermediate convolutions
        self.bn_real_1 = nn.BatchNorm3d(128)
        self.bn_imag_1 = nn.BatchNorm3d(128)
        
        self.bn_real_2 = nn.BatchNorm3d(64)
        self.bn_imag_2 = nn.BatchNorm3d(64)
        
        self.bn_real_3 = nn.BatchNorm3d(32)
        self.bn_imag_3 = nn.BatchNorm3d(32)

        # Activation function (used in intermediate layers only)
        self.relu = nn.ReLU()

    def forward(self, real_input, imag_input):
        # Convolutions for the real part
        real_out = self.bn_real_1(self.conv1_real(real_input))  # BatchNorm before ReLU
        imag_out = self.bn_imag_1(self.conv1_imag(imag_input))  # BatchNorm before ReLU
        real_out = self.relu(real_out)
        imag_out = self.relu(imag_out)

        # Convolutions for the imaginary part
        real_out = self.bn_real_2(self.conv2_real(real_out))  # BatchNorm before ReLU
        imag_out = self.bn_imag_2(self.conv2_imag(imag_out))  # BatchNorm before ReLU
        real_out = self.relu(real_out)
        imag_out = self.relu(imag_out)

        # Convolutions for the final feature extraction
        real_out = self.bn_real_3(self.conv3_real(real_out))  # BatchNorm before ReLU
        imag_out = self.bn_imag_3(self.conv3_imag(imag_out))  # BatchNorm before ReLU
        real_out = self.relu(real_out)
        imag_out = self.relu(imag_out)
        
        # Final convolution to reconstruct both real and imaginary parts
        real_output = self.conv4_real(real_out)
        imag_output = self.conv4_imag(imag_out)
        
        # Combine the real and imaginary parts
        output = torch.complex(real_output, imag_output)
        
        return output


""" # Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = MRSI3D().to(device)

# Example input tensor with shape (batch_size=10, 96, 8, 22, 22, 21)
# Separate real and imaginary parts
input_real = torch.randn(16, 96 * 8, 22, 22, 21).to(device)  # Real part
input_imag = torch.randn(16, 96 * 8, 22, 22, 21).to(device)  # Imaginary part

# Forward pass
output_tensor = model(input_real, input_imag)

# Print output shape
print("Output Tensor Shape:", output_tensor.shape)  # Should match (10, 768, 22, 22, 21)
 """