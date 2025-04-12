import torch
import numpy as np
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import torch

def normalize_complex_minmax_torch(data):
    """
    Min-max normalizes a complex tensor while preserving real and imaginary parts.

    Parameters:
    - data (torch.Tensor): Complex tensor of shape (...), dtype=torch.complex64 or complex128.

    Returns:
    - torch.Tensor: Min-max normalized complex tensor.
    """
    real_part = data.real
    imag_part = data.imag

    real_min, real_max = real_part.min(), real_part.max()
    imag_min, imag_max = imag_part.min(), imag_part.max()

    real_norm = (real_part - real_min) / (real_max - real_min + 1e-8)
    imag_norm = (imag_part - imag_min) / (imag_max - imag_min + 1e-8)

    return torch.complex(torch.tensor(real_norm), torch.tensor(imag_norm))   # Returns normalized complex tensor

def normalize_complex_minmax(data):
    real_min, real_max = data.real.min(), data.real.max()
    imag_min, imag_max = data.imag.min(), data.imag.max()

    real_norm = (data.real - real_min) / (real_max - real_min)
    imag_norm = (data.imag - imag_min) / (imag_max - imag_min)

    return real_norm + 1j * imag_norm


def nmse(ground_truth, reconstructed):
    return np.linalg.norm(reconstructed - ground_truth) ** 2 / np.linalg.norm(ground_truth) ** 2

def plot_mrsi_spectra(mrsi_fs1, mrsi_us1, mrsi_recons1, i=10, j=10, k=10):
    """
    Plots MRSI spectra for a fixed spatial location (i=10, j=10, k=10) across l=0 to 7.

    Parameters:
    - mrsi_fs1: Ground truth spectrum (22x22x21x96x8)
    - mrsi_us: Undersampled spectrum (22x22x21x96x8)
    - mrsi_recons1: Reconstructed spectrum (22x22x21x96x8)
    """
    i, j, k = i, j, k  # Fixed spatial location
    num_freqs = mrsi_fs1.shape[3]  # Frequency dimension (should be 96)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
    axes = axes.flatten()  # Flatten for easy indexing

    for l in range(8):
        ax = axes[l]
        
        # Extract 1D spectra for the given spatial location and dynamic index l
        fs_spectrum = mrsi_fs1[i, j, k, :, l]
        us_spectrum = mrsi_us1[i, j, k, :, l]
        recons_spectrum = mrsi_recons1[i, j, k, :, l]
        
        # Frequency axis (assuming normalized frequency index)
        freq_axis = np.arange(num_freqs)  # Adjust if actual frequency values are known

        # Plot magnitude spectra
        ax.plot(freq_axis, np.abs(fs_spectrum), label='Ground Truth',  color='blue')
        ax.plot(freq_axis, np.abs(us_spectrum), label='Undersampled',  color='green')
        ax.plot(freq_axis, np.abs(recons_spectrum), label='Reconstructed', color='red')

        #if i==0:
        #ax.set_title(f'Spectrum')
        ax.set_title(f't={l}')
        ax.set_xlabel('Freqs')
        ax.set_ylabel('Magnitude')
        ax.legend()
        #ax.grid(True)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()


def rotate_3d_tensor(tensor_3d_real_fs, tensor_3d_imag_fs, 
                     tensor_3d_real_us, tensor_3d_imag_us, angle=5):

    #tensor_3d_real=tensor_3d_real #N C H W=1,21,22,22
    """
    import torchvision.transforms as transforms

    random_rotate = transforms.RandomRotation(degrees=(-5, 5))  # Rotation range

    # Example: Applying to a single slice
    rotated_slice = random_rotate(tensor_3d_real[:, :, 0])  # Rotate only one slice


    """

    tensor_3d_real_fs=tensor_3d_real_fs.permute(2, 0, 1).unsqueeze(0)
    tensor_3d_imag_fs=tensor_3d_imag_fs.permute(2, 0, 1).unsqueeze(0)

    tensor_3d_real_us=tensor_3d_real_us.permute(2, 0, 1).unsqueeze(0)
    tensor_3d_imag_us=tensor_3d_imag_us.permute(2, 0, 1).unsqueeze(0)
 
    
    rotated_slices_real_fs = torch.zeros_like(tensor_3d_real_fs, device=tensor_3d_real_fs.device)
    rotated_slices_imag_fs = torch.zeros_like(tensor_3d_imag_fs, device=tensor_3d_imag_fs.device)

    
    rotated_slices_real_us = torch.zeros_like(tensor_3d_real_us, device=tensor_3d_real_us.device)
    rotated_slices_imag_us = torch.zeros_like(tensor_3d_imag_us, device=tensor_3d_imag_us.device)

    for i in range(tensor_3d_real_fs.shape[2]):
        rotangle = random.uniform(-angle, angle) 
 
        rotated_slices_real_fs[0,i:i+1,:,:]=TF.rotate(tensor_3d_real_fs[0,i:i+1,:,:], rotangle) 
        rotated_slices_imag_fs[0,i:i+1,:,:]=TF.rotate(tensor_3d_imag_fs[0,i:i+1,:,:], rotangle) 
 
        rotated_slices_real_us[0,i:i+1,:,:]=TF.rotate(tensor_3d_real_us[0,i:i+1,:,:], rotangle) 
        rotated_slices_imag_us[0,i:i+1,:,:]=TF.rotate(tensor_3d_imag_us[0,i:i+1,:,:], rotangle) 
 
   
    rotated_slices_imag_fs=rotated_slices_imag_fs.squeeze().permute(1,2,0)
    rotated_slices_real_fs=rotated_slices_real_fs.squeeze().permute(1,2,0)

    rotated_slices_imag_us=rotated_slices_imag_us.squeeze().permute(1,2,0)
    rotated_slices_real_us=rotated_slices_real_us.squeeze().permute(1,2,0)
    
    #print(rotated_slices_imag.shape)
    #exit()
    return rotated_slices_real_fs, rotated_slices_imag_fs, rotated_slices_real_us, rotated_slices_imag_us

def rotate_3d_tensor2(tensor_3d_real, tensor_3d_imag, angle=5):

   
    tensor_3d_real=tensor_3d_real.permute(2, 0, 1).unsqueeze(0)
    tensor_3d_imag=tensor_3d_imag.permute(2, 0, 1).unsqueeze(0)

    rotated_slices_real = torch.zeros_like(tensor_3d_real, device=tensor_3d_real.device)
    rotated_slices_imag = torch.zeros_like(tensor_3d_imag, device=tensor_3d_imag.device)


    for i in range(tensor_3d_real.shape[2]):
        rotangle = random.uniform(-angle, angle) 
        rotated_slices_real[0,i:i+1,:,:]=TF.rotate(tensor_3d_real[0,i:i+1,:,:], rotangle) 
        rotated_slices_imag[0,i:i+1,:,:]=TF.rotate(tensor_3d_imag[0,i:i+1,:,:], rotangle) 
 
   
    rotated_slices_imag=rotated_slices_imag.squeeze().permute(1,2,0)
    rotated_slices_real=rotated_slices_real.squeeze().permute(1,2,0)
    #print(rotated_slices_imag.shape)
    #exit()
    return rotated_slices_real, rotated_slices_imag


def extract_right_part(vector):
    right_parts = []
    for element in vector:
        right_part = element.split('_')[1]
        right_parts.append(right_part)
    return [int(right) for right in right_parts]


def vector(ind, total=None):
    result_vector = []
    for num in range(total):
        random_index = random.randint(2, ind)
        result_vector.append(f"{num}_{random_index}")
    #print("\nResulting vector:", result_vector)
    return extract_right_part(result_vector)



# Augmentation Functions
def random_rotation(tensor_real, tensor_imag, max_angle=3):
    """Apply small random rotation (±max_angle°) to a 3D volume"""
    angle = np.random.uniform(-max_angle, max_angle)
    return TF.rotate(tensor_real, angle), TF.rotate(tensor_imag, angle)

def random_shift(tensor_real, tensor_imag , max_shift=3):
    """Apply small random translation (±3 pixels)"""
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)

    # Apply padding and crop to shift
    paddedreal = F.pad(tensor_real, (abs(shift_x), abs(shift_x), abs(shift_y), abs(shift_y)), mode='replicate')
    _, H, W = tensor_real.shape
    start_xreal, start_yreal = abs(shift_x) + shift_x, abs(shift_y) + shift_y
    
    paddedimag = F.pad(tensor_imag, (abs(shift_x), abs(shift_x), abs(shift_y), abs(shift_y)), mode='replicate')
    _, H, W = tensor_imag.shape
    start_ximag, start_yimag = abs(shift_x) + shift_x, abs(shift_y) + shift_y
    
    return paddedreal[:, start_yreal:start_yreal + H, start_xreal:start_xreal + W], paddedimag[:, start_yimag:start_yimag + H, start_ximag:start_ximag + W]

def random_flip(tensor):
    """Apply random horizontal/vertical flipping"""
    if torch.rand(1) > 0.5:
        tensor = torch.flip(tensor, dims=[1])  # Flip along height
    if torch.rand(1) > 0.5:
        tensor = torch.flip(tensor, dims=[2])  # Flip along width
    return tensor


def batch_augmentation(batch_data_fs=None, batch_data_us=None, batchsize=None):
  
    #print(batch_data.shape, batch_data.dtype)
    batch_data_fs=batch_data_fs.reshape( 22, 22, 21, 96, 8, 5)
    batch_data_us=batch_data_us.reshape( 22, 22, 21, 96, 8, 5)
    data_aug_fs=torch.zeros(batchsize, 22, 22, 21, dtype=torch.complex64)
    data_aug_us=torch.zeros(batchsize, 22, 22, 21, dtype=torch.complex64) 
    
    """ subject = random.choices(range(5), k=1)[0]
        #subjects = torch.randperm(5)
    t = random.choices(range(96), k=1)[0]
    T = random.choices(range(8), k=1)[0] 
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))  # 1 row, 2 columns

        # Display first image
    axes[0,0].imshow(batch_data_us.real[:, :, 10, t, T,subject].cpu().numpy(), cmap="gray")
    axes[0,0].set_title("US Image")
    axes[0,0].axis("off")

    # Display second image
    axes[0,1].imshow(batch_data_fs.real[:, :, 10, t, T,subject].cpu().numpy(), cmap="gray")
    axes[0,1].set_title("FS Image")
    axes[0,1].axis("off")
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
 """

    for b in range(batchsize): 
        subject = random.choices(range(5), k=1)[0]
        #subjects = torch.randperm(5)
        t = random.choices(range(96), k=1)[0]
        T = random.choices(range(8), k=1)[0]      
        
        volume_real_fs = batch_data_fs[ :, :, :, t, T, subject].real  # Extract (22, 22, 21) volume
        volume_imag_fs = batch_data_fs[ :, :, :, t, T, subject].imag  # Extract (22, 22, 21) volume
        
        volume_real_us = batch_data_us[ :, :, :, t, T, subject].real  # Extract (22, 22, 21) volume
        volume_imag_us = batch_data_us[ :, :, :, t, T, subject].imag  # Extract (22, 22, 21) volume
        
        #volume_real_fs, volume_imag_fs = random_shift(volume_real_fs, volume_imag_fs)
        
        
        #volume_real_us1, volume_imag_us1 = volume_real_us, volume_imag_us
        
        volume_real_fs, volume_imag_fs, volume_real_us, volume_imag_us = \
                    rotate_3d_tensor(volume_real_fs, volume_imag_fs, volume_real_us, volume_imag_us)

        #volume_real_us, volume_imag_us = rotate_3d_tensor(volume_real_us, volume_imag_us)
        
        
        """ for i in range(7):  

            fig, axes = plt.subplots(2, 2, figsize=(10, 5))  # 1 row, 2 columns

            # Display first image
            axes[0,0].imshow(volume_real_us1[:, :, i+5].cpu().numpy(), cmap="gray")
            axes[0,0].set_title("US Image real")
            axes[0,0].axis("off")

            # Display second image
            axes[0,1].imshow(volume_imag_us1[:, :, i+5].cpu().numpy(), cmap="gray")
            axes[0,1].set_title("US Image imag")
            axes[0,1].axis("off")

            axes[1,0].imshow(volume_real_us[:, :, i+5].cpu().numpy(), cmap="gray")
            #axes[1,0].set_title("US Image")
            axes[1,0].axis("off")

            # Display second image
            axes[1,1].imshow(volume_imag_us[:, :, i+5].cpu().numpy(), cmap="gray")
            #axes[1,1].set_title("FS Image")
            axes[1,1].axis("off")


            plt.tight_layout()  # Adjust layout for better spacing
            plt.show()
        exit()
         """    
        #volume_real_us, volume_imag_us = random_shift(volume_real_us, volume_imag_us)
        
        data_aug_fs[b,...]=torch.complex(volume_real_fs, volume_imag_fs )
        data_aug_us[b,...]=torch.complex(volume_real_us, volume_imag_us )
        #volume = random_flip(volume)
        
    return data_aug_fs, data_aug_us


def batch_augmentation2(batch_data_fs=None, batch_data_us=None, batchsize=None):
  
    data_aug_fs=torch.zeros(batchsize, 22, 22, 21, dtype=torch.complex64)
    data_aug_us=torch.zeros(batchsize, 22, 22, 21, dtype=torch.complex64) 
    
    for b in range(batchsize): 
        subject = random.choices(range(5), k=1)[0]
        
        volume_real_fs = batch_data_fs[ ..., subject].real  # Extract (22, 22, 21) volume
        volume_imag_fs = batch_data_fs[ ..., subject].imag  # Extract (22, 22, 21) volume
        
        volume_real_us = batch_data_us[ ..., subject].real  # Extract (22, 22, 21) volume
        volume_imag_us = batch_data_us[ ..., subject].imag  # Extract (22, 22, 21) volume
        
        #print(volume_imag_us.shape)
        volume_real_fs, volume_imag_fs = rotate_3d_tensor2(volume_real_fs, volume_imag_fs)

        volume_real_us, volume_imag_us = rotate_3d_tensor2(volume_real_us, volume_imag_us)
        
        
        """ for i in range(7):  

            fig, axes = plt.subplots(2, 2, figsize=(10, 5))  # 1 row, 2 columns

            # Display first image
            axes[0,0].imshow(volume_real_us1[:, :, i+5].cpu().numpy(), cmap="gray")
            axes[0,0].set_title("US Image real")
            axes[0,0].axis("off")

            # Display second image
            axes[0,1].imshow(volume_imag_us1[:, :, i+5].cpu().numpy(), cmap="gray")
            axes[0,1].set_title("US Image imag")
            axes[0,1].axis("off")

            axes[1,0].imshow(volume_real_us[:, :, i+5].cpu().numpy(), cmap="gray")
            #axes[1,0].set_title("US Image")
            axes[1,0].axis("off")

            # Display second image
            axes[1,1].imshow(volume_imag_us[:, :, i+5].cpu().numpy(), cmap="gray")
            #axes[1,1].set_title("FS Image")
            axes[1,1].axis("off")


            plt.tight_layout()  # Adjust layout for better spacing
            plt.show()
        exit()
         """    
        #volume_real_us, volume_imag_us = random_shift(volume_real_us, volume_imag_us)
        
        data_aug_fs[b,...]=torch.complex(volume_real_fs, volume_imag_fs )
        data_aug_us[b,...]=torch.complex(volume_real_us, volume_imag_us )
        #volume = random_flip(volume)
        
    return data_aug_fs, data_aug_us

# Initialize random tensor with shape (22, 22, 21, 96, 8, 5)
#data = torch.randn(22, 22, 21, 96, 8, 5)

# Define batch size (choose subjects randomly)
#batch_size = 8  # Change as needed

def mrsi_train2(mrsi_fs=None,  mrsi_us=None, batchsize=None, device=None, snr=None, tN=None ):
    #print(mrsi_fs.shape, mrsi_fs.dtype) [22, 22, 21, 96, 8, 5]
    batch_fs, batch_us=batch_augmentation2(mrsi_fs, mrsi_us, batchsize=batchsize)
    batch_us=batch_us.to(torch.complex64)
    
    yield batch_fs, batch_us


def mrsi_train(mrsi_fs=None,  mrsi_us=None, batchsize=None, device=None, snr=None, tN=None ):
    #print(mrsi_fs.shape, mrsi_fs.dtype)

    batch_fs, batch_us=batch_augmentation(mrsi_fs, mrsi_us, batchsize=batchsize)
    batch_us=batch_us.to(torch.complex64)
    
    yield batch_fs, batch_us    