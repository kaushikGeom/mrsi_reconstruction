import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt
from mrsi_utils import normalize_complex_minmax_torch

""" # Load the .npy file
fs_path="C:\\Users\\Sumit\\Desktop\\Hauke_MRSI_Data\\Full_Rank_All.npy"
numpy_fs = np.load(fs_path)

us_path="C:\\Users\\Sumit\\Desktop\\Hauke_MRSI_Data\\Undersampled_AF5.npy"
numpy_us = np.load(us_path)

mask_path="C:\\Users\\Sumit\\Desktop\\Hauke_MRSI_Data\\masks.npy"
numpy_mask = np.load(mask_path)
 """
#exit()
tensor_fs=torch.load('lowranked_mrsi_fs.pt')
tensor_us=torch.load('lowranked_mrsi_us.pt')
tensor_mask=torch.load('numpy_mask.pt')

mrsi_fs=normalize_complex_minmax_torch(tensor_fs)
mrsi_us=normalize_complex_minmax_torch(tensor_us)

mrsi_fs=mrsi_fs*tensor_mask[:,:,:, None, None,:]
mrsi_us=mrsi_us*tensor_mask[:,:,:, None, None,:]

torch.save(mrsi_fs, 'mask_norm_lowranked_mrsi_fs.pt')
torch.save(mrsi_us, 'mask_norm_lowranked_mrsi_us.pt')




# Convert the NumPy array to a PyTorch tensor
tensor_fs = torch.abs(torch.tensor(tensor_fs))
tensor_us = torch.abs(torch.tensor(tensor_us))

patient=4

# Plot the images side by side
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
for i in range(15):
    
    image_fs= tensor_fs[:,:, 10, 0+i, patient, 0].numpy()
    image_us= tensor_us[:,:, 10, 0+i, patient, 0].numpy()


    # Display Image 1
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(image_fs, cmap='gray')  # Use a grayscale colormap
    plt.title("FullySampled")
    plt.axis('off')  # Hide axes for better visualization

    # Display Image 2
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.imshow(image_us, cmap='gray')  # Use a grayscale colormap
    plt.title("UnderSampled")
    plt.axis('off')  # Hide axes for better visualization

    # Show the images
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

