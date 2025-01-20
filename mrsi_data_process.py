import numpy as np
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load the .npy file
fs_path = "C:\\Users\\Sumit\\Desktop\\Hauke_MRSI_Data\\fulldata_mrsi\\Ground_Truth.npy"  # Replace with your .npy file path
numpy_fs = np.load(fs_path)
us_path = "C:\\Users\\Sumit\\Desktop\\Hauke_MRSI_Data\\fulldata_mrsi\\Undersampled_AF_3.npy"  # Replace with your .npy file path
numpy_us = np.load(us_path)

tensor_fs=torch.save(numpy_fs, 'mrsi_fs.pt')
tensor_us=torch.save(numpy_us, 'mrsi_us.pt')

exit()
tensor_fs=torch.load('mrsi_fs.pt')
tensor_us=torch.load('mrsi_us.pt')

# Convert the NumPy array to a PyTorch tensor
tensor_fs = torch.abs(torch.tensor(tensor_fs))
tensor_us = torch.abs(torch.tensor(tensor_fs))

# Plot the images side by side
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

for i in range(15):
    
    image_fs= tensor_fs[:,:, 10, 0+i, 6, 0].numpy()
    image_us= tensor_us[:,:, 10, 0+i, 6, 0].numpy()


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

