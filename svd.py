import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
# Load the .npy file

def low_rank(data, rank):
     """
     Computes a low-rank decomposition of a tensor with shape (22, 22,
     21, 96, 8)
     using truncated SVD.

     Args:
         data (np.ndarray): Numpy array of shape (x, y, z, t, T).
         rank (int): The number of singular values to keep (final rank).

     Returns:
         np.ndarray: The reconstructed tensor with rank 'rank'.
     """

     # Unpack dimensions
     x, y, z, t, T = data.shape

     # Reshape the 5D tensor into a 2D matrix of shape (x*y*z, t*T)
     # Use 'F' (Fortran) order to match MATLAB's column-major ordering
     reshaped_matrix = data.reshape((x * y * z * T, t), order='F')

     # Perform economy-size SVD (similar to MATLAB's "svd(..., 'econ')")
     U, singular_values, Vh = np.linalg.svd(reshaped_matrix,
     full_matrices=False)

     # Truncate the singular values to the desired rank
     k = min(rank, len(singular_values))  # safeguard: rank cannot exceed
# of singular values
     singular_values_truncated = np.zeros_like(singular_values)
     singular_values_truncated[:k] = singular_values[:k]

     # Form the diagonal matrix of truncated singular values
     S_truncated = np.diag(singular_values_truncated)

     # Reconstruct the matrix using the truncated SVD components
     reconstructed_matrix = U @ S_truncated @ Vh

     # Reshape back to the original 5D shape, again using 'F' order
     reconstructed_tensor = reconstructed_matrix.reshape((x, y, z, t, T), order='F')

     return reconstructed_tensor

mask_path="C:\\Users\\Sumit\\Desktop\\Hauke_MRSI_Data\\masks.npy"
numpy_mask = np.load(mask_path)

fs_path = "C:\\Users\\Sumit\\Desktop\\Hauke_MRSI_Data\\Full_Rank_All.npy"  
numpy_fs = np.load(fs_path)
numpy_fs_lr=np.zeros_like(numpy_fs)

#if np.isnan(numpy_fs).any() or np.isinf(numpy_fs).any():
 #   print("Matrix contains NaN or Inf values!")

#numpy_fs = np.nan_to_num(numpy_fs)

for i in range(6):
   numpy_fs_lr[...,i]=low_rank(numpy_fs[...,i], 8)


us_path="C:\\Users\\Sumit\\Desktop\\Hauke_MRSI_Data\\Undersampled_AF5.npy"
numpy_us = np.load(us_path)
numpy_us_lr=np.zeros_like(numpy_us)

for i in range(6):

  numpy_us_lr[...,i]=low_rank(numpy_us[...,i], 8)

torch.save(numpy_fs_lr, 'lowranked_mrsi_fs.pt')
torch.save(numpy_us_lr, 'lowranked_mrsi_us.pt')


