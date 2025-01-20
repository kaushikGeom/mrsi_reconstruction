import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from tqdm import tqdm    
from MRsiNet import MRSI3D
import torch
from torchmetrics import PeakSignalNoiseRatio 
from torchmetrics import StructuralSimilarityIndexMeasure as ssim
#from getPhantom_image_from_nii import data_rs as phantom_data
#from getInvivo_image_from_nii import data_rs as invivo_data
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mrsi_fs=torch.tensor(torch.load('mrsi_fs.pt')).to(device)
mrsi_us=torch.tensor(torch.load('mrsi_us.pt')).to(device)
#print(mrsi_fs.shape)
#exit()
mrsi_fs=mrsi_fs[...,4:5]
mrsi_us=mrsi_us[...,4:5]


model=MRSI3D().to(device)
batchsize=1

snr=None
modelname="mrsi_bs8_bpe50_snrNone_lr_0.001"
modelname="abs_mrsi_bs8_bpe50_snrNone_lr_0.001"
model.load_state_dict(torch.load(os.path.join("mrsi_models//",  modelname + '.pt')))

mrsi_us_original=mrsi_us
model.eval() 
with torch.inference_mode(): 

            print(mrsi_us.shape)
            mrsi_us=mrsi_us.permute(5, 4, 3,  0, 1, 2)
            mrsi_us=mrsi_us.reshape(1, 96*8, 22, 22, 21)  #8, 768, 22, 22, 21
            print(mrsi_us.shape)
            #exit()
            batch_us_real = mrsi_us.real.to(dtype=torch.float32)
            batch_us_imag = mrsi_us.imag.to(dtype=torch.float32)
            
            mrsi_recons = model(batch_us_real, batch_us_imag)
            mrsi_recons=mrsi_recons.reshape(batchsize, 96, 8, 22, 22, 21 )
            mrsi_recons=mrsi_recons.permute(3,4,5, 1,2, 0).abs().cpu().numpy()
            print(mrsi_recons.shape)
            print(mrsi_fs.shape)
            print(mrsi_us_original.shape)

mrsi_us=mrsi_us_original.abs().cpu().numpy()
mrsi_fs=mrsi_fs.abs().cpu().numpy()


# Parameters
slice_index = 9  # 10th slice (zero-based indexing)
selected_indices = range(8)  # Select first 8 images

# Extract the slice and reshape to (22, 21)
def extract_images(data, slice_idx, selected_indices):
    images = []
    for idx in selected_indices:
        image = data[:, :, slice_idx, idx, idx, 0]  # Slice and reshape
        images.append(image)
    return images

# Extract images for each dataset
us_images = extract_images(mrsi_us, slice_index, selected_indices)
fs_images = extract_images(mrsi_fs, slice_index, selected_indices)
recons_images = extract_images(mrsi_recons, slice_index, selected_indices)

# Create a grid of 3x8 images
fig, axes = plt.subplots(3, 8, figsize=(16, 6))

for i, ax_row in enumerate(axes):
    images = [us_images, fs_images, recons_images][i]  # Select the dataset for the row
    for j, ax in enumerate(ax_row):
        ax.imshow(images[j] )  # Display image
        ax.axis('off')
        if i == 0:
            ax.set_title(f"Image {j+1}")

# Set overall plot title and adjust layout
plt.suptitle("(Slice 10)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
